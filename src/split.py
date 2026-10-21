"""전역 시간 분할, 누수 검사, 평가 사용자·정답 규칙.

경계 규칙: date < 경계 는 앞 구간, date >= 경계 는 뒤 구간 (경계 시각과 같은 리뷰는 뒤 구간).
- validation: 이력 = [history_start, t_val), 평가 구간 = [t_val, t_test)
- test:       이력 = [history_start, t_test), 평가 구간 = [t_test, 끝]
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

EXCLUSION_REASONS = (
    "no_history",           # 평가 구간에만 등장 (이력 리뷰 없음)
    "no_positive_history",  # 이력은 있으나 긍정 리뷰(stars >= 기준) 없음 -> TF-IDF 프로필 불가
    "no_relevant_target",   # positive_visit 정의에서 평가 구간 긍정 리뷰 없음
    "revisit_only",         # 평가 구간 음식점이 모두 이력에 있던 곳 (재방문)
    "targets_all_cold",     # 새 음식점이 모두 이력 기간 리뷰가 없는 cold item
    "not_sampled",          # max_eval_users 초과분 (seed 고정 표본에서 빠짐)
)


class LeakageError(AssertionError):
    """학습 입력에 평가 구간 자료가 섞였을 때."""


@dataclass
class EvalSet:
    users: list[str]
    targets: dict[str, list[str]]
    seen: dict[str, set[str]]
    exclusions: dict[str, int]
    dropped_targets: dict[str, int]
    window_users: int
    meta: dict = field(default_factory=dict)


def window_bounds(
    stage: str, t_val: datetime, t_test: datetime
) -> tuple[datetime, datetime | None]:
    if stage == "validation":
        return t_val, t_test
    if stage == "test":
        return t_test, None
    raise ValueError(f"unknown stage {stage!r}")


def split_reviews(
    reviews: pd.DataFrame,
    stage: str,
    t_val: datetime,
    t_test: datetime,
    history_start: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """reviews['date']는 datetime64. (이력, 평가 구간) 두 DataFrame을 돌려준다."""
    window_start, window_end = window_bounds(stage, t_val, t_test)
    dates = reviews["date"]
    in_history = dates < window_start
    if history_start is not None:
        in_history &= dates >= history_start
    in_window = dates >= window_start
    if window_end is not None:
        in_window &= dates < window_end
    return reviews.loc[in_history].copy(), reviews.loc[in_window].copy()


def check_no_leakage(
    history: pd.DataFrame, window: pd.DataFrame, window_start: datetime
) -> dict:
    """이력의 마지막 시각 < 평가 구간 시작, 두 구간의 review_id가 겹치지 않음을 확인한다."""
    history_max = history["date"].max() if len(history) else None
    if history_max is not None and history_max >= window_start:
        raise LeakageError(f"history has reviews at/after window start: {history_max}")
    if len(window) and window["date"].min() < window_start:
        raise LeakageError("window has reviews before window start")
    overlap = set(history["review_id"]) & set(window["review_id"])
    if overlap:
        raise LeakageError(f"{len(overlap)} review_id appear in both history and window")
    return {
        "history_reviews": int(len(history)),
        "window_reviews": int(len(window)),
        "history_date_max": None if history_max is None else str(history_max),
        "window_start": str(window_start),
        "review_id_overlap": 0,
    }


def build_eval_set(
    history: pd.DataFrame,
    window: pd.DataFrame,
    catalog_ids: set[str],
    positive_min_stars: int,
    target_definition: str,
    max_users: int | None,
    seed: int,
) -> EvalSet:
    """모든 모델이 공유하는 평가 사용자·정답·기존 방문 목록을 만든다."""
    if target_definition not in ("visit", "positive_visit"):
        raise ValueError(f"unknown target_definition {target_definition!r}")
    history_items = history.groupby("user_id")["business_id"].agg(set).to_dict()
    positive_users = set(history.loc[history["stars"] >= positive_min_stars, "user_id"])
    relevant = window if target_definition == "visit" else window[
        window["stars"] >= positive_min_stars
    ]
    relevant_items = relevant.groupby("user_id")["business_id"].agg(set).to_dict()

    exclusions: Counter = Counter({reason: 0 for reason in EXCLUSION_REASONS})
    dropped: Counter = Counter({"revisit_targets": 0, "cold_targets": 0})
    eligible: dict[str, list[str]] = {}
    window_users = sorted(window["user_id"].unique())
    for user in window_users:
        if user not in history_items:
            exclusions["no_history"] += 1
            continue
        if user not in positive_users:
            exclusions["no_positive_history"] += 1
            continue
        items = relevant_items.get(user, set())
        if not items:
            exclusions["no_relevant_target"] += 1
            continue
        new_items = items - history_items[user]
        dropped["revisit_targets"] += len(items) - len(new_items)
        if not new_items:
            exclusions["revisit_only"] += 1
            continue
        warm = new_items & catalog_ids
        dropped["cold_targets"] += len(new_items) - len(warm)
        if not warm:
            exclusions["targets_all_cold"] += 1
            continue
        eligible[user] = sorted(warm)

    users = sorted(eligible)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(users), size=max_users, replace=False)
        exclusions["not_sampled"] = len(users) - max_users
        users = sorted(users[i] for i in chosen)

    return EvalSet(
        users=users,
        targets={u: eligible[u] for u in users},
        seen={u: set(history_items[u]) for u in users},
        exclusions=dict(exclusions),
        dropped_targets=dict(dropped),
        window_users=len(window_users),
        meta={
            "target_definition": target_definition,
            "positive_min_stars": positive_min_stars,
            "max_users": max_users,
            "seed": seed,
            "eligible_users": len(eligible),
        },
    )


def eval_manifest(eval_set: EvalSet) -> dict:
    """평가 사용자 목록을 파일로 남기기 위한 JSON 직렬화 형태."""
    return {
        "meta": eval_set.meta,
        "window_users": eval_set.window_users,
        "exclusions": eval_set.exclusions,
        "dropped_targets": eval_set.dropped_targets,
        "users": [{"user_id": u, "targets": eval_set.targets[u]} for u in eval_set.users],
    }


def json_sha256(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
