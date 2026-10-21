"""사람 라벨 기준 속성별 macro-F1·혼동행렬과 상태(실패·보류·미호출) 집계.

- 라벨 순서는 LABELS로 고정한다. 혼동행렬은 행=사람 라벨, 열=예측.
- 라벨별 F1 = 2TP / (2TP + FP + FN). 분모가 0(정답·예측 모두 없음)이면 0으로 둔다.
- macro-F1은 항상 고정된 4개 라벨 F1의 평균이다. 라벨별 실제 정답 수(support)를 함께 남긴다.
- 표본이 0개면 not_computable.
- OpenJev 상태가 ok가 아닌 항목(api_error·schema_error·abstained·not_called)은 라벨로 바꾸지 않는다.
  OpenJev와 키워드의 직접 비교는 ok 응답 표본에 한정하고, 전체·비교 표본 수와 상태별 수를 함께 남긴다.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from src.aspects import ASPECTS, LABELS

NON_OK_STATUSES = ("api_error", "schema_error", "abstained", "not_called")


def load_labels(path: str | Path) -> dict[int, dict[str, str]]:
    """라벨 CSV를 item 번호 기준으로 읽는다. 빈칸·허용되지 않은 값이 있으면 ValueError(목록 포함).

    UTF-8(BOM 허용)만 읽는다. Excel에서 저장할 때는 'CSV UTF-8' 형식을 써야 한다.
    """
    labels: dict[int, dict[str, str]] = {}
    problems = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            try:
                item = int((row.get("item") or "").strip())
            except ValueError:
                problems.append(f"line {line_no}: item is not a number")
                continue
            if item in labels:
                problems.append(f"line {line_no}: duplicate item {item}")
                continue
            entry = {"review_id": (row.get("review_id") or "").strip()}
            for aspect in ASPECTS:
                value = (row.get(f"{aspect}_label") or "").strip().lower()
                if value not in LABELS:
                    problems.append(f"line {line_no}: {aspect}_label={value!r}")
                entry[aspect] = value
            labels[item] = entry
    if problems:
        raise ValueError(f"{path}: {len(problems)} label problems, e.g. {problems[:5]}")
    return labels


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[index[t]][index[p]] += 1
    return matrix


def per_label_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    out = {}
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        denominator = 2 * tp + fp + fn
        out[label] = 0.0 if denominator == 0 else 2 * tp / denominator
    return out


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if not y_true:
        return {"n": 0, "status": "not_computable"}
    f1 = per_label_f1(y_true, y_pred)
    return {
        "n": len(y_true),
        "macro_f1": sum(f1[label] for label in LABELS) / len(LABELS),
        "per_label_f1": f1,
        "support": {label: sum(1 for t in y_true if t == label) for label in LABELS},
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true),
        "labels": list(LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def summarize(rows: list[dict]) -> dict:
    """rows: {review_id, human{aspect}, keyword{aspect}, openjev_status, openjev{aspect}|None}."""
    statuses = Counter(r["openjev_status"] for r in rows)
    ok_rows = [r for r in rows if r["openjev_status"] == "ok"]
    result = {
        "n_total": len(rows),
        "openjev_status_counts": {s: statuses.get(s, 0) for s in ("ok",) + NON_OK_STATUSES},
        "n_compared": len(ok_rows),
        "compared_ratio": (len(ok_rows) / len(rows)) if rows else None,
        "note": "OpenJev vs keyword comparison uses only rows with an ok OpenJev response",
        "aspects": {},
    }
    for aspect in ASPECTS:
        human_all = [r["human"][aspect] for r in rows]
        keyword_all = [r["keyword"][aspect] for r in rows]
        human_ok = [r["human"][aspect] for r in ok_rows]
        result["aspects"][aspect] = {
            "keyword_all_rows": classification_metrics(human_all, keyword_all),
            "keyword_compared_rows": classification_metrics(
                human_ok, [r["keyword"][aspect] for r in ok_rows]
            ),
            "openjev_compared_rows": classification_metrics(
                human_ok, [r["openjev"][aspect] for r in ok_rows]
            ),
        }
    return result
