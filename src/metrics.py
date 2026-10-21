"""Recall@K, nDCG@K와 모델 공통 평가 루프.

- Recall@K = |TopK ∩ 정답| / |정답|  (정답이 K개보다 많으면 1에 도달할 수 없다)
- nDCG@K: 이진 관련성. DCG = Σ_{순위 i의 추천이 정답} 1/log2(i+1),
  IDCG = Σ_{i=1..m} 1/log2(i+1), m = min(|정답|, K)
- 정답이 비어 있거나 추천 목록에 중복이 있으면 ValueError (조용히 0점 처리하지 않는다).
- 추천 목록이 K보다 짧으면 있는 만큼만 계산한다.
"""
from __future__ import annotations

import math
import time
from typing import Protocol, Sequence

import numpy as np


def _validate(ranked: Sequence[str], targets: Sequence[str], k: int) -> set[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    target_set = set(targets)
    if not target_set:
        raise ValueError("targets must not be empty")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked list contains duplicate items")
    return target_set


def recall_at_k(ranked: Sequence[str], targets: Sequence[str], k: int) -> float:
    target_set = _validate(ranked, targets, k)
    hits = sum(1 for item in ranked[:k] if item in target_set)
    return hits / len(target_set)


def ndcg_at_k(ranked: Sequence[str], targets: Sequence[str], k: int) -> float:
    target_set = _validate(ranked, targets, k)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, item in enumerate(ranked[:k], start=1)
        if item in target_set
    )
    m = min(len(target_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg


class Recommender(Protocol):
    def recommend(self, user_id: str, seen: set[str], k: int) -> list[str]: ...


def evaluate_recommender(recommender: Recommender, eval_set, k: int) -> dict:
    """eval_set.users 전원을 같은 순서로 평가한다. 빈 추천도 0점으로 포함하고 개수를 따로 센다."""
    rows = []
    latencies_ms = []
    for user in eval_set.users:
        start = time.perf_counter()
        ranked = recommender.recommend(user, eval_set.seen[user], k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        overlap = set(ranked) & eval_set.seen[user]
        if overlap:
            raise ValueError(f"recommender returned already-visited items for {user}")
        targets = eval_set.targets[user]
        rows.append({
            "user_id": user,
            "n_targets": len(targets),
            "n_recommended": len(ranked),
            "recall": recall_at_k(ranked, targets, k),
            "ndcg": ndcg_at_k(ranked, targets, k),
            "top_k": ranked,
        })
    n = len(rows)
    lat = np.array(latencies_ms) if latencies_ms else np.array([np.nan])
    return {
        "n_users": n,
        f"recall_at_{k}": float(np.mean([r["recall"] for r in rows])) if n else None,
        f"ndcg_at_{k}": float(np.mean([r["ndcg"] for r in rows])) if n else None,
        "empty_recommendations": sum(1 for r in rows if r["n_recommended"] == 0),
        "short_recommendations": sum(1 for r in rows if 0 < r["n_recommended"] < k),
        "latency_ms_per_user_p50": float(np.nanpercentile(lat, 50)) if n else None,
        "latency_ms_per_user_p95": float(np.nanpercentile(lat, 95)) if n else None,
        "recommend_seconds_total": float(sum(latencies_ms) / 1000.0),
        "rows": rows,
    }
