"""이력 구간 리뷰만으로 만드는 기준 추천 모델 (인기순·TF-IDF·무작위).

공통 규칙
- 후보(catalog): 이력 구간 리뷰가 1개 이상인 대상 음식점, business_id 오름차순.
- 사용자가 이력에서 리뷰한 음식점은 추천에서 제외한다.
- 동점은 business_id 오름차순 (catalog 순서 + 안정 정렬).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_catalog(history: pd.DataFrame, businesses: pd.DataFrame) -> pd.DataFrame:
    reviewed = set(history["business_id"])
    catalog = businesses[businesses["business_id"].isin(reviewed)]
    return catalog.sort_values("business_id").reset_index(drop=True)


def rank_with_exclusion(scores: np.ndarray, excluded: list[int], k: int) -> list[int]:
    """점수 내림차순 상위 k개의 catalog 인덱스. 제외 항목은 빼고, 동점은 인덱스 오름차순."""
    masked = np.asarray(scores, dtype=np.float64).copy()
    if excluded:
        masked[excluded] = -np.inf
    order = np.argsort(-masked, kind="stable")
    return [int(i) for i in order[:k] if np.isfinite(masked[i])]


class _CatalogRecommender:
    def __init__(self, catalog: pd.DataFrame):
        self.business_ids: list[str] = catalog["business_id"].tolist()
        self.index = {b: i for i, b in enumerate(self.business_ids)}

    def _excluded(self, seen: set[str]) -> list[int]:
        return [self.index[b] for b in seen if b in self.index]

    def _to_ids(self, idxs: list[int]) -> list[str]:
        return [self.business_ids[i] for i in idxs]


class PopularityRecommender(_CatalogRecommender):
    """이력 구간에서 그 음식점을 리뷰한 고유 사용자 수."""

    name = "popularity"

    def __init__(self, catalog: pd.DataFrame, history: pd.DataFrame):
        super().__init__(catalog)
        counts = history.groupby("business_id")["user_id"].nunique()
        self.scores = catalog["business_id"].map(counts).fillna(0).to_numpy(dtype=np.float64)

    def recommend(self, user_id: str, seen: set[str], k: int) -> list[str]:
        return self._to_ids(rank_with_exclusion(self.scores, self._excluded(seen), k))


class TfidfRecommender(_CatalogRecommender):
    """기존 개인 담당 모델과 같은 방식: 음식점 문서 TF-IDF, 긍정 이력 평균 프로필, 내적 점수.

    차이: 문서·어휘·IDF·프로필을 모두 이력 구간 리뷰로만 만든다(평가 구간 리뷰 제외).
    """

    name = "tfidf"

    def __init__(
        self,
        catalog: pd.DataFrame,
        history: pd.DataFrame,
        positive_min_stars: int,
        max_features: int,
        ngram_range: tuple[int, int],
        min_df: int,
        stop_words: str | None,
    ):
        super().__init__(catalog)
        review_text = history.groupby("business_id")["text"].agg(
            lambda texts: " ".join(t for t in texts if isinstance(t, str))
        )
        docs = (
            catalog["name"].fillna("").astype(str) + " "
            + catalog["categories"].fillna("").astype(str) + " "
            + catalog["business_id"].map(review_text).fillna("")
        ).tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=min_df,
            stop_words=stop_words, dtype=np.float32,
        )
        self.item_matrix = self.vectorizer.fit_transform(docs)  # 행마다 L2 정규화됨 (norm='l2')
        positive = history[history["stars"] >= positive_min_stars]
        self.user_positive = (
            positive.groupby("user_id")["business_id"].agg(lambda s: sorted(set(s))).to_dict()
        )
        self.fit_documents = len(docs)
        self.empty_profiles = 0

    def profile(self, user_id: str) -> np.ndarray | None:
        idxs = [self.index[b] for b in self.user_positive.get(user_id, []) if b in self.index]
        if not idxs:
            return None
        mean = np.asarray(self.item_matrix[idxs].mean(axis=0)).ravel()
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            return None
        return (mean / norm).astype(np.float32)

    def recommend(self, user_id: str, seen: set[str], k: int) -> list[str]:
        vector = self.profile(user_id)
        if vector is None:
            self.empty_profiles += 1
            return []
        scores = self.item_matrix @ vector
        return self._to_ids(rank_with_exclusion(scores, self._excluded(seen), k))


class RandomRecommender(_CatalogRecommender):
    """사용자별 seed(seed + user_id 해시)로 섞는다. 평가 순서와 무관하게 재현된다."""

    name = "random"

    def __init__(self, catalog: pd.DataFrame, seed: int):
        super().__init__(catalog)
        self.seed = seed

    def recommend(self, user_id: str, seen: set[str], k: int) -> list[str]:
        digest = hashlib.sha256(f"{self.seed}:{user_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        excluded = set(self._excluded(seen))
        order = rng.permutation(len(self.business_ids))
        return self._to_ids([int(i) for i in order if int(i) not in excluded][:k])
