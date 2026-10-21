"""로컬 후보 필터와 질의 TF-IDF 순위. B1 구현(TfidfRecommender, rank_with_exclusion)을 그대로 쓴다.

- TF-IDF 어휘·문서는 B1과 같은 학습 구간(date < t_val) 리뷰로 만든다.
- 사용자 질문(+해석된 조건의 영어 용어)을 같은 TF-IDF 공간으로 변환해 코사인 유사도로 순위를 매긴다.
- 후보 목록·점수·결과는 로컬에만 남기고 외부로 보내지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import TfidfRecommender, build_catalog, rank_with_exclusion
from src.config import parse_day, read_config, text_sha256


@dataclass
class LocalIndex:
    model: TfidfRecommender
    catalog: pd.DataFrame          # model.business_ids 와 같은 순서
    info: dict = field(default_factory=dict)


def build_index(history: pd.DataFrame, businesses: pd.DataFrame, tfidf: dict,
                positive_min_stars: int = 4) -> LocalIndex:
    catalog = build_catalog(history, businesses)
    model = TfidfRecommender(
        catalog, history, positive_min_stars, tfidf["max_features"],
        tuple(tfidf["ngram_range"]), tfidf["min_df"], tfidf["stop_words"],
    )
    return LocalIndex(model=model, catalog=catalog, info={
        "catalog_size": int(len(catalog)), "history_reviews": int(len(history)),
        "vocabulary_size": len(model.vectorizer.vocabulary_),
    })


def load_local_index(baseline_config_path: str) -> LocalIndex:
    """B1 설정·준비 데이터(해시 검증 포함)로 학습 구간 TF-IDF 인덱스를 만든다. 수 분 걸릴 수 있다."""
    from src.run_baseline import load_prepared

    cfg, raw = read_config(baseline_config_path)
    t_val = parse_day(cfg["split"]["t_val"], "split.t_val")
    businesses, reviews, manifest = load_prepared(Path(cfg["data"]["output_dir"]))
    history = reviews[reviews["date"] < t_val]
    index = build_index(history, businesses, cfg["tfidf"], cfg["eval"]["positive_min_stars"])
    index.info.update({
        "baseline_config_sha256": text_sha256(raw),
        "t_val": str(t_val.date()),
        "data_reviews_sha256": manifest["files"]["reviews.jsonl.gz"]["sha256"],
    })
    return index


def candidate_mask(catalog: pd.DataFrame, cities: list[str] | None,
                   categories: list[str] | None) -> np.ndarray:
    mask = np.ones(len(catalog), dtype=bool)
    if cities is not None:
        wanted = {c.lower() for c in cities}
        mask &= catalog["city"].fillna("").str.strip().str.lower().isin(wanted).to_numpy()
    if categories is not None:
        wanted_cats = {c.lower() for c in categories}
        has = catalog["categories"].fillna("").map(
            lambda s: bool(wanted_cats & {c.strip().lower() for c in s.split(",")})
        )
        mask &= has.to_numpy()
    return mask


def recommend(index: LocalIndex, query_text: str, cities: list[str] | None,
              categories: list[str] | None, k: int) -> dict:
    mask = candidate_mask(index.catalog, cities, categories)
    n_candidates = int(mask.sum())
    base = {"n_candidates": n_candidates, "query_text_sha256": text_sha256(query_text)}
    if n_candidates == 0:
        return {**base, "status": "no_candidates", "items": []}
    qvec = index.model.vectorizer.transform([query_text])
    if qvec.nnz == 0:
        return {**base, "status": "empty_query_vector", "items": [],
                "message": "no query word is in the TF-IDF vocabulary"}
    scores = np.asarray((index.model.item_matrix @ qvec.T).todense()).ravel()
    excluded = np.flatnonzero(~mask).tolist()
    top = rank_with_exclusion(scores, excluded, k)
    items = [{
        "rank": rank, "business_id": index.catalog.at[i, "business_id"],
        "name": index.catalog.at[i, "name"], "city": index.catalog.at[i, "city"],
        "categories": index.catalog.at[i, "categories"], "score": float(scores[i]),
    } for rank, i in enumerate(top, start=1)]
    return {**base, "status": "recommended", "items": items,
            "all_top_scores_zero": all(item["score"] == 0.0 for item in items)}
