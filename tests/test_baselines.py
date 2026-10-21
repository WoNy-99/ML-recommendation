import numpy as np
import pytest

from conftest import T_TEST, T_VAL
from src.baselines import (
    PopularityRecommender, RandomRecommender, TfidfRecommender, build_catalog,
    rank_with_exclusion,
)
from src.metrics import evaluate_recommender
from src.split import build_eval_set, split_reviews


@pytest.fixture
def setup(reviews, businesses):
    history, window = split_reviews(reviews, "validation", T_VAL, T_TEST)
    catalog = build_catalog(history, businesses)
    eval_set = build_eval_set(
        history, window, set(catalog["business_id"]), 4, "visit", None, 42
    )
    return history, catalog, eval_set


def _tfidf(catalog, history):
    return TfidfRecommender(catalog, history, 4, 1000, (1, 1), 1, None)


def test_catalog_has_only_items_with_history_reviews(setup):
    _, catalog, _ = setup
    assert catalog["business_id"].tolist() == ["b1", "b2", "b3", "b4"]  # b5 cold 제외


def test_rank_ties_break_by_business_order_and_skip_excluded():
    assert rank_with_exclusion(np.array([1.0, 2.0, 2.0, 0.5]), [], 3) == [1, 2, 0]
    assert rank_with_exclusion(np.array([1.0, 2.0, 2.0, 0.5]), [1], 10) == [2, 0, 3]


def test_popularity_counts_history_only(setup):
    history, catalog, _ = setup
    model = PopularityRecommender(catalog, history)
    # 이력 고유 사용자 수: b1=2, b2=3, b3=2, b4=1 (평가 구간 b1 리뷰 4건은 반영되지 않아야 함)
    assert model.scores.tolist() == [2.0, 3.0, 2.0, 1.0]
    assert model.recommend("u6", {"b3"}, 10) == ["b2", "b1", "b4"]  # 짧은 목록, 기존 방문 제외


def test_tfidf_vocabulary_excludes_window_and_test_reviews(setup):
    history, catalog, _ = setup
    vocab = _tfidf(catalog, history).vectorizer.vocabulary_
    assert "zzwindowtoken" not in vocab
    assert "zztestperiod" not in vocab
    assert "ramen" in vocab


def test_tfidf_test_stage_sees_window_but_not_test_period(reviews, businesses):
    history, _ = split_reviews(reviews, "test", T_VAL, T_TEST)
    vocab = _tfidf(build_catalog(history, businesses), history).vectorizer.vocabulary_
    assert "zzwindowtoken" in vocab
    assert "zztestperiod" not in vocab


def test_tfidf_excludes_seen_and_handles_empty_profile(setup):
    history, catalog, _ = setup
    model = _tfidf(catalog, history)
    recs = model.recommend("u1", {"b1", "b2"}, 10)
    assert set(recs) == {"b3", "b4"}
    assert model.recommend("u2", {"b2"}, 10) == []  # 긍정 이력 없음 -> 빈 추천
    assert model.empty_profiles == 1


def test_random_is_reproducible_and_excludes_seen(setup):
    _, catalog, _ = setup
    a = RandomRecommender(catalog, 42).recommend("u1", {"b1"}, 2)
    b = RandomRecommender(catalog, 42).recommend("u1", {"b1"}, 2)
    assert a == b and "b1" not in a and len(a) == 2


def test_all_models_use_the_same_eval_users(setup):
    history, catalog, eval_set = setup
    models = [
        PopularityRecommender(catalog, history), _tfidf(catalog, history),
        RandomRecommender(catalog, 42),
    ]
    for model in models:
        result = evaluate_recommender(model, eval_set, 10)
        assert [r["user_id"] for r in result["rows"]] == eval_set.users == ["u1", "u6"]
        assert result["n_users"] == 2


def test_popularity_end_to_end_metric_values(setup):
    history, catalog, eval_set = setup
    result = evaluate_recommender(PopularityRecommender(catalog, history), eval_set, 10)
    # u1: 추천 [b3, b4], 정답 [b3] -> 1.0 / u6: 추천 [b2, b1, b4], 정답 [b2, b4]
    u6_ndcg = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert result["recall_at_10"] == pytest.approx(1.0)
    assert result["ndcg_at_10"] == pytest.approx((1.0 + u6_ndcg) / 2)
