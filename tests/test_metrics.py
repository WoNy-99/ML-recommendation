import math

import pytest

from src.metrics import ndcg_at_k, recall_at_k

LOG3 = math.log2(3)


def test_no_targets_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 10)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], [], 10)


def test_duplicate_recommendations_are_rejected():
    with pytest.raises(ValueError):
        ndcg_at_k(["a", "a"], ["a"], 10)


def test_two_targets_at_rank_1_and_2_give_ndcg_1():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 10) == pytest.approx(1.0)
    assert recall_at_k(["a", "b", "x"], ["a", "b"], 10) == pytest.approx(1.0)


def test_multiple_targets_partial_hits():
    # 정답 a,b,c 중 a(1위), b(3위)만 추천
    ranked, targets = ["a", "x", "b"], ["a", "b", "c"]
    assert recall_at_k(ranked, targets, 10) == pytest.approx(2 / 3)
    idcg = 1 + 1 / LOG3 + 1 / 2  # m = min(3, 10) = 3
    assert ndcg_at_k(ranked, targets, 10) == pytest.approx((1 + 1 / 2) / idcg)


def test_rank_change_lowers_ndcg_but_not_recall():
    assert ndcg_at_k(["a", "x"], ["a"], 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], ["a"], 10) == pytest.approx(1 / LOG3)
    assert recall_at_k(["x", "a"], ["a"], 10) == pytest.approx(1.0)


def test_hit_outside_k_does_not_count():
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert recall_at_k(ranked, ["a"], 10) == 0.0
    assert ndcg_at_k(ranked, ["a"], 10) == 0.0


def test_short_recommendation_list():
    assert recall_at_k(["x"], ["a"], 10) == 0.0
    assert ndcg_at_k(["a"], ["a"], 10) == pytest.approx(1.0)
    assert ndcg_at_k([], ["a"], 10) == 0.0


def test_more_targets_than_k_uses_m_equal_k():
    targets = [f"t{i}" for i in range(12)]
    assert recall_at_k(targets, targets, 10) == pytest.approx(10 / 12)
    assert ndcg_at_k(targets, targets, 10) == pytest.approx(1.0)
