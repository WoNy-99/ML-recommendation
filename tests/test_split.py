from datetime import datetime

import pandas as pd
import pytest

from conftest import T_TEST, T_VAL
from src.split import LeakageError, build_eval_set, check_no_leakage, split_reviews


def _validation_split(reviews):
    return split_reviews(reviews, "validation", T_VAL, T_TEST)


def test_boundary_review_goes_to_later_part(reviews):
    history, window = _validation_split(reviews)
    assert "w1" in set(window["review_id"])   # 2021-01-01 00:00:00 == T_VAL
    assert "w1" not in set(history["review_id"])
    assert "h8" in set(history["review_id"])  # 2020-12-31 23:59:59


def test_validation_split_excludes_window_and_test_period(reviews):
    history, window = _validation_split(reviews)
    assert set(history["review_id"]) == {f"h{i}" for i in range(1, 9)}
    assert set(window["review_id"]) == {f"w{i}" for i in range(1, 9)}
    assert history["date"].max() < pd.Timestamp(T_VAL)
    assert check_no_leakage(history, window, T_VAL)["review_id_overlap"] == 0


def test_test_stage_uses_everything_before_t_test_as_history(reviews):
    history, window = split_reviews(reviews, "test", T_VAL, T_TEST)
    assert set(window["review_id"]) == {"t1"}
    assert "w8" in set(history["review_id"])


def test_history_start_limits_history(reviews):
    history, _ = split_reviews(
        reviews, "validation", T_VAL, T_TEST, history_start=datetime(2020, 4, 1)
    )
    assert {"h1", "h2", "h3"}.isdisjoint(set(history["review_id"]))
    assert "h4" in set(history["review_id"])  # 2020-04-01 10:00 >= history_start


def test_leakage_check_rejects_window_rows_in_history(reviews):
    history, window = _validation_split(reviews)
    leaked = pd.concat([history, window.iloc[[0]]])
    with pytest.raises(LeakageError):
        check_no_leakage(leaked, window, T_VAL)


def _eval_set(reviews, target_definition="visit", max_users=None):
    history, window = _validation_split(reviews)
    catalog_ids = set(history["business_id"])  # b1~b4, b5 는 cold
    return build_eval_set(history, window, catalog_ids, 4, target_definition, max_users, 42)


def test_eval_set_rules_visit(reviews):
    es = _eval_set(reviews)
    assert es.users == ["u1", "u6"]
    assert es.targets == {"u1": ["b3"], "u6": ["b2", "b4"]}
    assert es.seen["u1"] == {"b1", "b2"}
    assert es.window_users == 6
    assert es.exclusions == {
        "no_history": 1, "no_positive_history": 1, "no_relevant_target": 0,
        "revisit_only": 1, "targets_all_cold": 1, "not_sampled": 0,
    }
    assert es.dropped_targets == {"revisit_targets": 2, "cold_targets": 1}


def test_eval_set_rules_positive_visit(reviews):
    es = _eval_set(reviews, "positive_visit")
    assert es.users == ["u6"]
    assert es.targets == {"u6": ["b2"]}
    assert es.exclusions["revisit_only"] == 2  # u1 (b3 가 3점이라 정답 아님), u4


def test_max_users_sampling_is_seeded_and_counted(reviews):
    first = _eval_set(reviews, max_users=1)
    second = _eval_set(reviews, max_users=1)
    assert first.users == second.users
    assert len(first.users) == 1
    assert first.exclusions["not_sampled"] == 1
