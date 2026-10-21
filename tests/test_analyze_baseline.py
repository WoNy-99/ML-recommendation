"""B1 결과 해석 분석의 집계 규칙. 합성 자료만 쓴다."""
import json

import numpy as np
import pandas as pd
import pytest

import src.analyze_baseline as ab
from src.analyze_baseline import (
    AnalysisError, check_overall, compare_by_history, history_bin, load_user_metrics,
    popularity_order, positive_history_counts, target_concentration, top_sets,
)
from src.config import text_sha256
from src.split import json_sha256


def test_popularity_order_ties_and_top_set_sizes():
    ids = [f"b{i:02d}" for i in range(11)]                 # 이미 business_id 오름차순
    scores = np.array([1, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0], dtype=float)
    ordered = popularity_order(ids, scores)
    assert ordered[:4] == ["b01", "b02", "b04", "b00"]       # 동률(3,3)은 id 오름차순
    sets = top_sets(ordered)
    assert sets["top_10pct"] == {"b01", "b02"}               # ceil(0.1 * 11) = 2
    assert sets["top_20pct"] == {"b01", "b02", "b04"}        # ceil(0.2 * 11) = 3


def test_pair_share_and_user_mean_share_use_different_denominators():
    targets = {"u1": ["b", "x"], "u2": ["x"], "u3": ["b", "c"]}
    c = target_concentration(targets, {"b", "c"})
    assert (c["pairs"], c["pairs_in_top"]) == (5, 3)
    assert c["pair_share"] == pytest.approx(3 / 5)
    assert c["user_mean_share"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)


def test_positive_history_counts_distinct_businesses():
    history = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u1", "u2"],
        "business_id": ["b1", "b1", "b2", "b3", "b1"],
        "stars": [5, 4, 3, 4, 2],
    })
    assert positive_history_counts(history, ["u1", "u2"], 4) == {"u1": 2, "u2": 0}


def test_history_bins_are_fixed_and_zero_stops():
    assert [history_bin(n) for n in (1, 2, 4, 5, 30)] == ["1", "2-4", "2-4", "5+", "5+"]
    with pytest.raises(AnalysisError):
        history_bin(0)


def _write_metrics(path, rows):
    pd.DataFrame(rows, columns=["user_id", "n_targets", "n_recommended", "recall", "ndcg",
                                "top_k"]).to_csv(path, index=False, encoding="utf-8")


def test_user_metrics_must_match_manifest_exactly(tmp_path):
    good = tmp_path / "good.csv"
    _write_metrics(good, [["u1", 1, 10, 1.0, 1.0, "x"], ["u2", 2, 10, 0.5, 0.3, "y"]])
    assert load_user_metrics(good, ["u1", "u2"]) == {"u1": (1.0, 1.0), "u2": (0.5, 0.3)}
    with pytest.raises(AnalysisError, match="missing 1"):
        load_user_metrics(good, ["u1", "u2", "u3"])
    dup = tmp_path / "dup.csv"
    _write_metrics(dup, [["u1", 1, 10, 1.0, 1.0, "x"], ["u1", 1, 10, 0.0, 0.0, "x"]])
    with pytest.raises(AnalysisError, match="duplicate"):
        load_user_metrics(dup, ["u1"])
    bad = tmp_path / "bad.csv"
    _write_metrics(bad, [["u1", 1, 10, 1.5, 1.0, "x"]])
    with pytest.raises(AnalysisError, match="outside"):
        load_user_metrics(bad, ["u1"])


def test_bins_report_means_differences_and_empty_bins():
    counts = {"u1": 1, "u2": 3, "u3": 2}
    per_model = {
        "popularity": {"u1": (1.0, 1.0), "u2": (0.0, 0.0), "u3": (1.0, 0.5)},
        "tfidf": {"u1": (0.0, 0.0), "u2": (1.0, 1.0), "u3": (0.0, 0.0)},
    }
    out = compare_by_history(counts, per_model)
    assert out["1"]["n_users"] == 1
    assert out["1"]["tfidf_minus_popularity_recall"] == pytest.approx(-1.0)
    assert out["2-4"]["n_users"] == 2
    assert out["2-4"]["popularity_recall_at_10"] == pytest.approx(0.5)
    assert out["2-4"]["tfidf_ndcg_at_10"] == pytest.approx(0.5)
    assert out["5+"] == {"n_users": 0, "status": "not_computable"}   # 0점이 아님


def test_overall_check_stops_on_mismatch():
    per_model = {"popularity": {"u1": (1.0, 0.5)}, "tfidf": {"u1": (0.0, 0.0)}}
    ok = {"models": {"popularity": {"recall_at_10": 1.0, "ndcg_at_10": 0.5},
                     "tfidf": {"recall_at_10": 0.0, "ndcg_at_10": 0.0}}}
    assert all(v["match"] for v in check_overall(per_model, ok).values())
    wrong = {"models": {"popularity": {"recall_at_10": 0.9, "ndcg_at_10": 0.5},
                        "tfidf": {"recall_at_10": 0.0, "ndcg_at_10": 0.0}}}
    with pytest.raises(AnalysisError):
        check_overall(per_model, wrong)


CONFIG_TEXT = "stage: validation\neval:\n  positive_min_stars: 4\n"


class _Reached(Exception):
    """설정 대조를 통과해 설정 검증 단계에 들어왔다는 표식."""


def _source_run(tmp_path, config_text, recorded_sha):
    """B1 실행 폴더 모양의 합성 자료. 설정 사본은 Windows write_text처럼 CRLF 바이트로 쓴다."""
    run = tmp_path / "run"
    run.mkdir()
    manifest = {"users": []}
    (run / "eval_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "config.yaml").write_bytes(config_text.replace("\n", "\r\n").encode("utf-8"))
    results = {"status": "completed", "eval": {"manifest_sha256": json_sha256(manifest)}}
    if recorded_sha is not None:
        results["config_sha256"] = recorded_sha
    (run / "results.json").write_text(json.dumps(results), encoding="utf-8")
    for name in ("per_user_popularity.csv", "per_user_tfidf.csv"):
        (run / name).write_bytes(b"")
    return run


def _guard_later_steps(monkeypatch, on_validate):
    monkeypatch.setattr(ab, "validate_baseline", on_validate)

    def no_load(*_):
        raise AssertionError("review data must not be loaded")
    monkeypatch.setattr(ab, "load_review_fields", no_load)


def test_config_copy_matching_b1_hash_is_accepted_despite_crlf(tmp_path, monkeypatch):
    # B1 run_baseline 은 read_config 텍스트(LF)의 text_sha256 을 기록한다
    run = _source_run(tmp_path, CONFIG_TEXT, text_sha256(CONFIG_TEXT))
    seen = []

    def reached(cfg):
        seen.append(cfg)
        raise _Reached
    _guard_later_steps(monkeypatch, reached)
    with pytest.raises(_Reached):
        ab.analyze(run)
    assert seen[0]["eval"]["positive_min_stars"] == 4


@pytest.mark.parametrize("copied_text, recorded", [
    (CONFIG_TEXT.replace("positive_min_stars: 4", "positive_min_stars: 3"),
     text_sha256(CONFIG_TEXT)),
    (CONFIG_TEXT, None),
])
def test_changed_or_unrecorded_config_stops_before_validation_and_loading(
        tmp_path, monkeypatch, copied_text, recorded):
    run = _source_run(tmp_path, copied_text, recorded)

    def no_validate(_):
        raise AssertionError("config must be checked before validation")
    _guard_later_steps(monkeypatch, no_validate)
    with pytest.raises(AnalysisError, match="config_sha256"):
        ab.analyze(run)
