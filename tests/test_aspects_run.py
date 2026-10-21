"""run_pilot 전체 흐름. 가짜 transport만 쓰며 실제 API·리뷰 전송은 없다."""
import csv
import json

import pytest
import yaml

from aspects_fakes import FAKE_KEY, QUESTIONS_PATH, FakeClock, FakeTransport, error, ok
from src.aspects import keyword_rules
from src.aspects.client import load_questions
from src.aspects.run_pilot import SYNTHETIC_TEXTS, PilotRefused, format_summary, main, run
from src.aspects.sample import LABEL_COLUMNS, stratified_sample, write_sample
from test_aspects_sample import synthetic_train

ENV = {"TYPESAFE_API_KEY": FAKE_KEY, "TYPESAFE_BASE_URL": "https://example.invalid"}


def setup_pilot(tmp_path, *, transfer=False, connection=False, label_dev=True, freeze=False):
    sample_dir = tmp_path / "sample"
    sample = stratified_sample(synthetic_train(), 42, per_star=2, dev_per_star=1, max_chars=2000)
    write_sample(sample, sample_dir, {"seed": 42})
    if label_dev:
        path = sample_dir / "labels_dev.csv"
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "food_label": "positive", "service_label": "not_mentioned"})
    cfg = {
        "baseline_config": "configs/baseline.yaml",
        "questions_file": str(QUESTIONS_PATH),
        "sample": {"seed": 42, "per_star": 2, "dev_per_star": 1, "max_chars": 2000,
                   "output_dir": str(sample_dir)},
        "approval": {"connection_check": connection, "real_review_transfer": transfer},
        "freeze": {"questions_sha256": load_questions(QUESTIONS_PATH)[2] if freeze else None,
                   "keyword_rules_sha256": keyword_rules.rules_fingerprint() if freeze else None},
        "api": {"provider": "test", "expected_base_url": "https://example.invalid",
                "base_url_env": "TYPESAFE_BASE_URL", "api_key_env": "TYPESAFE_API_KEY",
                "endpoint_path": "/v9/test-endpoint", "request_id_header": "x-request-id",
                "halt_error_types": ["quota_exceeded_error"],
                "model": "openjev-latest", "timeout_seconds": 30, "min_interval_seconds": 1.0,
                "max_retries": 3},
        "budget": {"ledger_path": str(tmp_path / "ledger.jsonl"),
                   "connection_check": {"max_requests": 3, "max_estimated_input_tokens": 5000,
                                        "max_run_seconds": 120},
                   "pilot": {"max_requests": 100, "max_estimated_input_tokens": 200000,
                             "max_run_seconds": 900}},
        "output_root": str(tmp_path / "runs"),
    }
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(config_path), sample


def _run(config_path, mode, transport=None, keyword_only=False, env=ENV):
    clock = FakeClock()
    return run(config_path, mode, keyword_only=keyword_only, transport=transport, env=env,
               sleep=clock.sleep, clock=clock)


def test_real_reviews_are_not_sent_without_approval(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=False)
    fake = FakeTransport([])
    with pytest.raises(PilotRefused, match="real_review_transfer"):
        _run(config, "dev", fake)
    assert fake.calls == [] and not (tmp_path / "runs").exists()


def test_unlabelled_split_is_refused(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True, label_dev=False)
    with pytest.raises(PilotRefused, match="label problems"):
        _run(config, "dev", FakeTransport([]))


def test_other_base_url_is_refused(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    env = {**ENV, "TYPESAFE_BASE_URL": "http://127.0.0.1:8080"}
    with pytest.raises(PilotRefused, match="no silent switch"):
        _run(config, "dev", FakeTransport([]), env=env)


def test_final_requires_frozen_questions_and_rules(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    with pytest.raises(PilotRefused, match="freeze"):
        _run(config, "final", FakeTransport([]))


def test_keyword_only_dev_runs_without_key_and_marks_model_not_computable(tmp_path):
    config, _ = setup_pilot(tmp_path)
    run_dir = _run(config, "dev", keyword_only=True, env={})
    results = json.loads((run_dir / "results.json").read_text("utf-8"))
    assert results["status"] == "completed"
    assert results["calls"]["status_counts"]["not_called"] == 5
    food = results["metrics"]["aspects"]["food"]
    assert food["keyword_all_rows"]["n"] == 5
    assert food["openjev_compared_rows"]["status"] == "not_computable"


def test_dev_run_with_fake_api_records_evidence_and_never_the_key(tmp_path):
    config, sample = setup_pilot(tmp_path, transfer=True)
    fake = FakeTransport([], default=ok("positive", "not_mentioned"))
    run_dir = _run(config, "dev", fake)
    dev_texts = sample[sample["split"] == "dev"].sort_values("sample_order")["text"].tolist()
    assert [c["payload"]["state"] for c in fake.calls] == dev_texts
    assert fake.calls[0]["url"] == "https://example.invalid/v9/test-endpoint"  # 설정에서 주입
    results = json.loads((run_dir / "results.json").read_text("utf-8"))
    assert results["status"] == "completed"
    assert results["calls"]["models_returned"] == ["openjev-0.1"]
    assert results["budget_after"]["requests_reserved"] == 5
    assert "src/aspects/client.py" in results["code"]["files"]
    oj = results["metrics"]["aspects"]["food"]["openjev_compared_rows"]
    # 사람·모델 모두 positive만: positive F1=1, 나머지 3개 라벨 0 -> 고정 4라벨 평균 0.25
    assert oj["n"] == 5 and oj["macro_f1"] == pytest.approx(0.25)
    lines = (run_dir / "responses.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 5
    for path in list(run_dir.iterdir()) + [tmp_path / "ledger.jsonl"]:
        assert FAKE_KEY not in path.read_text(encoding="utf-8-sig")


def test_api_failures_stay_failures(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    fake = FakeTransport([], default=error(400, "invalid_request_error"))
    run_dir = _run(config, "dev", fake)
    results = json.loads((run_dir / "results.json").read_text("utf-8"))
    assert results["status"] == "completed_with_failures"
    m = results["metrics"]
    assert m["openjev_status_counts"]["api_error"] == 5 and m["n_compared"] == 0
    assert m["aspects"]["food"]["openjev_compared_rows"]["status"] == "not_computable"
    with open(run_dir / "predictions.csv", newline="", encoding="utf-8-sig") as fh:
        preds = list(csv.DictReader(fh))
    assert all(p["openjev"] == "" and p["openjev_status"] == "api_error" for p in preds)


def test_connection_check_sends_only_synthetic_text(tmp_path):
    config, sample = setup_pilot(tmp_path, connection=True)
    fake = FakeTransport([], default=ok())
    run_dir = _run(config, "connection-check", fake)
    sent = [c["payload"]["state"] for c in fake.calls]
    assert sent == list(SYNTHETIC_TEXTS)
    assert set(sent).isdisjoint(set(sample["text"]))
    results = json.loads((run_dir / "results.json").read_text("utf-8"))
    assert results["budget_after"]["scope"] == "connection_check"
    assert results["budget_after"]["requests_reserved"] == 3


def test_final_with_freeze_needs_final_labels(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True, freeze=True)
    with pytest.raises(PilotRefused, match="label problems"):
        _run(config, "final", FakeTransport([]))


def test_final_is_refused_when_rule_logic_changed_after_freeze(tmp_path, monkeypatch):
    config, _ = setup_pilot(tmp_path, transfer=True, freeze=True)   # 현재 규칙 해시로 고정
    # 고정 이후 규칙 소스(처리 로직)가 바뀐 상황을 해시만 바꿔 흉내 낸다
    monkeypatch.setattr(keyword_rules, "rules_fingerprint", lambda: "0" * 64)
    fake = FakeTransport([])
    with pytest.raises(PilotRefused, match="keyword_rules_sha256"):
        _run(config, "final", fake)
    assert fake.calls == []


def _main(config, mode, transport=None, keyword_only=False, env=ENV):
    clock = FakeClock()
    argv = ["--config", config, "--mode", mode] + (["--keyword-only"] if keyword_only else [])
    return main(argv, transport=transport, env=env, sleep=clock.sleep, clock=clock)


def _latest_results(tmp_path):
    run_dir = sorted((tmp_path / "runs").iterdir())[-1]
    return run_dir, json.loads((run_dir / "results.json").read_text("utf-8"))


def test_exit_code_zero_only_for_completed(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    assert _main(config, "dev", FakeTransport([], default=ok())) == 0


def test_exit_code_for_partial_failures_keeps_results(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    fake = FakeTransport([ok(), error(400, "invalid_request_error"), ok(), ok(), ok()])
    assert _main(config, "dev", fake) == 3
    run_dir, results = _latest_results(tmp_path)
    assert results["status"] == "completed_with_failures"
    assert (run_dir / "responses.jsonl").exists() and (run_dir / "failures.csv").exists()


def test_exit_code_for_halt_keeps_partial_results(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    fake = FakeTransport([ok(), error(429, "quota_exceeded_error")])
    assert _main(config, "dev", fake) == 4
    run_dir, results = _latest_results(tmp_path)
    assert results["status"] == "halted"
    assert results["calls"]["status_counts"] == {"ok": 1, "api_error": 1, "schema_error": 0,
                                                 "not_called": 3}
    assert len((run_dir / "responses.jsonl").read_text("utf-8").splitlines()) == 5


def test_exit_code_for_refusal_and_keyword_only(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=False)
    fake = FakeTransport([])
    assert _main(config, "dev", fake) == 2 and fake.calls == []
    assert _main(config, "dev", keyword_only=True, env={}) == 0   # 의도된 not_called는 실패 아님


def test_console_compares_keyword_and_openjev_on_the_same_rows(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    fake = FakeTransport([ok(), error(400, "invalid_request_error"), ok(), ok(), ok()])
    run_dir = _run(config, "dev", fake)
    results = json.loads((run_dir / "results.json").read_text("utf-8"))
    food = results["metrics"]["aspects"]["food"]
    assert food["keyword_compared_rows"]["n"] == food["openjev_compared_rows"]["n"] == 4
    assert food["keyword_all_rows"]["n"] == 5
    lines = format_summary(results)
    same = next(line for line in lines if line.startswith("[pilot] food same rows"))
    assert "n=4" in same
    assert f"keyword macro_f1={food['keyword_compared_rows']['macro_f1']:.4f}" in same
    assert any("compared rows (OpenJev ok only): 4/5" in line for line in lines)
    assert any(line.startswith("[pilot] food keyword on all labelled rows n=5") for line in lines)


def test_console_shows_not_computable_instead_of_zero(tmp_path):
    config, _ = setup_pilot(tmp_path, transfer=True)
    run_dir = _run(config, "dev", FakeTransport([], default=error(400, "invalid_request_error")))
    lines = format_summary(json.loads((run_dir / "results.json").read_text("utf-8")))
    same = next(line for line in lines if line.startswith("[pilot] food same rows"))
    assert "openjev macro_f1=not_computable" in same
    assert "keyword macro_f1=not_computable" in same
