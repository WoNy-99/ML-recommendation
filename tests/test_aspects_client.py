import json
import math
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from aspects_fakes import FAKE_KEY, QUESTIONS, FakeClock, FakeTransport, error, ok, ok_body
from src.aspects.client import (
    BudgetLedger, ConnectionFailure, InvalidRetryAfter, SystemOneClient, TransportResponse,
    build_payload, estimate_input_tokens, parse_retry_after,
)

TEXT = "The pasta was great but the server was rude."
ENDPOINT = "https://example.invalid/v1/systemone"   # 시험용 주소 (실제 주소는 설정 파일에서 주입)
NOW = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)


def make(tmp_path, transport, max_requests=100, max_tokens=10**7, deadline=None, timeout=30.0):
    clock = FakeClock()
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", "pilot", max_requests, max_tokens)
    client = SystemOneClient(
        endpoint_url=ENDPOINT, api_key=FAKE_KEY, model="openjev-latest", questions=QUESTIONS,
        ledger=ledger, run_id="run-1", request_id_header="X-TypeSafe-Request-Id",
        halt_error_types=["quota_exceeded_error"], transport=transport, timeout=timeout,
        deadline=deadline, sleep=clock.sleep, clock=clock, wall_clock=lambda: NOW,
    )
    return client, ledger, clock


def test_ok_response_and_payload_has_only_text(tmp_path):
    fake = FakeTransport([ok("positive", "negative")])
    client, ledger, _ = make(tmp_path, fake)
    out = client.classify(TEXT)
    assert out.status == "ok"
    assert out.answers["food"]["choice"] == "positive"
    assert out.answers["service"]["choice"] == "negative"
    assert out.model == "openjev-0.1" and out.usage["input_tokens"] == 300
    assert out.request_id == "req-1"                     # 헤더 이름도 설정에서 주입
    call = fake.calls[0]
    assert call["url"] == ENDPOINT                       # 주소·경로를 코드에서 만들지 않음
    assert set(call["payload"]) == {"model", "state", "questions"}   # 메타데이터 없음
    assert call["payload"]["state"] == TEXT
    assert call["payload"]["model"] == "openjev-latest"
    assert ledger.totals()["usage_input_tokens_known_sum"] == 300


def _bad(mutate):
    body = ok_body()
    mutate(body)
    return TransportResponse(200, {}, body)


@pytest.mark.parametrize("mutate", [
    lambda b: b["answers"].pop("service"),
    lambda b: b["answers"]["food"].update(choice="great"),
    lambda b: b["answers"]["food"]["probabilities"].update(positive=0.5),   # 합이 1이 아님
    lambda b: b["answers"]["food"]["probabilities"].update(positive=math.nan),
    lambda b: b["answers"]["food"]["probabilities"].pop("mixed"),
    lambda b: b["answers"]["food"].pop("confidence"),
])
def test_schema_errors_are_not_retried_or_relabelled(tmp_path, mutate):
    fake = FakeTransport([_bad(mutate)])
    client, _, _ = make(tmp_path, fake)
    out = client.classify(TEXT)
    assert out.status == "schema_error" and out.answers is None
    assert len(fake.calls) == 1


def test_rate_limit_retry_after_is_honoured(tmp_path):
    fake = FakeTransport([error(429, "rate_limit_error", retry_after="2"), ok()])
    client, _, clock = make(tmp_path, fake)
    out = client.classify(TEXT)
    assert out.status == "ok" and len(out.attempts) == 2
    assert 2.0 in clock.sleeps


def test_long_retry_after_is_not_shortened(tmp_path):
    fake = FakeTransport([error(429, "rate_limit_error", retry_after="120"), ok()])
    client, _, clock = make(tmp_path, fake)
    assert client.classify(TEXT).status == "ok"
    assert clock.sleeps == [120.0]                       # 60초로 줄이지 않음


def test_retry_after_longer_than_time_left_stops_without_another_call(tmp_path):
    fake = FakeTransport([error(429, "rate_limit_error", retry_after="120")])
    client, ledger, clock = make(tmp_path, fake, deadline=100.0)
    out = client.classify(TEXT)
    assert out.status == "api_error" and client.halted_reason == "run_time_limit"
    assert len(fake.calls) == 1 and clock.sleeps == []
    assert ledger.totals()["requests_reserved"] == 1


def test_http_date_retry_after_waits_until_that_time(tmp_path):
    later = format_datetime(NOW + timedelta(seconds=90), usegmt=True)
    fake = FakeTransport([error(529, "overloaded_error", retry_after=later), ok()])
    client, _, clock = make(tmp_path, fake)
    assert client.classify(TEXT).status == "ok"
    assert clock.sleeps == [pytest.approx(90.0)]


@pytest.mark.parametrize("bad", ["-5", "soon", "nan", "inf"])
def test_invalid_retry_after_stops_instead_of_retrying_early(tmp_path, bad):
    fake = FakeTransport([error(429, "rate_limit_error", retry_after=bad)])
    client, _, clock = make(tmp_path, fake)
    out = client.classify(TEXT)
    assert out.status == "api_error" and client.halted_reason == "invalid_retry_after"
    assert len(fake.calls) == 1 and clock.sleeps == []


def test_parse_retry_after_values():
    assert parse_retry_after(None, NOW) is None
    assert parse_retry_after("0", NOW) == 0.0
    past = format_datetime(NOW - timedelta(seconds=30), usegmt=True)
    assert parse_retry_after(past, NOW) == 0.0           # 이미 지난 날짜는 즉시
    with pytest.raises(InvalidRetryAfter):
        parse_retry_after("-1", NOW)


def test_overloaded_gives_up_after_first_plus_three_retries(tmp_path):
    fake = FakeTransport([error(529, "overloaded_error")] * 4)
    client, ledger, _ = make(tmp_path, fake)
    out = client.classify(TEXT)
    assert out.status == "api_error" and len(fake.calls) == 4
    assert ledger.totals()["requests_reserved"] == 4


def test_connection_failure_counts_and_usage_is_unknown(tmp_path):
    fake = FakeTransport([ConnectionFailure("Timeout"), ok()])
    client, ledger, _ = make(tmp_path, fake)
    out = client.classify(TEXT)
    assert out.status == "ok" and len(out.attempts) == 2
    totals = ledger.totals()
    assert totals["requests_reserved"] == 2
    assert totals["results_with_unknown_usage"] == 1   # 실패 요청을 0토큰으로 확정하지 않음


def test_quota_exceeded_stops_immediately_and_halts(tmp_path):
    fake = FakeTransport([error(429, "quota_exceeded_error")])
    client, _, _ = make(tmp_path, fake)
    out = client.classify(TEXT)
    assert out.status == "api_error" and len(fake.calls) == 1
    assert client.halted_reason == "quota_exceeded_error"
    assert client.classify(TEXT).status == "not_called"
    assert len(fake.calls) == 1


def test_auth_error_halts_without_retry(tmp_path):
    fake = FakeTransport([error(401, "authentication_error")])
    client, _, _ = make(tmp_path, fake)
    assert client.classify(TEXT).status == "api_error"
    assert client.halted_reason == "http_401" and len(fake.calls) == 1


def test_request_limit_includes_retries(tmp_path):
    fake = FakeTransport([error(529, "overloaded_error")] * 2)
    client, ledger, _ = make(tmp_path, fake, max_requests=2)
    out = client.classify(TEXT)
    assert out.status == "api_error" and len(fake.calls) == 2
    assert client.halted_reason == "budget_exceeded"
    assert client.classify(TEXT).status == "not_called" and len(fake.calls) == 2


def test_budget_accumulates_across_separate_runs(tmp_path):
    first = FakeTransport([ok(), ok()])
    client1, _, _ = make(tmp_path, first, max_requests=3)
    assert [client1.classify(TEXT).status for _ in range(2)] == ["ok", "ok"]
    second = FakeTransport([ok()])
    client2, ledger2, _ = make(tmp_path, second, max_requests=3)   # 새 실행, 같은 기록 파일
    assert client2.classify(TEXT).status == "ok"
    assert client2.classify(TEXT).status == "not_called"
    assert len(second.calls) == 1 and ledger2.totals()["requests_reserved"] == 3


def test_estimated_token_limit_blocks_before_any_network_call(tmp_path):
    fake = FakeTransport([])
    estimate = estimate_input_tokens(build_payload("openjev-latest", TEXT, QUESTIONS))
    client, ledger, _ = make(tmp_path, fake, max_tokens=estimate - 1)
    out = client.classify(TEXT)
    assert out.status == "not_called" and fake.calls == []
    assert ledger.totals()["requests_reserved"] == 0


def test_estimate_counts_questions_not_only_text():
    payload = build_payload("openjev-latest", "", QUESTIONS)
    assert estimate_input_tokens(payload) > 100


def test_min_interval_between_request_starts(tmp_path):
    fake = FakeTransport([ok(), ok()])
    client, _, clock = make(tmp_path, fake)
    client.classify(TEXT)
    client.classify(TEXT)
    assert clock.sleeps == [1.0]


def test_deadline_reached_during_interval_wait_makes_no_call_or_reservation(tmp_path):
    fake = FakeTransport([ok()])
    client, ledger, clock = make(tmp_path, fake, deadline=0.5)
    assert client.classify(TEXT).status == "ok"          # t=0 에 첫 호출
    out = client.classify(TEXT)                           # 1초 간격을 기다리면 마감을 넘김
    assert out.status == "not_called" and client.halted_reason == "run_time_limit"
    assert len(fake.calls) == 1 and clock.sleeps == []
    assert ledger.totals()["requests_reserved"] == 1


def test_call_timeout_is_limited_by_time_left(tmp_path):
    fake = FakeTransport([ok()])
    client, _, _ = make(tmp_path, fake, deadline=10.0, timeout=30.0)
    out = client.classify(TEXT)
    assert fake.calls[0]["timeout"] == pytest.approx(10.0)
    assert out.attempts[0]["timeout_seconds"] == pytest.approx(10.0)


def test_api_key_is_never_recorded(tmp_path):
    fake = FakeTransport([ok(), error(529, "overloaded_error"), ConnectionFailure("x"), ok()])
    client, _, _ = make(tmp_path, fake)
    outs = [client.classify(TEXT), client.classify(TEXT)]
    dumped = json.dumps([o.__dict__ for o in outs], default=str)
    assert FAKE_KEY not in dumped
    assert FAKE_KEY not in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert fake.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
