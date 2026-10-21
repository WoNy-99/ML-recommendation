"""System One 형식 API(Codiv 호스팅 OpenJev 등)의 최소 클라이언트와 누적 예산 기록.

주소·경로·모델·요청 ID 헤더·중단 오류 이름은 코드에 고정하지 않고 설정에서 주입한다.
2026-10-22 공식 문서 기준 형식:
  POST <endpoint_url>, Authorization: Bearer <키>
  요청 {model, state, questions{id: {type: "choice", instructions, criteria{선택지: 설명}}}}
  응답 {model, answers{id: {type, choice, probabilities, confidence}}, usage{input_tokens, ...}}
- 보내는 리뷰 데이터는 처리 후 텍스트(state) 하나뿐이다. review_id·별점 등 메타데이터는 넣지 않는다.
- 키 값은 헤더에만 쓰고 결과·로그·예산 기록에 저장하지 않는다.
- confidence는 모델 확률 분포의 퍼짐 정도일 뿐 보정된 정확도가 아니다.
- 시간 상한: 호출 직전에 남은 시간을 다시 확인하고, 요청 timeout을 남은 시간 이하로 줄인다.
  다만 requests의 timeout은 연결·읽기 단계별 제한이어서 전체 실행 시간을 엄밀하게 보장하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from src.aspects import ASPECTS, LABELS

RETRYABLE_STATUS = {429, 529}  # 요청 과다·과부하 (HTTP 의미)
HALT_STATUS = {401, 403}       # 인증 문제: 이후 요청도 실패하므로 중단
CHARS_PER_TOKEN_ESTIMATE = 2   # 임시 추정치. 실제 토큰 수의 상한을 보장하지 않는다.


class InvalidRetryAfter(ValueError):
    """retry-after 값을 해석할 수 없음. 서버 지시보다 일찍 재시도하지 않도록 중단한다."""


class SchemaError(ValueError):
    """응답이 질문·선택지·확률 형식과 맞지 않음."""


class BudgetExceeded(RuntimeError):
    """요청 전 예약 단계에서 누적 상한을 넘음 (네트워크 호출 없음)."""


class ConnectionFailure(RuntimeError):
    """연결 실패·timeout. 응답을 받지 못했어도 예산에는 시도 1회로 남는다."""


@dataclass
class TransportResponse:
    status: int
    headers: dict
    body: Any
    text: str = ""


Transport = Callable[[str, dict, dict, float], TransportResponse]


def requests_transport(url: str, headers: dict, payload: dict, timeout: float) -> TransportResponse:
    """실제 HTTP 호출. 시험에서는 가짜 transport로 대체한다."""
    import requests

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:  # 메시지에 헤더(키)가 들어가지 않도록 형식만 남긴다
        raise ConnectionFailure(type(exc).__name__) from None
    try:
        body = resp.json()
    except ValueError:
        body = None
    return TransportResponse(resp.status_code, {k.lower(): v for k, v in resp.headers.items()},
                             body, resp.text[:2000] if body is None else "")


def load_questions(path: str | Path) -> tuple[dict, str, str]:
    """질문 YAML을 읽고 (questions, version, sha256)을 돌려준다. 선택지는 LABELS와 같아야 한다."""
    raw = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    questions = doc.get("questions") if isinstance(doc, dict) else None
    if not isinstance(questions, dict) or tuple(questions) != ASPECTS:
        raise ValueError(f"questions must define exactly {ASPECTS} in this order")
    for qid, q in questions.items():
        if q.get("type") != "choice" or not q.get("instructions"):
            raise ValueError(f"question {qid} must be a choice with instructions")
        if tuple(q.get("criteria") or {}) != LABELS:
            raise ValueError(f"question {qid} criteria must be {LABELS} in this order")
    return questions, str(doc.get("version")), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_payload(model: str, text: str, questions: dict) -> dict:
    return {"model": model, "state": text, "questions": questions}


def estimate_input_tokens(payload: dict) -> int:
    """요청 본문 전체(텍스트+질문+선택지) 문자 수 / 2. 사전 예산 검사용 임시 추정치."""
    chars = len(json.dumps(payload, ensure_ascii=False))
    return math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE)


def validate_response(body: Any, questions: dict) -> dict:
    """정상 응답이면 {질문id: {choice, probabilities, confidence}}. 아니면 SchemaError."""
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise SchemaError("response has no answers object")
    answers = {}
    for qid, q in questions.items():
        ans = body["answers"].get(qid)
        if not isinstance(ans, dict):
            raise SchemaError(f"answer missing for {qid}")
        options = tuple(q["criteria"])
        choice = ans.get("choice")
        if choice not in options:
            raise SchemaError(f"{qid}: unknown choice {choice!r}")
        probs = ans.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(options):
            raise SchemaError(f"{qid}: probabilities must cover exactly {options}")
        values = list(probs.values())
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   and 0.0 <= v <= 1.0 for v in values):
            raise SchemaError(f"{qid}: probabilities must be finite numbers in [0, 1]")
        if abs(sum(values) - 1.0) > 0.02:
            raise SchemaError(f"{qid}: probabilities sum to {sum(values):.4f}")
        conf = ans.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (
            math.isfinite(conf) and 0.0 <= conf <= 1.0
        ):
            raise SchemaError(f"{qid}: confidence must be a number in [0, 1]")
        answers[qid] = {"choice": choice, "probabilities": dict(probs), "confidence": float(conf)}
    return answers


class BudgetLedger:
    """요청 직전 시도 수·추정 입력을 예약하고, 응답 후 실제 usage를 따로 남기는 JSONL 기록.

    같은 scope의 기록은 실행·재시작을 넘어 누적된다. 파일을 지우면 누적이 사라지므로 지우지 않는다.
    """

    def __init__(self, path: str | Path, scope: str, max_requests: int, max_estimated_tokens: int):
        self.path = Path(path)
        self.scope = scope
        self.max_requests = max_requests
        self.max_estimated_tokens = max_estimated_tokens

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as exc:  # 손상 시 조용히 넘기지 않고 멈춘다
                        raise RuntimeError(f"budget ledger line {n} is corrupt") from exc
        return entries

    def totals(self) -> dict:
        mine = [e for e in self._entries() if e["scope"] == self.scope]
        reserved = [e for e in mine if e["event"] == "reserve"]
        results = [e for e in mine if e["event"] == "result"]
        known = [e["usage_input_tokens"] for e in results if e.get("usage_input_tokens") is not None]
        return {
            "scope": self.scope,
            "requests_reserved": len(reserved),
            "estimated_input_tokens_reserved": sum(e["estimated_input_tokens"] for e in reserved),
            "results_recorded": len(results),
            "usage_input_tokens_known_sum": sum(known),
            "results_with_unknown_usage": len(results) - len(known),
            "max_requests": self.max_requests,
            "max_estimated_input_tokens": self.max_estimated_tokens,
        }

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()

    def reserve(self, run_id: str, estimated_tokens: int) -> int:
        t = self.totals()
        if t["requests_reserved"] + 1 > self.max_requests:
            raise BudgetExceeded(f"request limit {self.max_requests} reached ({self.scope})")
        if t["estimated_input_tokens_reserved"] + estimated_tokens > self.max_estimated_tokens:
            raise BudgetExceeded(
                f"estimated input token limit {self.max_estimated_tokens} reached ({self.scope})"
            )
        reservation = t["requests_reserved"] + 1
        self._append({
            "event": "reserve", "scope": self.scope, "run_id": run_id, "reservation": reservation,
            "estimated_input_tokens": estimated_tokens,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return reservation

    def record_result(self, run_id: str, reservation: int, outcome: str,
                      http_status: int | None, usage_input_tokens: int | None) -> None:
        self._append({
            "event": "result", "scope": self.scope, "run_id": run_id, "reservation": reservation,
            "outcome": outcome, "http_status": http_status,
            "usage_input_tokens": usage_input_tokens,  # 모르면 null (0으로 확정하지 않음)
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


@dataclass
class CallOutcome:
    status: str                      # ok | api_error | schema_error | not_called
    answers: dict | None = None
    model: str | None = None
    usage: dict | None = None
    request_id: str | None = None
    error: str | None = None
    raw_response: Any = None
    estimated_input_tokens: int = 0
    attempts: list = field(default_factory=list)


def _error_type(body: Any) -> str | None:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return detail.get("error_type")
        err = body.get("error")
        if isinstance(err, dict):
            return err.get("type")
    return None


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """retry-after 헤더 → 기다릴 초. 헤더가 없으면 None.

    - 0 이상 숫자(초): 그대로 쓴다. 길어도 줄이지 않는다.
    - HTTP 날짜: 그 시각까지 남은 초. 이미 지난 날짜면 0.
    - 음수·NaN·무한대·해석 불가: InvalidRetryAfter (호출하는 쪽에서 재시도하지 않고 중단).
    """
    if value is None:
        return None
    text = str(value).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError) as exc:
            raise InvalidRetryAfter(f"unparseable retry-after {text!r}") from exc
        if when is None:
            raise InvalidRetryAfter(f"unparseable retry-after {text!r}")
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - now).total_seconds())
    if not math.isfinite(seconds) or seconds < 0:
        raise InvalidRetryAfter(f"invalid retry-after {text!r}")
    return seconds


class SystemOneClient:
    """한 번에 한 요청. 요청 시작 간격 min_interval, 최초 1회 + 재시도 최대 max_retries회.

    예산(ledger)·실행 시간 상한(deadline)이 재시도보다 우선한다. 중단 사유가 생기면 halted_reason을
    세우고 이후 classify는 네트워크 호출 없이 not_called를 돌려준다.
    """

    def __init__(self, *, endpoint_url: str, api_key: str, model: str, questions: dict,
                 ledger: BudgetLedger, run_id: str, request_id_header: str | None = None,
                 halt_error_types: Iterable[str] = (),
                 transport: Transport = requests_transport, timeout: float = 30.0,
                 min_interval: float = 1.0, max_retries: int = 3, deadline: float | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._url = endpoint_url
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.model = model
        self.questions = questions
        self.ledger = ledger
        self.run_id = run_id
        self.request_id_header = request_id_header.lower() if request_id_header else None
        self.halt_error_types = frozenset(halt_error_types)
        self.transport = transport
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.deadline = deadline
        self.sleep = sleep
        self.clock = clock
        self.wall_clock = wall_clock
        self.halted_reason: str | None = None
        self._last_start: float | None = None

    def _time_left(self) -> float:
        return math.inf if self.deadline is None else self.deadline - self.clock()

    def _halt(self, outcome: CallOutcome, reason: str, error: str) -> CallOutcome:
        """중단 사유를 남긴다. 이번 항목에서 아직 호출하지 않았으면 not_called."""
        self.halted_reason = reason
        outcome.status = "api_error" if outcome.attempts else "not_called"
        outcome.error = error
        return outcome

    def classify(self, text: str) -> CallOutcome:
        payload = build_payload(self.model, text, self.questions)
        outcome = CallOutcome(
            status="not_called", estimated_input_tokens=estimate_input_tokens(payload)
        )
        if self.halted_reason:
            outcome.error = f"halted: {self.halted_reason}"
            return outcome
        for attempt in range(1 + self.max_retries):
            # 1) 요청 간격: 기다릴 시간이 남은 실행 시간 이상이면 기다리지 않고 중단
            if self._last_start is not None:
                wait = self.min_interval - (self.clock() - self._last_start)
                if wait > 0:
                    if wait >= self._time_left():
                        return self._halt(outcome, "run_time_limit",
                                          "no time left for the request interval")
                    self.sleep(wait)
            # 2) 호출 직전 재확인 -> 3) 예산 예약 -> 4) 남은 시간 이하의 timeout으로 호출
            time_left = self._time_left()
            if time_left <= 0:
                return self._halt(outcome, "run_time_limit", "run time limit reached")
            try:
                reservation = self.ledger.reserve(self.run_id, outcome.estimated_input_tokens)
            except BudgetExceeded as exc:
                return self._halt(outcome, "budget_exceeded", str(exc))
            call_timeout = min(self.timeout, time_left)
            started = self.clock()
            self._last_start = started
            record = {"attempt": attempt + 1, "reservation": reservation,
                      "timeout_seconds": call_timeout,
                      "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                resp = self.transport(self._url, self._headers, payload, call_timeout)
            except ConnectionFailure as exc:
                record.update(latency_ms=(self.clock() - started) * 1000, http_status=None,
                              error=f"connection: {exc}")
                outcome.attempts.append(record)
                outcome.status, outcome.error = "api_error", f"connection: {exc}"
                self.ledger.record_result(self.run_id, reservation, "connection_error", None, None)
                if not self._backoff(attempt, None):
                    break
                continue

            usage = resp.body.get("usage") if isinstance(resp.body, dict) else None
            usage_in = usage.get("input_tokens") if isinstance(usage, dict) else None
            etype = _error_type(resp.body)
            request_id = (resp.headers.get(self.request_id_header)
                          if self.request_id_header else None)
            record.update(latency_ms=(self.clock() - started) * 1000, http_status=resp.status,
                          error_type=etype, request_id=request_id)
            outcome.attempts.append(record)
            outcome.request_id = record["request_id"]
            outcome.raw_response = resp.body if resp.body is not None else resp.text
            self.ledger.record_result(self.run_id, reservation,
                                      "http_ok" if resp.status == 200 else "http_error",
                                      resp.status, usage_in if isinstance(usage_in, int) else None)
            if resp.status == 200:
                outcome.model = resp.body.get("model") if isinstance(resp.body, dict) else None
                outcome.usage = usage if isinstance(usage, dict) else None
                try:
                    outcome.answers = validate_response(resp.body, self.questions)
                    outcome.status = "ok"
                except SchemaError as exc:
                    outcome.status, outcome.error = "schema_error", str(exc)
                return outcome
            outcome.status = "api_error"
            outcome.error = f"http {resp.status} {etype or ''}".strip()
            if etype in self.halt_error_types:
                self.halted_reason = etype
                return outcome
            if resp.status in HALT_STATUS:
                self.halted_reason = f"http_{resp.status}"
                return outcome
            if resp.status not in RETRYABLE_STATUS:
                return outcome
            try:
                retry_after = parse_retry_after(resp.headers.get("retry-after"),
                                                self.wall_clock())
            except InvalidRetryAfter as exc:
                return self._halt(outcome, "invalid_retry_after", str(exc))
            if not self._backoff(attempt, retry_after):
                return outcome
        outcome.status = "api_error"
        outcome.error = outcome.error or "retries exhausted"
        return outcome

    def _backoff(self, attempt: int, retry_after: float | None) -> bool:
        """재시도할 수 있으면 기다리고 True. 마지막 시도였거나 남은 시간 안에 못 기다리면 False.

        서버가 준 retry-after는 줄이지 않는다. 없으면 1, 2, 4초 지수 백오프.
        """
        if attempt >= self.max_retries:
            return False
        delay = retry_after if retry_after is not None else float(2 ** attempt)
        if delay >= self._time_left():
            self.halted_reason = "run_time_limit"
            return False
        self.sleep(delay)
        return True
