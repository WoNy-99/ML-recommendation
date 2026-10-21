"""B2 시험용 가짜 네트워크·시계. 실제 HTTP 호출은 하지 않는다."""
from pathlib import Path

from src.aspects import LABELS
from src.aspects.client import TransportResponse, load_questions

REPO = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = REPO / "configs" / "questions" / "aspects_v1.yaml"
QUESTIONS = load_questions(QUESTIONS_PATH)[0]
FAKE_KEY = "sk-test-SECRET-should-never-be-written"


def answer(choice: str, confidence: float = 0.9) -> dict:
    probs = {label: (0.97 if label == choice else 0.01) for label in LABELS}
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": confidence}


def ok_body(food: str = "positive", service: str = "not_mentioned") -> dict:
    return {"model": "openjev-0.1",
            "answers": {"food": answer(food), "service": answer(service)},
            "usage": {"input_tokens": 300, "output_tokens": 0}}


def ok(food: str = "positive", service: str = "not_mentioned") -> TransportResponse:
    return TransportResponse(200, {"x-typesafe-request-id": "req-1"}, ok_body(food, service))


def error(status: int, error_type: str, retry_after: str | None = None) -> TransportResponse:
    headers = {"retry-after": retry_after} if retry_after else {}
    return TransportResponse(status, headers, {"detail": {"error_type": error_type,
                                                          "message": "x"}})


class FakeTransport:
    def __init__(self, responses, default=None):
        self.responses = list(responses)
        self.default = default
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload,
                           "timeout": timeout})
        item = self.responses.pop(0) if self.responses else self.default
        if item is None:
            raise AssertionError("unexpected extra request")
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    """sleep이 시간을 앞으로 보낸다. 실제로 기다리지 않는다."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
