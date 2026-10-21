"""읽고 설명할 수 있는 단순 키워드 규칙 (비교 기준선).

규칙
1. 텍스트를 마침표·느낌표·물음표·세미콜론·쉼표와 ' but '으로 절 단위로 나눈다.
2. 절에 속성 단어(ASPECT_TERMS)가 있으면 그 절의 긍정·부정 단어를 센다.
   바로 앞 2단어 안에 부정어(NEGATORS)가 있으면 극성을 뒤집는다 ("not good" -> 부정).
3. 속성의 모든 절에서 긍정과 부정이 모두 나오면 mixed, 긍정만 positive, 부정만 negative,
   속성 단어가 없거나 평가 단어가 없으면 not_mentioned.
규칙을 바꾸면 RULES_VERSION을 올리고, 최종 표본 평가 전에 고정한다.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

RULES_VERSION = "v1"

ASPECT_TERMS = {
    "food": [
        "food", "dish", "dishes", "meal", "meals", "menu", "taste", "tasted", "tasty",
        "flavor", "flavors", "flavour", "delicious", "pizza", "burger", "burgers", "sandwich",
        "chicken", "steak", "sushi", "pasta", "salad", "soup", "fries", "dessert", "breakfast",
        "lunch", "dinner", "portion", "portions", "appetizer", "appetizers", "cooked", "bland",
    ],
    "service": [
        "service", "staff", "server", "servers", "waiter", "waiters", "waitress", "waitstaff",
        "bartender", "host", "hostess", "manager", "employee", "employees", "cashier",
        "attentive", "rude", "wait", "waited", "waiting",
    ],
}
POSITIVE = [
    "good", "great", "excellent", "amazing", "delicious", "tasty", "fresh", "friendly",
    "attentive", "best", "love", "loved", "awesome", "perfect", "nice", "fantastic",
    "wonderful", "helpful", "quick", "fast", "polite", "yummy",
]
NEGATIVE = [
    "bad", "terrible", "awful", "horrible", "rude", "slow", "cold", "bland", "overcooked",
    "undercooked", "stale", "worst", "disappointing", "disappointed", "dirty", "mediocre",
    "gross", "salty", "dry", "burnt", "poor", "wrong", "ignored", "rushed", "greasy",
]
NEGATORS = ["not", "no", "never", "isn't", "wasn't", "weren't", "didn't", "don't", "hardly"]

_CLAUSE_SPLIT = re.compile(r"[.!?;,]+|\bbut\b", flags=re.IGNORECASE)
_TOKEN = re.compile(r"[a-z']+")


def fingerprint_of(source: bytes) -> str:
    """규칙 소스 바이트(단어 목록 + 절 분리·부정어 처리 코드)의 해시."""
    return hashlib.sha256(source).hexdigest()


def rules_fingerprint() -> str:
    """이 파일 전체 내용의 해시. freeze 검사와 실행 기록에 같은 값을 쓴다.

    처리 로직을 바꿔도 해시가 바뀐다. 줄바꿈(CRLF/LF) 변환이나 주석 수정도 해시를 바꾸므로,
    freeze 값은 실제로 final을 실행할 같은 작업 사본에서 계산한다.
    """
    return fingerprint_of(Path(__file__).read_bytes())


def _clause_polarity(tokens: list[str]) -> tuple[int, int]:
    pos = neg = 0
    positive, negative, negators = set(POSITIVE), set(NEGATIVE), set(NEGATORS)
    for i, token in enumerate(tokens):
        if token not in positive and token not in negative:
            continue
        flipped = any(t in negators for t in tokens[max(0, i - 2):i])
        is_positive = (token in positive) != flipped
        if is_positive:
            pos += 1
        else:
            neg += 1
    return pos, neg


def classify(text: str) -> dict[str, str]:
    """속성별 라벨 dict. 항상 ASPECT_TERMS의 모든 속성을 돌려준다."""
    clauses = [_TOKEN.findall(c.lower()) for c in _CLAUSE_SPLIT.split(text or "")]
    result = {}
    for aspect, terms in ASPECT_TERMS.items():
        term_set = set(terms)
        has_pos = has_neg = False
        for tokens in clauses:
            if not term_set.intersection(tokens):
                continue
            pos, neg = _clause_polarity(tokens)
            has_pos |= pos > 0
            has_neg |= neg > 0
        if has_pos and has_neg:
            result[aspect] = "mixed"
        elif has_pos:
            result[aspect] = "positive"
        elif has_neg:
            result[aspect] = "negative"
        else:
            result[aspect] = "not_mentioned"
    return result
