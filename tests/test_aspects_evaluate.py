from pathlib import Path

import pytest

from src.aspects import keyword_rules
from src.aspects.evaluate import classification_metrics, confusion_matrix, per_label_f1, summarize


def test_confusion_matrix_and_macro_f1_hand_example():
    y_true = ["positive", "positive", "negative", "mixed"]
    y_pred = ["positive", "negative", "negative", "not_mentioned"]
    # 행=사람, 열=예측, 순서 positive, negative, mixed, not_mentioned
    assert confusion_matrix(y_true, y_pred) == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    f1 = per_label_f1(y_true, y_pred)
    assert f1["positive"] == pytest.approx(2 / 3)    # P=1, R=1/2
    assert f1["negative"] == pytest.approx(2 / 3)    # P=1/2, R=1
    assert f1["mixed"] == 0.0 and f1["not_mentioned"] == 0.0
    m = classification_metrics(y_true, y_pred)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 4)
    assert m["accuracy"] == pytest.approx(0.5)


def test_macro_f1_always_averages_the_four_fixed_labels():
    m = classification_metrics(["positive", "negative"], ["positive", "negative"])
    assert m["per_label_f1"] == {"positive": 1.0, "negative": 1.0, "mixed": 0.0,
                                 "not_mentioned": 0.0}
    assert m["macro_f1"] == pytest.approx(0.5)
    assert m["support"] == {"positive": 1, "negative": 1, "mixed": 0, "not_mentioned": 0}


def test_empty_sample_is_not_computable():
    assert classification_metrics([], []) == {"n": 0, "status": "not_computable"}


def _row(status, human="positive", keyword="positive", openjev="positive"):
    return {"review_id": "r", "human": {"food": human, "service": "not_mentioned"},
            "keyword": {"food": keyword, "service": "not_mentioned"}, "openjev_status": status,
            "openjev": ({"food": openjev, "service": "not_mentioned"} if status == "ok" else None)}


def test_comparison_uses_only_ok_rows_and_reports_counts():
    rows = [_row("ok"), _row("api_error", keyword="negative"), _row("not_called")]
    s = summarize(rows)
    assert s["n_total"] == 3 and s["n_compared"] == 1
    assert s["compared_ratio"] == pytest.approx(1 / 3)
    assert s["openjev_status_counts"] == {"ok": 1, "api_error": 1, "schema_error": 0,
                                          "abstained": 0, "not_called": 1}
    food = s["aspects"]["food"]
    assert food["keyword_all_rows"]["n"] == 3
    assert food["keyword_compared_rows"]["n"] == 1
    assert food["openjev_compared_rows"]["n"] == 1


def test_zero_ok_responses_is_not_computable():
    s = summarize([_row("api_error"), _row("schema_error")])
    assert s["aspects"]["food"]["openjev_compared_rows"] == {"n": 0, "status": "not_computable"}
    assert s["aspects"]["food"]["keyword_all_rows"]["n"] == 2


@pytest.mark.parametrize("text, food, service", [
    ("The food was delicious.", "positive", "not_mentioned"),
    ("Rude staff but great pizza", "positive", "negative"),
    ("The service was slow, but the staff was friendly", "not_mentioned", "mixed"),
    ("The food was not good", "negative", "not_mentioned"),
    ("We went on Tuesday.", "not_mentioned", "not_mentioned"),
])
def test_keyword_rules_examples(text, food, service):
    assert keyword_rules.classify(text) == {"food": food, "service": service}


def test_keyword_rules_fingerprint_covers_processing_logic():
    source = Path(keyword_rules.__file__).read_bytes()
    assert keyword_rules.rules_fingerprint() == keyword_rules.fingerprint_of(source)
    assert len(keyword_rules.rules_fingerprint()) == 64
    # 단어 목록이 아니라 절 분리 정규식만 바꿔도 해시가 달라져야 한다
    marker = b"[.!?;,]+"
    assert marker in source
    changed_logic = source.replace(marker, b"[.!?;]+")
    assert keyword_rules.fingerprint_of(changed_logic) != keyword_rules.rules_fingerprint()
