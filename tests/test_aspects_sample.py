import csv
from datetime import datetime

import pandas as pd
import pytest

from src.aspects.evaluate import load_labels
from src.aspects.sample import (
    LABEL_COLUMNS, mask_text, process_text, sample_sha256, select_train, stratified_sample,
    write_sample,
)


def synthetic_train(per_star: int = 4) -> pd.DataFrame:
    rows = []
    for star in (1, 2, 3, 4, 5):
        for i in range(per_star):
            rows.append({"review_id": f"r{star}_{i}", "stars": star,
                         "date": pd.Timestamp("2020-06-01"),
                         "text": f"Review {star}-{i}: the food was ok.\nSecond line."})
    return pd.DataFrame(rows)


def test_select_train_excludes_boundary_and_later():
    # 세 값 모두 같은 'YYYY-MM-DD HH:MM:SS' 형식 (형식이 섞이면 pandas 버전에 따라 파싱이 달라질 수 있음)
    df = pd.DataFrame({"review_id": ["a", "b", "c"], "date": pd.to_datetime(
        ["2020-12-31 23:59:59", "2021-01-01 00:00:00", "2021-03-01 00:00:00"],
        format="%Y-%m-%d %H:%M:%S")})
    assert select_train(df, datetime(2021, 1, 1))["review_id"].tolist() == ["a"]


def test_stratified_sample_is_reproducible_and_disjoint():
    train = synthetic_train()
    a = stratified_sample(train, 42, per_star=3, dev_per_star=1, max_chars=2000)
    b = stratified_sample(train.sample(frac=1, random_state=7), 42, 3, 1, 2000)  # 입력 순서만 다름
    assert a.equals(b)
    assert len(a) == 15 and a["review_id"].is_unique
    assert (a.groupby("stars").size() == 3).all()
    assert (a[a["split"] == "dev"].groupby("stars").size() == 1).all()
    dev = set(a.loc[a["split"] == "dev", "review_id"])
    final = set(a.loc[a["split"] == "final", "review_id"])
    assert len(dev) == 5 and len(final) == 10 and dev.isdisjoint(final)


def test_duplicate_review_ids_and_too_few_candidates():
    train = pd.concat([synthetic_train(3), synthetic_train(3)])  # 같은 review_id 두 번
    sample = stratified_sample(train, 42, 3, 1, 2000)
    assert sample["review_id"].is_unique
    with pytest.raises(ValueError):
        stratified_sample(synthetic_train(2), 42, 3, 1, 2000)


@pytest.mark.parametrize("raw, expected", [
    ("mail john.doe@example.com now", "mail [EMAIL] now"),
    ("see https://x.com/a?b=1 and www.site.org", "see [URL] and [URL]"),
    ("call (215) 555-1234 or 215-555-1234", "call [PHONE] or [PHONE]"),
    ("call +1 215.555.1234 now", "call [PHONE] now"),
    ("waited 2 hours, paid $12.50 for 3 tacos", "waited 2 hours, paid $12.50 for 3 tacos"),
])
def test_masking(raw, expected):
    assert mask_text(raw) == expected


def test_mask_before_truncate_and_newline_normalization():
    text = "a" * 1990 + " x@ex.com " + "b" * 600
    out = process_text(text, 2000)
    assert len(out) == 2000 and "@" not in out
    assert process_text("one\r\ntwo\rthree", 100) == "one\ntwo\nthree"


def test_write_sample_label_files_and_no_overwrite(tmp_path):
    sample = stratified_sample(synthetic_train(), 42, 3, 1, 2000)
    manifest = write_sample(sample, tmp_path, {"seed": 42})
    with open(tmp_path / "labels_dev.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == LABEL_COLUMNS                     # 예측·confidence·별점 열 없음
    dev = sample[sample["split"] == "dev"]
    assert [r["text"] for r in rows] == dev["text"].tolist()   # 사람도 같은 처리 후 텍스트
    assert all(r["food_label"] == "" and r["service_label"] == "" for r in rows)
    reread = pd.read_csv(tmp_path / "sample.csv", encoding="utf-8-sig", keep_default_na=False,
                         dtype={"review_id": str, "split": str, "text": str})
    assert sample_sha256(reread) == manifest["sample_sha256"]  # 줄바꿈 포함 텍스트 왕복
    (tmp_path / "labels_dev.csv").write_text("kept", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_sample(sample, tmp_path, {"seed": 42})
    assert (tmp_path / "labels_dev.csv").read_text(encoding="utf-8") == "kept"


def test_load_labels_requires_every_label(tmp_path):
    path = tmp_path / "labels.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerow({"item": 0, "review_id": "r", "text": "t", "food_label": "positive",
                    "service_label": ""})
    with pytest.raises(ValueError):
        load_labels(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerow({"item": 0, "review_id": "=-abc", "text": "t", "food_label": "Positive ",
                    "service_label": "not_mentioned"})
    assert load_labels(path) == {0: {"review_id": "=-abc", "food": "positive",
                                     "service": "not_mentioned"}}
