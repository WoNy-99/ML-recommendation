import gzip
import json

import pytest

from conftest import write_yelp_zip
from src.prepare_data import detect_tar_compression, prepare_dataset

BUSINESS_ROWS = [
    {"business_id": "bA", "name": "A", "categories": "Restaurants, Pizza", "state": "PA",
     "city": "X", "stars": 4.5, "review_count": 10},
    {"business_id": "bB", "name": "B", "categories": "Shopping", "state": "PA", "city": "X"},
    {"business_id": "bC", "name": "C", "categories": "Restaurants", "state": "NV", "city": "Y"},
    {"business_id": "bD", "name": "D", "categories": None, "state": "PA", "city": "X"},
    {"business_id": "bE", "name": "E", "categories": "Food, Restaurants", "state": "PA",
     "city": "X"},
]


def _review(rid, bid, stars=4, date="2019-05-01 12:00:00", text="ok", user="u1"):
    return {"review_id": rid, "user_id": user, "business_id": bid, "stars": stars,
            "date": date, "text": text}


REVIEW_ROWS = [
    _review("r1", "bA", 5, text="first"),
    _review("r1", "bA", 4, "2019-06-01 12:00:00", "duplicate id", user="u2"),
    _review("r3", "bB"),
    _review("r4", "bA", date="not-a-date"),
    _review("r5", "bA", text=None),
    _review("r6", "bA", stars=6),
    "{bad json",
    "",
    _review("r7", "bE", 3, "2020-01-02 03:04:05", "later", user="u3"),
    _review("r8", "bC"),
]


@pytest.fixture
def zip_path(tmp_path):
    return write_yelp_zip(tmp_path, BUSINESS_ROWS, REVIEW_ROWS)


def _read_reviews(out_dir):
    with gzip.open(out_dir / "reviews.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_filters_pa_restaurants_and_counts_drop_reasons(zip_path, tmp_path):
    out = tmp_path / "out"
    report = prepare_dataset(zip_path, out, "PA", "Restaurants")
    assert report["business_stats"]["kept"] == 2          # bA, bE ("Food, Restaurants")
    assert report["business_stats"]["not_target"] == 3
    stats = report["review_stats"]
    assert stats["kept"] == 2
    assert stats["duplicate_review_id"] == 1
    assert stats["other_business"] == 2                   # bB(쇼핑), bC(NV)
    assert stats["invalid_date"] == 1
    assert stats["missing_field"] == 1
    assert stats["invalid_stars"] == 1
    assert stats["parse_error"] == 1
    assert stats["blank_line"] == 1
    assert report["scale"]["date_min"] == "2019-05-01 12:00:00"
    assert report["scale"]["date_max"] == "2020-01-02 03:04:05"
    assert report["reviews_per_year"] == {"2019": 1, "2020": 1}

    rows = _read_reviews(out)
    assert [r["review_id"] for r in rows] == ["r1", "r7"]
    assert rows[0]["text"] == "first"                      # 중복 id 는 처음 것을 유지
    assert set(rows[0]) == {"review_id", "user_id", "business_id", "stars", "date", "text"}
    lines = (out / "businesses.jsonl").read_text("utf-8").splitlines()
    businesses = [json.loads(line) for line in lines]
    assert [b["business_id"] for b in businesses] == ["bA", "bE"]
    assert "stars" not in businesses[0] and "review_count" not in businesses[0]
    manifest = json.loads((out / "prepare_manifest.json").read_text("utf-8"))
    assert set(manifest["files"]) == {"businesses.jsonl", "reviews.jsonl.gz", "data_report.json"}
    assert manifest["limited"] is False


def test_plain_tar_is_reported_as_uncompressed(zip_path, tmp_path):
    report = prepare_dataset(zip_path, tmp_path / "out", "PA", "Restaurants")
    assert report["source"]["tar_compression"] is None


def test_gzip_compressed_tar_named_dot_tar_is_read(tmp_path):
    # 실제 Yelp-JSON.zip 에서 mode="r|" 가 'invalid header' 로 실패한 회귀 시험
    gz_zip = write_yelp_zip(tmp_path, BUSINESS_ROWS, REVIEW_ROWS, compression="gz")
    report = prepare_dataset(gz_zip, tmp_path / "out", "PA", "Restaurants")
    assert report["source"]["tar_compression"] == "gz"
    assert report["review_stats"]["kept"] == 2
    assert [r["review_id"] for r in _read_reviews(tmp_path / "out")] == ["r1", "r7"]


def test_detect_tar_compression_magic_numbers():
    assert detect_tar_compression(b"\x1f\x8b\x08\x00\x00\x00") == "gz"
    assert detect_tar_compression(b"BZh91A") == "bz2"
    assert detect_tar_compression(b"\xfd7zXZ\x00") == "xz"
    assert detect_tar_compression(b"yelp_d") is None


def test_refuses_to_overwrite_existing_output(zip_path, tmp_path):
    out = tmp_path / "out"
    prepare_dataset(zip_path, out, "PA", "Restaurants")
    with pytest.raises(FileExistsError):
        prepare_dataset(zip_path, out, "PA", "Restaurants")
    prepare_dataset(zip_path, out, "PA", "Restaurants", overwrite=True)


def test_limit_review_lines_marks_sample_as_limited(zip_path, tmp_path):
    out = tmp_path / "smoke"
    report = prepare_dataset(zip_path, out, "PA", "Restaurants", limit_review_lines=1)
    assert report["review_stats"]["kept"] == 1
    assert report["review_stats"]["stopped_by_limit"] == 1
    assert json.loads((out / "prepare_manifest.json").read_text("utf-8"))["limited"] is True
