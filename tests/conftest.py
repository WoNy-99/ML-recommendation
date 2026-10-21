"""합성 자료. 실제 Yelp 리뷰는 시험에 쓰지 않는다.

기본 경계: T_VAL = 2021-01-01, T_TEST = 2021-07-01
- u1: 이력 b1(5), b2(4) / 평가 b3(경계 시각 정각, 3점), b1 재방문 / 테스트 구간 b4
- u2: 이력 b2(2점뿐) -> 긍정 이력 없음
- u3: 평가 구간에만 등장 -> 이력 없음
- u4: 이력 b1 / 평가 b1 -> 재방문만
- u5: 이력 b2 / 평가 b5 (이력 리뷰가 없는 cold item)
- u6: 이력 b3 / 평가 b4(1점), b2(5점)
- u7: 이력 b4, b3(경계 1초 전) / 평가 없음
"""
from __future__ import annotations

import io
import json
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

T_VAL = datetime(2021, 1, 1)
T_TEST = datetime(2021, 7, 1)

BUSINESSES = [
    {"business_id": "b1", "name": "Pizza Place", "categories": "Restaurants, Pizza",
     "city": "Philadelphia", "state": "PA"},
    {"business_id": "b2", "name": "Pasta House", "categories": "Restaurants, Italian",
     "city": "Philadelphia", "state": "PA"},
    {"business_id": "b3", "name": "Ramen Bar", "categories": "Restaurants, Japanese",
     "city": "Philadelphia", "state": "PA"},
    {"business_id": "b4", "name": "Sushi Spot", "categories": "Japanese, Restaurants",
     "city": "Philadelphia", "state": "PA"},
    {"business_id": "b5", "name": "Taco Truck", "categories": "Restaurants, Mexican",
     "city": "Philadelphia", "state": "PA"},
]

# (review_id, user_id, business_id, stars, date, text)
REVIEW_ROWS = [
    ("h1", "u1", "b1", 5, "2020-01-01 10:00:00", "pizza cheese crust"),
    ("h2", "u1", "b2", 4, "2020-02-01 10:00:00", "pizza pasta sauce"),
    ("h3", "u2", "b2", 2, "2020-03-01 10:00:00", "cold pasta"),
    ("h4", "u4", "b1", 5, "2020-04-01 10:00:00", "great pizza"),
    ("h5", "u5", "b2", 5, "2020-04-15 10:00:00", "pasta lovely"),
    ("h6", "u6", "b3", 5, "2020-05-01 10:00:00", "ramen broth noodles"),
    ("h7", "u7", "b4", 4, "2020-06-01 10:00:00", "sushi fish rice"),
    ("h8", "u7", "b3", 4, "2020-12-31 23:59:59", "ramen spicy"),
    ("w1", "u1", "b3", 3, "2021-01-01 00:00:00", "zzwindowtoken ramen"),
    ("w2", "u1", "b1", 5, "2021-02-01 10:00:00", "pizza again"),
    ("w3", "u2", "b1", 4, "2021-02-02 10:00:00", "pizza"),
    ("w4", "u3", "b1", 4, "2021-02-03 10:00:00", "pizza"),
    ("w5", "u4", "b1", 5, "2021-02-04 10:00:00", "pizza"),
    ("w6", "u5", "b5", 4, "2021-03-01 10:00:00", "tacos"),
    ("w7", "u6", "b4", 1, "2021-03-02 10:00:00", "bad sushi"),
    ("w8", "u6", "b2", 5, "2021-03-03 10:00:00", "pasta"),
    ("t1", "u1", "b4", 5, "2021-08-01 10:00:00", "zztestperiod sushi"),
]
COLUMNS = ("review_id", "user_id", "business_id", "stars", "date", "text")


def review_records() -> list[dict]:
    return [dict(zip(COLUMNS, row)) for row in REVIEW_ROWS]


@pytest.fixture
def reviews() -> pd.DataFrame:
    df = pd.DataFrame(review_records())
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    return df


@pytest.fixture
def businesses() -> pd.DataFrame:
    return pd.DataFrame(BUSINESSES)


def _jsonl(rows) -> bytes:
    return ("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n").encode()


def write_yelp_zip(tmp_path: Path, business_rows, review_rows, compression: str = "") -> Path:
    """실제 자료와 같은 구조: zip 안 'Yelp JSON/yelp_dataset.tar' + __MACOSX 잔여 파일.
    tar 안에서는 리뷰 파일을 음식점 파일보다 앞에 둔다(순서 무관 동작 확인).
    compression="gz"이면 이름은 .tar 그대로 두고 내용만 gzip 압축한다."""
    tar_path = tmp_path / "yelp_dataset.tar"
    members = (
        ("yelp_dataset/yelp_academic_dataset_review.json", _jsonl(review_rows)),
        ("yelp_dataset/yelp_academic_dataset_business.json", _jsonl(business_rows)),
    )
    with tarfile.open(tar_path, f"w:{compression}" if compression else "w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    zip_path = tmp_path / "Yelp-JSON.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tar_path, "Yelp JSON/yelp_dataset.tar")
        zf.writestr("__MACOSX/Yelp JSON/._yelp_dataset.tar", b"resource fork")
    tar_path.unlink()
    return zip_path
