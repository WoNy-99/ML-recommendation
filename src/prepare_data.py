"""Yelp-JSON.zip을 풀지 않고 스트리밍으로 읽어 PA 음식점 리뷰 표본과 데이터 확인 보고서를 만든다.

실행: python -m src.prepare_data --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterable, Iterator

from src.config import read_config, validate_prepare

BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"
REVIEW_FIELDS = ("review_id", "user_id", "business_id", "stars", "date", "text")
BUSINESS_FIELDS = ("business_id", "name", "categories", "city", "state")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
OUTPUT_FILES = ("businesses.jsonl", "reviews.jsonl.gz", "data_report.json", "prepare_manifest.json")


class DataError(RuntimeError):
    """원본 자료 구조가 예상과 달라 진행하면 안 되는 경우."""


def detect_tar_compression(head: bytes) -> str | None:
    """파일 앞 바이트(매직 넘버)로 압축 형식 판별. 이름이 .tar여도 실제로는 압축돼 있을 수 있다."""
    if head.startswith(b"\x1f\x8b"):
        return "gz"
    if head.startswith(b"BZh"):
        return "bz2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    return None


@contextmanager
def open_dataset_file(
    zip_path: Path, filename: str
) -> Iterator[tuple[IO[bytes], str, int, str | None]]:
    """zip 안의 .tar에서 filename(이름 끝 일치) 하나를 순차 스트림으로 연다. 디스크에 풀지 않는다.

    돌려주는 값: (스트림, tar 안 경로, 압축 해제 후 크기, tar 압축 형식 또는 None)
    """
    with zipfile.ZipFile(zip_path) as zf:
        tars = [
            n for n in zf.namelist()
            if n.endswith(".tar") and not n.startswith("__MACOSX/")
        ]
        if len(tars) != 1:
            raise DataError(f"expected exactly one .tar inside zip, found {tars}")
        with zf.open(tars[0]) as tar_stream:
            head = tar_stream.peek(6)[:6]
            compression = detect_tar_compression(head)
            mode = "r|" if compression is None else f"r|{compression}"
            try:
                tar = tarfile.open(fileobj=tar_stream, mode=mode)
            except tarfile.ReadError as exc:
                raise DataError(
                    f"cannot open {tars[0]} as tar (mode={mode}, first bytes={head.hex()}): {exc}"
                ) from exc
            with tar:
                for member in tar:
                    if member.isfile() and Path(member.name).name == filename:
                        handle = tar.extractfile(member)
                        if handle is None:
                            raise DataError(f"cannot read {member.name}")
                        yield handle, member.name, member.size, compression
                        return
    raise DataError(f"{filename} not found inside {zip_path}")


def is_target_business(record: dict, state: str, category: str) -> bool:
    if record.get("state") != state:
        return False
    categories = record.get("categories") or ""
    return category in (c.strip() for c in categories.split(","))


def load_target_businesses(
    lines: Iterable[bytes], state: str, category: str
) -> tuple[dict[str, dict], Counter]:
    stats: Counter = Counter()
    kept: dict[str, dict] = {}
    for line in lines:
        stats["lines"] += 1
        if not line.strip():
            stats["blank_line"] += 1
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["parse_error"] += 1
            continue
        if not isinstance(record, dict) or not record.get("business_id"):
            stats["missing_field"] += 1
            continue
        if not is_target_business(record, state, category):
            stats["not_target"] += 1
            continue
        if record["business_id"] in kept:
            stats["duplicate_business_id"] += 1
            continue
        # stars·review_count는 데이터셋 종료 시점 값이라 미래 정보가 섞이므로 보관하지 않는다.
        kept[record["business_id"]] = {k: record.get(k) for k in BUSINESS_FIELDS}
        stats["kept"] += 1
    return kept, stats


def normalize_date(value) -> str | None:
    """'YYYY-MM-DD HH:MM:SS'로 정규화한다. 해석할 수 없거나 시간대가 붙어 있으면 None."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return None
    return parsed.strftime(DATE_FORMAT)


def filter_reviews(
    lines: Iterable[bytes],
    business_ids: set[str],
    stats: Counter,
    limit_lines: int | None = None,
) -> Iterator[dict]:
    """대상 음식점 리뷰만 검사해 내보낸다. 버린 이유는 stats에 사유별로 센다."""
    seen_ids: set[str] = set()
    for line in lines:
        if limit_lines is not None and stats["lines"] >= limit_lines:
            stats["stopped_by_limit"] = 1
            break
        stats["lines"] += 1
        if not line.strip():
            stats["blank_line"] += 1
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["parse_error"] += 1
            continue
        if not isinstance(record, dict):
            stats["parse_error"] += 1
            continue
        if record.get("business_id") not in business_ids:
            stats["other_business"] += 1
            continue
        if any(record.get(f) is None for f in REVIEW_FIELDS) or not all(
            isinstance(record[f], str) and record[f] for f in ("review_id", "user_id")
        ):
            stats["missing_field"] += 1
            continue
        stars = record["stars"]
        if isinstance(stars, bool) or not isinstance(stars, (int, float)) or stars not in (
            1, 2, 3, 4, 5
        ):
            stats["invalid_stars"] += 1
            continue
        normalized = normalize_date(record["date"])
        if normalized is None:
            stats["invalid_date"] += 1
            continue
        if record["review_id"] in seen_ids:
            stats["duplicate_review_id"] += 1
            continue
        if not isinstance(record["text"], str):
            stats["missing_field"] += 1
            continue
        if not record["text"].strip():
            stats["empty_text_kept"] += 1
        seen_ids.add(record["review_id"])
        stats["kept"] += 1
        yield {
            "review_id": record["review_id"],
            "user_id": record["user_id"],
            "business_id": record["business_id"],
            "stars": int(stars),
            "date": normalized,
            "text": record["text"],
        }


def month_quantiles(month_counts: Counter) -> dict[str, str]:
    """누적 리뷰 비율이 처음 q 이상이 되는 달. 분할 경계 추천의 참고 자료."""
    total = sum(month_counts.values())
    out: dict[str, str] = {}
    if total == 0:
        return out
    running = 0
    pending = list(QUANTILES)
    for month in sorted(month_counts):
        running += month_counts[month]
        while pending and running >= pending[0] * total:
            out[f"p{int(pending.pop(0) * 100)}"] = month
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(
    zip_path: Path,
    output_dir: Path,
    state: str,
    category: str,
    limit_review_lines: int | None = None,
    overwrite: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    existing = [n for n in OUTPUT_FILES if (output_dir / n).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already has {existing}; use another --output-dir or --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1차 순회: 대상 음식점 목록 (tar 안 파일 순서와 무관하게 동작하도록 두 번 연다)
    with open_dataset_file(zip_path, BUSINESS_FILE) as (handle, business_member, _, compression):
        businesses, business_stats = load_target_businesses(handle, state, category)
    if not businesses:
        raise DataError(f"no businesses matched state={state!r} category={category!r}")

    # 2차 순회: 리뷰를 걸러 gzip JSONL로 쓴다. 성공하면 .partial을 정식 이름으로 바꾼다.
    review_stats: Counter = Counter()
    month_counts: Counter = Counter()
    year_counts: Counter = Counter()
    star_counts: Counter = Counter()
    user_counts: Counter = Counter()
    reviewed_businesses: set[str] = set()
    date_min: str | None = None
    date_max: str | None = None
    reviews_path = output_dir / "reviews.jsonl.gz"
    partial = reviews_path.with_name(reviews_path.name + ".partial")
    with open_dataset_file(zip_path, REVIEW_FILE) as (handle, review_member, review_size, _):
        with gzip.open(partial, "wt", encoding="utf-8", newline="\n") as out:
            for row in filter_reviews(handle, set(businesses), review_stats, limit_review_lines):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                month_counts[row["date"][:7]] += 1
                year_counts[row["date"][:4]] += 1
                star_counts[str(row["stars"])] += 1
                user_counts[row["user_id"]] += 1
                reviewed_businesses.add(row["business_id"])
                date_min = row["date"] if date_min is None or row["date"] < date_min else date_min
                date_max = row["date"] if date_max is None or row["date"] > date_max else date_max
    os.replace(partial, reviews_path)

    businesses_path = output_dir / "businesses.jsonl"
    with open(businesses_path, "w", encoding="utf-8", newline="\n") as out:
        for business_id in sorted(businesses):
            out.write(json.dumps(businesses[business_id], ensure_ascii=False) + "\n")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "zip_path": str(zip_path),
            "zip_size_bytes": zip_path.stat().st_size,
            "tar_compression": compression,
            "business_member": business_member,
            "review_member": review_member,
            "review_member_size_bytes": review_size,
        },
        "filter": {"state": state, "category": category, "limit_review_lines": limit_review_lines},
        "business_stats": dict(business_stats),
        "review_stats": dict(review_stats),
        "scale": {
            "target_businesses": len(businesses),
            "businesses_with_reviews": len(reviewed_businesses),
            "reviews_kept": review_stats["kept"],
            "users": len(user_counts),
            "users_with_2plus_reviews": sum(1 for c in user_counts.values() if c >= 2),
            "date_min": date_min,
            "date_max": date_max,
        },
        "reviews_per_year": dict(sorted(year_counts.items())),
        "reviews_per_month": dict(sorted(month_counts.items())),
        "review_month_quantiles": month_quantiles(month_counts),
        "stars": dict(sorted(star_counts.items())),
    }
    (output_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": report["created_at"],
        "files": {
            name: {
                "sha256": file_sha256(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in ("businesses.jsonl", "reviews.jsonl.gz", "data_report.json")
        },
        "limited": limit_review_lines is not None,
    }
    (output_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="override data.output_dir (e.g. a smoke-test folder)")
    parser.add_argument(
        "--limit-review-lines", type=int, help="stop after N review lines (smoke test)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing prepared files")
    args = parser.parse_args(argv)

    cfg, _ = read_config(args.config)
    settings = validate_prepare(cfg)
    output_dir = Path(args.output_dir) if args.output_dir else settings["output_dir"]
    report = prepare_dataset(
        settings["zip_path"], output_dir, settings["state"], settings["category"],
        args.limit_review_lines, args.overwrite,
    )
    scale = report["scale"]
    print(f"[prepare] output_dir={output_dir}")
    print(f"[prepare] tar_compression={report['source']['tar_compression']}")
    print(f"[prepare] business_stats={report['business_stats']}")
    print(f"[prepare] review_stats={report['review_stats']}")
    print(
        f"[prepare] businesses={scale['target_businesses']} "
        f"with_reviews={scale['businesses_with_reviews']} reviews={scale['reviews_kept']} "
        f"users={scale['users']} users_2plus={scale['users_with_2plus_reviews']}"
    )
    print(f"[prepare] date_min={scale['date_min']} date_max={scale['date_max']}")
    print(f"[prepare] reviews_per_year={report['reviews_per_year']}")
    print(f"[prepare] month_quantiles={report['review_month_quantiles']}")
    print("[prepare] done: see data_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
