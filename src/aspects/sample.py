"""학습 구간(date < t_val) 리뷰에서 파일럿 표본을 뽑고 라벨용 CSV를 만든다.

실행: python -m src.aspects.sample --config configs/aspects_pilot.yaml
- 별점(1~5)별 per_star개, 그중 dev_per_star개는 개발용, 나머지는 최종용 (처음부터 고정)
- 이메일·URL·전화번호 패턴만 가린 뒤 max_chars로 자른다. 개인정보가 모두 지워진다는 보장은 없다.
- 사람·키워드 규칙·모델은 모두 이 처리 후 텍스트를 사용한다.
- review_id는 로컬 연결용이다. 출력은 data/ 아래(gitignore)에 두고 기존 파일을 덮지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.aspects import ASPECTS
from src.config import parse_day, read_config

MASK_RULES_VERSION = "v1"
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")
SAMPLE_FILES = ("sample.csv", "labels_dev.csv", "labels_final.csv", "sample_manifest.json")
# item(=sample_order)이 라벨 연결 키다. Excel이 '-'로 시작하는 review_id를 수식으로 바꿔도 깨지지 않게 한다.
LABEL_COLUMNS = ("item", "review_id", "text") + tuple(f"{a}_label" for a in ASPECTS) + ("notes",)
CSV_ENCODING = "utf-8-sig"  # Windows Excel에서 한글·특수문자가 깨지지 않도록 BOM 포함


def mask_text(text: str) -> str:
    """URL → 이메일 → 전화번호 순으로 가린다. 이름·주소 등 다른 식별 정보는 처리하지 않는다."""
    text = _URL.sub("[URL]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    return _PHONE.sub("[PHONE]", text)


def process_text(text: str, max_chars: int) -> str:
    """줄바꿈을 \\n으로 통일하고 가린 뒤 자른다(자르면서 패턴 일부가 남지 않도록 순서를 고정)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return mask_text(text)[:max_chars]


def select_train(reviews: pd.DataFrame, t_val) -> pd.DataFrame:
    """B1과 같은 경계 규칙: date < t_val 만 학습 구간 (경계 시각과 같은 리뷰는 제외)."""
    return reviews[reviews["date"] < t_val]


def stratified_sample(
    train: pd.DataFrame, seed: int, per_star: int, dev_per_star: int, max_chars: int
) -> pd.DataFrame:
    """별점별로 per_star개를 seed로 고르고 앞의 dev_per_star개를 dev, 나머지를 final로 둔다.

    후보는 review_id 오름차순으로 정렬한 뒤 고르므로 입력 파일 순서와 무관하게 재현된다.
    처리 후 텍스트가 비어 있는 리뷰는 후보에서 뺀다.
    """
    if not 0 < dev_per_star < per_star:
        raise ValueError("need 0 < dev_per_star < per_star")
    candidates = train.drop_duplicates("review_id").copy()
    candidates["processed"] = candidates["text"].fillna("").astype(str).map(
        lambda t: process_text(t, max_chars)
    )
    candidates = candidates[candidates["processed"].str.strip() != ""]
    rng = np.random.default_rng(seed)
    rows = []
    for star in (1, 2, 3, 4, 5):
        pool = candidates[candidates["stars"] == star].sort_values("review_id")
        if len(pool) < per_star:
            raise ValueError(f"only {len(pool)} candidates for stars={star}, need {per_star}")
        picked = pool.iloc[rng.choice(len(pool), size=per_star, replace=False)]
        for rank, (_, row) in enumerate(picked.iterrows()):
            rows.append({
                "review_id": row["review_id"],
                "split": "dev" if rank < dev_per_star else "final",
                "stars": int(star),
                "text": row["processed"],
            })
    sample = pd.DataFrame(rows)
    sample.insert(0, "sample_order", range(len(sample)))
    return sample


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_sha256(sample: pd.DataFrame) -> str:
    """표본 식별 해시: 순서·review_id·split·처리 후 텍스트 해시. run_pilot이 같은 방식으로 재확인한다."""
    identity = "\n".join(
        f"{r.sample_order}\t{r.review_id}\t{r.split}\t{_sha256_text(r.text)}"
        for r in sample.sort_values("sample_order").itertuples(index=False)
    )
    return _sha256_text(identity)


def write_sample(sample: pd.DataFrame, out_dir: Path, meta: dict) -> dict:
    """sample.csv(로컬 분석용, 별점 포함)와 라벨용 CSV 2개(별점·예측 없음)를 쓴다. 덮어쓰지 않는다."""
    out_dir = Path(out_dir)
    existing = [n for n in SAMPLE_FILES if (out_dir / n).exists()]
    if existing:
        raise FileExistsError(f"{out_dir} already has {existing}; existing labels are kept")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_dir / "sample.csv", index=False, encoding=CSV_ENCODING)
    for split in ("dev", "final"):
        part = sample[sample["split"] == split]
        with open(out_dir / f"labels_{split}.csv", "w", newline="", encoding=CSV_ENCODING) as fh:
            writer = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            for row in part.itertuples(index=False):
                writer.writerow(
                    {"item": row.sample_order, "review_id": row.review_id, "text": row.text}
                )
    manifest = {
        **meta,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mask_rules_version": MASK_RULES_VERSION,
        "counts": sample.groupby(["split", "stars"]).size().rename("n").reset_index()
        .to_dict(orient="records"),
        "sample_sha256": sample_sha256(sample),
    }
    (out_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg, _ = read_config(args.config)
    base_cfg, base_raw = read_config(cfg["baseline_config"])
    t_val = parse_day(base_cfg["split"]["t_val"], "split.t_val")
    data_dir = Path(base_cfg["data"]["output_dir"])

    from src.run_baseline import load_prepared  # B1 해시 검증 로더를 그대로 사용
    _, reviews, prepare_manifest = load_prepared(data_dir)
    train = select_train(reviews, t_val)
    s = cfg["sample"]
    sample = stratified_sample(train, s["seed"], s["per_star"], s["dev_per_star"], s["max_chars"])
    manifest = write_sample(sample, Path(s["output_dir"]), {
        "seed": s["seed"],
        "per_star": s["per_star"],
        "dev_per_star": s["dev_per_star"],
        "max_chars": s["max_chars"],
        "t_val": str(t_val.date()),
        "baseline_config_sha256": _sha256_text(base_raw),
        "data_reviews_sha256": prepare_manifest["files"]["reviews.jsonl.gz"]["sha256"],
        "train_reviews": int(len(train)),
    })
    print(f"[sample] output_dir={s['output_dir']}")
    print(f"[sample] train_reviews={len(train)} sample={len(sample)} "
          f"dev={int((sample['split'] == 'dev').sum())} "
          f"final={int((sample['split'] == 'final').sum())}")
    print(f"[sample] sample_sha256={manifest['sample_sha256']}")
    print("[sample] next: fill food_label/service_label in labels_dev.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
