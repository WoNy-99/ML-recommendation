"""B1 검증 기준선 결과 해석용 최소 분석 (모델 재학습·추천 재실행 없음).

실행: python -m src.analyze_baseline --run artifacts/runs/20261022-012737_validation

A. 정답의 인기도 분포
   - B1과 같은 학습 구간·후보·인기도(학습 구간 고유 리뷰 사용자 수)·동률 규칙(business_id 오름차순)
   - 후보를 인기 순으로 세운 뒤 상위 ceil(10%·N), ceil(20%·N)개를 "상위 집합"으로 고정한다
   - 평가 사용자(기존 manifest 5,000명)의 정답 중 상위 집합에 속한 비중을
     (1) 전체 사용자-정답 쌍 기준, (2) 사용자별 비중의 평균(사용자별 평균 Recall과 같은 단위) 두 가지로 낸다
B. 긍정 이력 길이별 성능
   - 긍정 이력 길이 = 학습 구간의 별점 >= positive_min_stars 리뷰가 있는 서로 다른 음식점 수
   - 구간 1 / 2-4 / 5+ (사전 고정). 구간별 사용자 수, 두 모델의 Recall@10·nDCG@10 평균, TF-IDF - 인기순
   - 저장된 사용자별 지표(per_user_*.csv)만 쓰고, 전체 평균을 원래 results.json 과 대조한다

입력은 기존 실행 폴더(설정 사본·manifest·사용자별 지표)와 준비된 리뷰 파일의
user_id·business_id·stars·date 필드뿐이다. 리뷰 text·개별 ID·추천 목록은 산출물에 남기지 않는다.
누락·중복·값 오류·해시 불일치가 있으면 조용히 빼지 않고 멈춘다.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import PopularityRecommender, build_catalog
from src.config import read_config, text_sha256, validate_baseline
from src.prepare_data import DATE_FORMAT, file_sha256
from src.split import json_sha256, split_reviews

REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_FRACTIONS = (0.10, 0.20)            # 결과를 보기 전에 고정
HISTORY_BINS = (("1", 1, 1), ("2-4", 2, 4), ("5+", 5, None))
MODELS = ("popularity", "tfidf")
TOLERANCE = 1e-9


class AnalysisError(RuntimeError):
    """입력 불일치·누락·값 오류. 결과를 만들지 않고 멈춘다."""


# ---------- A. 인기도 집중 ----------

def popularity_order(catalog_ids: list[str], scores: np.ndarray) -> list[str]:
    """B1 rank_with_exclusion 과 같은 순서: 점수 내림차순, 동률은 catalog 순서(business_id 오름차순)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return [catalog_ids[i] for i in order]


def top_sets(ordered_ids: list[str], fractions=TOP_FRACTIONS) -> dict[str, set[str]]:
    n = len(ordered_ids)
    return {f"top_{int(round(f * 100))}pct": set(ordered_ids[:math.ceil(f * n)])
            for f in fractions}


def target_concentration(targets: dict[str, list[str]], top: set[str]) -> dict:
    """(1) 전체 사용자-정답 쌍 중 상위 집합 비중, (2) 사용자별 비중의 평균."""
    if not targets:
        return {"status": "not_computable", "reason": "no users"}
    pairs = in_top = 0
    per_user = []
    for user_targets in targets.values():
        if not user_targets:
            raise AnalysisError("a user has no targets (B1 manifest should never contain this)")
        hit = sum(1 for b in user_targets if b in top)
        pairs += len(user_targets)
        in_top += hit
        per_user.append(hit / len(user_targets))
    return {"n_users": len(targets), "pairs": pairs, "pairs_in_top": in_top,
            "pair_share": in_top / pairs, "user_mean_share": float(np.mean(per_user))}


# ---------- B. 긍정 이력 길이 ----------

def positive_history_counts(history: pd.DataFrame, users: list[str],
                            min_stars: int) -> dict[str, int]:
    """학습 구간 별점 >= min_stars 인 서로 다른 음식점 수 (B1 프로필과 같은 중복 제거)."""
    positive = history[history["stars"] >= min_stars]
    counts = positive.groupby("user_id")["business_id"].nunique().to_dict()
    return {u: int(counts.get(u, 0)) for u in users}


def history_bin(n: int) -> str:
    for label, low, high in HISTORY_BINS:
        if n >= low and (high is None or n <= high):
            return label
    raise AnalysisError(f"positive history length {n} is outside the fixed bins "
                        "(B1 evaluation users must have at least 1)")


def load_user_metrics(path: Path, users: list[str]) -> dict[str, tuple[float, float]]:
    """per_user_<model>.csv 에서 user_id·recall·ndcg 만 읽는다. 누락·중복·이상값이면 멈춘다."""
    metrics: dict[str, tuple[float, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            user = row["user_id"]
            if user in metrics:
                raise AnalysisError(f"{path.name}: duplicate user row")
            try:
                recall, ndcg = float(row["recall"]), float(row["ndcg"])
            except (TypeError, ValueError) as exc:
                raise AnalysisError(f"{path.name}: non-numeric metric") from exc
            if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (recall, ndcg)):
                raise AnalysisError(f"{path.name}: metric outside [0, 1]")
            metrics[user] = (recall, ndcg)
    expected = set(users)
    if set(metrics) != expected:
        raise AnalysisError(f"{path.name}: users differ from eval manifest "
                            f"(missing {len(expected - set(metrics))}, "
                            f"extra {len(set(metrics) - expected)})")
    return metrics


def compare_by_history(counts: dict[str, int], per_model: dict[str, dict]) -> dict:
    out = {}
    for label, _, _ in HISTORY_BINS:
        users = [u for u, n in counts.items() if history_bin(n) == label]
        if not users:
            out[label] = {"n_users": 0, "status": "not_computable"}
            continue
        entry = {"n_users": len(users)}
        for model in MODELS:
            entry[f"{model}_recall_at_10"] = float(np.mean([per_model[model][u][0] for u in users]))
            entry[f"{model}_ndcg_at_10"] = float(np.mean([per_model[model][u][1] for u in users]))
        entry["tfidf_minus_popularity_recall"] = (entry["tfidf_recall_at_10"]
                                                  - entry["popularity_recall_at_10"])
        entry["tfidf_minus_popularity_ndcg"] = (entry["tfidf_ndcg_at_10"]
                                                - entry["popularity_ndcg_at_10"])
        out[label] = entry
    return out


def check_overall(per_model: dict[str, dict], results: dict) -> dict:
    """사용자별 지표의 평균이 원래 results.json 의 전체 지표와 같은지 확인한다."""
    checks = {}
    for model in MODELS:
        values = list(per_model[model].values())
        recall = float(np.mean([v[0] for v in values]))
        ndcg = float(np.mean([v[1] for v in values]))
        original = results["models"][model]
        ok = (abs(recall - original["recall_at_10"]) <= TOLERANCE
              and abs(ndcg - original["ndcg_at_10"]) <= TOLERANCE)
        checks[model] = {"recomputed_recall_at_10": recall, "recomputed_ndcg_at_10": ndcg,
                         "original_recall_at_10": original["recall_at_10"],
                         "original_ndcg_at_10": original["ndcg_at_10"], "match": ok}
        if not ok:
            raise AnalysisError(f"{model}: per-user mean differs from results.json")
    return checks


# ---------- 입력 적재 ----------

def load_review_fields(path: Path) -> pd.DataFrame:
    """준비된 reviews.jsonl.gz 에서 user_id·business_id·stars·date 만 남긴다(text는 버림)."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows.append((r["user_id"], r["business_id"], int(r["stars"]), r["date"]))
    df = pd.DataFrame(rows, columns=["user_id", "business_id", "stars", "date"])
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
    df["review_id"] = np.arange(len(df))  # split_reviews 누수 검사용 자리값 (원본 ID 아님)
    return df


def analyze(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    paths = {name: run_dir / name for name in (
        "results.json", "config.yaml", "eval_manifest.json",
        "per_user_popularity.csv", "per_user_tfidf.csv")}
    for name, p in paths.items():
        if not p.is_file():
            raise AnalysisError(f"missing {name} in {run_dir}")
    results = json.loads(paths["results.json"].read_text(encoding="utf-8"))
    if results.get("status") != "completed":
        raise AnalysisError("source run is not completed")
    manifest = json.loads(paths["eval_manifest.json"].read_text(encoding="utf-8"))
    if json_sha256(manifest) != results["eval"]["manifest_sha256"]:
        raise AnalysisError("eval_manifest.json does not match results.json")

    # 설정 사본은 B1과 같은 방식(read_config 텍스트의 text_sha256)으로 대조한다. 텍스트 모드로
    # 읽으므로 Windows에서 CRLF로 저장된 사본도 원래 해시와 같다. 파일 바이트 해시(input_sha256)는 기록용이다.
    cfg, raw = read_config(paths["config.yaml"])
    expected_config = results.get("config_sha256")
    if not expected_config:
        raise AnalysisError("results.json has no config_sha256 to check config.yaml against")
    if text_sha256(raw) != expected_config:
        raise AnalysisError("config.yaml does not match config_sha256 in results.json")
    settings = validate_baseline(cfg)
    reviews_path = settings.data_dir / "reviews.jsonl.gz"
    if file_sha256(reviews_path) != results["data"]["reviews_sha256"]:
        raise AnalysisError("prepared reviews differ from the source run")
    businesses_path = settings.data_dir / "businesses.jsonl"
    if file_sha256(businesses_path) != results["data"]["businesses_sha256"]:
        raise AnalysisError("prepared businesses differ from the source run")

    reviews = load_review_fields(reviews_path)
    with open(businesses_path, encoding="utf-8") as fh:
        business_ids = [json.loads(line)["business_id"] for line in fh if line.strip()]
    history, _ = split_reviews(reviews, settings.stage, settings.t_val, settings.t_test,
                               settings.history_start)
    if len(history) != results["split"]["leakage_check"]["history_reviews"]:
        raise AnalysisError("history size differs from the source run")
    catalog = build_catalog(history, pd.DataFrame({"business_id": business_ids}))
    if len(catalog) != results["split"]["catalog_size"]:
        raise AnalysisError("catalog size differs from the source run")

    users = [u["user_id"] for u in manifest["users"]]
    if len(set(users)) != len(users) or len(users) != results["eval"]["n_users"]:
        raise AnalysisError("eval manifest users are duplicated or miscounted")
    targets = {u["user_id"]: u["targets"] for u in manifest["users"]}

    popularity = PopularityRecommender(catalog, history)
    ordered = popularity_order(popularity.business_ids, popularity.scores)
    concentration = {name: target_concentration(targets, top)
                     for name, top in top_sets(ordered).items()}
    for name, frac in zip(concentration, TOP_FRACTIONS):
        concentration[name].update(top_size=math.ceil(frac * len(ordered)),
                                   catalog_size=len(ordered), catalog_fraction=frac)

    counts = positive_history_counts(history, users, settings.positive_min_stars)
    per_model = {m: load_user_metrics(paths[f"per_user_{m}.csv"], users) for m in MODELS}
    by_history = compare_by_history(counts, per_model)
    overall = check_overall(per_model, results)

    return {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run_id": results["run_id"],
        "input_sha256": {name: file_sha256(p) for name, p in paths.items()},
        "data_sha256": {"reviews": results["data"]["reviews_sha256"],
                        "businesses": results["data"]["businesses_sha256"]},
        "code_sha256": {p.relative_to(REPO_ROOT).as_posix(): file_sha256(p) for p in (
            REPO_ROOT / "src" / "analyze_baseline.py", REPO_ROOT / "src" / "baselines.py",
            REPO_ROOT / "src" / "split.py", REPO_ROOT / "src" / "config.py")},
        "definitions": {
            "popularity": "distinct users who reviewed the business in the history period "
                          "(same as B1); ties by business_id ascending",
            "top_set": "first ceil(fraction * catalog_size) businesses in popularity order",
            "pair_share": "targets in top set / all user-target pairs",
            "user_mean_share": "mean over users of (targets in top set / user's targets); "
                               "same unit as per-user mean Recall",
            "positive_history_length": f"distinct businesses with stars >= "
                                       f"{settings.positive_min_stars} in the history period",
            "history_bins": [label for label, _, _ in HISTORY_BINS],
        },
        "checks": {"eval_users": len(users), "positive_history_zero_users":
                   sum(1 for n in counts.values() if n == 0), "overall_metrics": overall},
        "A_target_popularity_concentration": concentration,
        "B_by_positive_history_length": by_history,
        "note": "observational summaries; they do not establish causes of the ranking gap",
    }


def _new_dir(root: Path) -> Path:
    base = datetime.now().strftime("%Y%m%d-%H%M%S") + "_baseline-analysis"
    root.mkdir(parents=True, exist_ok=True)
    for n in range(1000):
        path = root / (base if n == 0 else f"{base}-{n}")
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("could not create an output directory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="B1 run folder (artifacts/runs/<run_id>)")
    parser.add_argument("--output-root", default="artifacts/analysis")
    args = parser.parse_args(argv)
    try:
        report = analyze(Path(args.run))
    except AnalysisError as exc:
        print(f"[analysis] stopped: {exc}")
        return 1
    out_dir = _new_dir(Path(args.output_root))
    (out_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"[analysis] output={out_dir / 'analysis.json'} source={report['source_run_id']}")
    for name, c in report["A_target_popularity_concentration"].items():
        print(f"[analysis] A {name}: top_size={c['top_size']}/{c['catalog_size']} "
              f"pair_share={c['pair_share']:.4f} user_mean_share={c['user_mean_share']:.4f}")
    for label, b in report["B_by_positive_history_length"].items():
        if b.get("status") == "not_computable":
            print(f"[analysis] B {label}: n_users=0 not_computable")
            continue
        print(f"[analysis] B {label}: n_users={b['n_users']} "
              f"recall pop={b['popularity_recall_at_10']:.4f} tfidf={b['tfidf_recall_at_10']:.4f} "
              f"diff={b['tfidf_minus_popularity_recall']:+.4f} | "
              f"ndcg pop={b['popularity_ndcg_at_10']:.4f} tfidf={b['tfidf_ndcg_at_10']:.4f} "
              f"diff={b['tfidf_minus_popularity_ndcg']:+.4f}")
    match = all(v["match"] for v in report["checks"]["overall_metrics"].values())
    print(f"[analysis] overall metrics match results.json: {match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
