"""누수 없는 시간 분할 기준선 실행 (인기순·TF-IDF·무작위).

실행: python -m src.run_baseline --config configs/baseline.yaml
산출물: <tracking.output_root>/<시각>_<stage>/ 아래 config.yaml, eval_manifest.json, results.json,
        per_user_<model>.csv, run.log. MLflow를 쓸 수 없으면 이 파일들만으로 근거를 남긴다.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.baselines import (
    PopularityRecommender, RandomRecommender, TfidfRecommender, build_catalog,
)
from src.config import BaselineSettings, read_config, text_sha256, validate_baseline
from src.metrics import evaluate_recommender
from src.prepare_data import DATE_FORMAT, file_sha256
from src.split import (
    build_eval_set, check_no_leakage, eval_manifest, json_sha256, split_reviews, window_bounds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("pandas", "numpy", "scipy", "scikit-learn", "PyYAML", "mlflow")
log = logging.getLogger("run_baseline")


def load_prepared(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((data_dir / "prepare_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        actual = file_sha256(data_dir / name)
        if actual != info["sha256"]:
            raise RuntimeError(f"{name} sha256 differs from prepare_manifest.json")
    with open(data_dir / "businesses.jsonl", encoding="utf-8") as fh:
        businesses = pd.DataFrame([json.loads(line) for line in fh if line.strip()])
    with gzip.open(data_dir / "reviews.jsonl.gz", "rt", encoding="utf-8") as fh:
        reviews = pd.DataFrame([json.loads(line) for line in fh if line.strip()])
    if reviews.empty or businesses.empty:
        raise RuntimeError(f"prepared data in {data_dir} has no reviews or no businesses")
    reviews["date"] = pd.to_datetime(reviews["date"], format=DATE_FORMAT)
    reviews["stars"] = reviews["stars"].astype(int)
    return businesses, reviews, manifest


def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    tracked_changes = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # 추적 파일의 미커밋 변경 여부. 새 파일(untracked)은 포함하지 않는다.
        "tracked_changes": None if tracked_changes is None else bool(tracked_changes),
    }


def code_fingerprint(root: Path, config_path: Path, config_sha256: str) -> dict:
    """실제로 실행한 코드 식별용. src/*.py·의존성 파일의 상대 경로와 SHA-256, 사용한 설정 해시.

    src/가 미추적이거나 미커밋이어도 실행 코드 버전을 구분할 수 있게 한다.
    원본 zip·data/·비밀 파일은 대상에 넣지 않는다.
    """
    root = Path(root).resolve()
    paths = sorted((root / "src").glob("*.py")) + [
        root / name for name in ("requirements.txt", "requirements-mlflow.txt")
        if (root / name).is_file()
    ]
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in paths}
    resolved = Path(config_path).resolve()
    config_key = (
        resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)
    )
    lines = [f"{path}\t{digest}" for path, digest in sorted(files.items())]
    lines.append(f"config:{config_key}\t{config_sha256}")
    return {
        "files": files,
        "config": {"path": config_key, "sha256": config_sha256},
        "combined_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
    }


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def build_models(settings: BaselineSettings, catalog: pd.DataFrame, history: pd.DataFrame):
    for name in settings.models:
        start = time.perf_counter()
        if name == "popularity":
            model = PopularityRecommender(catalog, history)
        elif name == "tfidf":
            model = TfidfRecommender(
                catalog, history, settings.positive_min_stars, settings.tfidf_max_features,
                settings.tfidf_ngram_range, settings.tfidf_min_df, settings.tfidf_stop_words,
            )
        else:
            model = RandomRecommender(catalog, settings.seed)
        yield name, model, time.perf_counter() - start


def log_to_mlflow(settings: BaselineSettings, results: dict, run_dir: Path) -> dict:
    """모델마다 MLflow run 하나. 설치·연결 문제는 실패로 기록하고 JSON 결과는 그대로 둔다."""
    if not settings.mlflow_enabled:
        return {"status": "disabled"}
    try:
        import mlflow
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"import failed: {exc}"}
    k = settings.top_k
    run_ids = {}
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        for name, model_result in results["models"].items():
            with mlflow.start_run(run_name=f"{results['run_id']}-{name}") as run:
                mlflow.set_tags({
                    "model": name,
                    "stage": settings.stage,
                    "run_dir": str(run_dir),
                    "git_commit": results["git"]["commit"],
                    "git_tracked_changes": str(results["git"]["tracked_changes"]),
                    "code_combined_sha256": results["code"]["combined_sha256"],
                    "config_sha256": results["config_sha256"],
                    "eval_manifest_sha256": results["eval"]["manifest_sha256"],
                    "data_reviews_sha256": results["data"]["reviews_sha256"],
                    "peak_memory": "unmeasured",
                })
                mlflow.log_params({
                    "t_val": str(settings.t_val.date()),
                    "t_test": str(settings.t_test.date()),
                    "history_start": str(settings.history_start or ""),
                    "target_definition": settings.target_definition,
                    "positive_min_stars": settings.positive_min_stars,
                    "top_k": k,
                    "seed": settings.seed,
                    "max_eval_users": settings.max_eval_users or "",
                    "tfidf_max_features": settings.tfidf_max_features,
                    "tfidf_ngram_range": str(list(settings.tfidf_ngram_range)),
                    "tfidf_min_df": settings.tfidf_min_df,
                })
                metrics = {
                    key: value for key, value in model_result.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                metrics["n_eval_users"] = results["eval"]["n_users"]
                mlflow.log_metrics(metrics)
                for path in sorted(run_dir.iterdir()):
                    if path.name in {"config.yaml", "results.json", "eval_manifest.json",
                                     "run.log", f"per_user_{name}.csv"}:
                        mlflow.log_artifact(str(path))
                run_ids[name] = run.info.run_id
    except Exception as exc:  # MLflow 실패가 기준선 결과를 지우지 않도록
        return {"status": "failed", "reason": repr(exc), "run_ids": run_ids}
    return {"status": "logged", "tracking_uri": settings.mlflow_tracking_uri, "run_ids": run_ids}


def run(config_path: str) -> Path:
    cfg, raw = read_config(config_path)
    settings = validate_baseline(cfg)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{settings.stage}"
    run_dir = settings.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # 이전 실행 결과를 덮지 않는다
    (run_dir / "config.yaml").write_text(raw, encoding="utf-8")

    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()], force=True)

    results: dict = {
        "status": "running",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "config_path": str(config_path),
        "config_sha256": text_sha256(raw),
        "git": git_state(),
        "code": code_fingerprint(REPO_ROOT, Path(config_path), text_sha256(raw)),
        "environment": environment(),
        "peak_memory": {"status": "unmeasured"},
    }
    total_start = time.perf_counter()
    try:
        log.info("run_id=%s stage=%s git=%s", run_id, settings.stage, results["git"])
        businesses, reviews, manifest = load_prepared(settings.data_dir)
        results["data"] = {
            "data_dir": str(settings.data_dir),
            "reviews_sha256": manifest["files"]["reviews.jsonl.gz"]["sha256"],
            "businesses_sha256": manifest["files"]["businesses.jsonl"]["sha256"],
            "limited_sample": manifest.get("limited", False),
            "reviews": int(len(reviews)),
            "businesses": int(len(businesses)),
        }
        log.info("loaded reviews=%d businesses=%d", len(reviews), len(businesses))

        history, window = split_reviews(
            reviews, settings.stage, settings.t_val, settings.t_test, settings.history_start
        )
        window_start, window_end = window_bounds(settings.stage, settings.t_val, settings.t_test)
        leakage = check_no_leakage(history, window, window_start)
        catalog = build_catalog(history, businesses)
        eval_set = build_eval_set(
            history, window, set(catalog["business_id"]), settings.positive_min_stars,
            settings.target_definition, settings.max_eval_users, settings.seed,
        )
        manifest_obj = eval_manifest(eval_set)
        (run_dir / "eval_manifest.json").write_text(
            json.dumps(manifest_obj, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        results["split"] = {
            "stage": settings.stage,
            "window_start": str(window_start),
            "window_end": None if window_end is None else str(window_end),
            "history_start": str(settings.history_start) if settings.history_start else None,
            "boundary_rule": "date < boundary -> earlier part; date >= boundary -> later part",
            "leakage_check": leakage,
            "catalog_size": int(len(catalog)),
        }
        results["eval"] = {
            "n_users": len(eval_set.users),
            "window_users": eval_set.window_users,
            "exclusions": eval_set.exclusions,
            "dropped_targets": eval_set.dropped_targets,
            "meta": eval_set.meta,
            "manifest_sha256": json_sha256(manifest_obj),
        }
        log.info("eval users=%d exclusions=%s", len(eval_set.users), eval_set.exclusions)
        if not eval_set.users:
            raise RuntimeError("0 evaluation users: check t_val/t_test and target_definition")

        results["models"] = {}
        for name, model, build_seconds in build_models(settings, catalog, history):
            evaluated = evaluate_recommender(model, eval_set, settings.top_k)
            rows = evaluated.pop("rows")
            if [r["user_id"] for r in rows] != eval_set.users:
                raise RuntimeError(f"{name} did not evaluate the shared user list")
            pd.DataFrame(
                [{**r, "top_k": "|".join(r["top_k"])} for r in rows]
            ).to_csv(run_dir / f"per_user_{name}.csv", index=False, encoding="utf-8")
            evaluated["build_seconds"] = build_seconds
            if isinstance(model, TfidfRecommender):
                evaluated["fit_documents"] = model.fit_documents
                evaluated["vocabulary_size"] = len(model.vectorizer.vocabulary_)
                evaluated["empty_profiles"] = model.empty_profiles
            results["models"][name] = evaluated
            log.info("%s %s", name, {k: v for k, v in evaluated.items()})

        results["total_seconds"] = time.perf_counter() - total_start
        results["status"] = "completed"
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = repr(exc)
        results["traceback"] = traceback.format_exc()
        results["total_seconds"] = time.perf_counter() - total_start
        log.error("run failed: %r", exc)
        _write_results(run_dir, results)
        _close_log(handler)
        raise

    _write_results(run_dir, results)
    results["mlflow"] = log_to_mlflow(settings, results, run_dir)
    _write_results(run_dir, results)
    log.info("mlflow=%s", results["mlflow"])
    _close_log(handler)
    return run_dir


def _close_log(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def _write_results(run_dir: Path, results: dict) -> None:
    (run_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_dir = run(args.config)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    print(f"[baseline] run_dir={run_dir}")
    print(f"[baseline] eval_users={results['eval']['n_users']} "
          f"exclusions={results['eval']['exclusions']}")
    for name, res in results["models"].items():
        keys = [k for k in res if k.startswith(("recall_at_", "ndcg_at_"))]
        print(f"[baseline] {name}: " + " ".join(f"{k}={res[k]:.4f}" for k in keys))
    print(f"[baseline] mlflow={results['mlflow']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
