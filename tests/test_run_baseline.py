"""합성 zip -> prepare -> run_baseline 전체 흐름 (MLflow 끔)과 실행 코드 식별 정보."""
import hashlib
import json

import yaml

from conftest import BUSINESSES, review_records, write_yelp_zip
from src.prepare_data import prepare_dataset
from src.run_baseline import REPO_ROOT, code_fingerprint, run


def test_code_fingerprint_tracks_src_and_requirements_only(tmp_path):
    (tmp_path / "src").mkdir()
    # 해시 기대값을 바이트로 고정하기 위해 write_bytes (write_text 는 Windows 에서 CRLF 가 될 수 있음)
    (tmp_path / "src" / "a.py").write_bytes(b"x = 1\n")
    (tmp_path / "requirements.txt").write_text("pandas\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "reviews.jsonl.gz").write_bytes(b"raw")
    (tmp_path / "Yelp-JSON.zip").write_bytes(b"zip")
    config = tmp_path / "configs" / "c.yaml"
    config.parent.mkdir()
    config.write_text("a: 1\n", encoding="utf-8")

    first = code_fingerprint(tmp_path, config, "cfg-hash")
    assert set(first["files"]) == {"src/a.py", "requirements.txt"}  # data·zip 제외
    assert first["files"]["src/a.py"] == hashlib.sha256(b"x = 1\n").hexdigest()
    assert first["config"] == {"path": "configs/c.yaml", "sha256": "cfg-hash"}
    assert code_fingerprint(tmp_path, config, "cfg-hash") == first  # 같은 내용이면 같은 해시

    # 설정만 바뀐 경우 (코드는 동일)
    config_only = code_fingerprint(tmp_path, config, "other")
    assert config_only["files"] == first["files"]
    assert config_only["combined_sha256"] != first["combined_sha256"]

    # 코드만 바뀐 경우 (설정 해시는 동일)
    (tmp_path / "src" / "a.py").write_bytes(b"x = 2\n")
    code_only = code_fingerprint(tmp_path, config, "cfg-hash")
    assert code_only["files"]["src/a.py"] == hashlib.sha256(b"x = 2\n").hexdigest()
    assert code_only["combined_sha256"] != first["combined_sha256"]


def test_end_to_end_run_writes_linked_evidence(tmp_path):
    zip_path = write_yelp_zip(tmp_path, BUSINESSES, review_records())
    data_dir = tmp_path / "prepared"
    prepare_dataset(zip_path, data_dir, "PA", "Restaurants")
    cfg = {
        "experiment": {"stage": "validation", "seed": 42},
        "data": {"zip_path": str(zip_path), "output_dir": str(data_dir), "state": "PA",
                 "category": "Restaurants"},
        "split": {"t_val": "2021-01-01", "t_test": "2021-07-01", "history_start": None},
        "eval": {"target_definition": "visit", "positive_min_stars": 4, "top_k": 10,
                 "max_eval_users": None},
        "models": ["popularity", "tfidf", "random"],
        "tfidf": {"max_features": 100, "ngram_range": [1, 1], "min_df": 1, "stop_words": None},
        "tracking": {"output_root": str(tmp_path / "runs"), "mlflow": False,
                     "mlflow_tracking_uri": "sqlite:///unused.db", "mlflow_experiment": "t"},
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    run_dir = run(str(config_path))

    names = {p.name for p in run_dir.iterdir()}
    assert {"config.yaml", "eval_manifest.json", "results.json", "run.log",
            "per_user_popularity.csv", "per_user_tfidf.csv", "per_user_random.csv"} <= names
    results = json.loads((run_dir / "results.json").read_text("utf-8"))
    assert results["status"] == "completed"
    assert results["mlflow"] == {"status": "disabled"}
    assert results["eval"]["n_users"] == 2
    assert results["split"]["leakage_check"]["review_id_overlap"] == 0
    assert results["split"]["catalog_size"] == 4
    assert results["peak_memory"] == {"status": "unmeasured"}
    assert set(results["models"]) == {"popularity", "tfidf", "random"}
    assert all(m["n_users"] == 2 for m in results["models"].values())
    assert results["models"]["popularity"]["recall_at_10"] == 1.0
    manifest = json.loads((run_dir / "eval_manifest.json").read_text("utf-8"))
    assert [u["user_id"] for u in manifest["users"]] == ["u1", "u6"]

    code = results["code"]
    actual = hashlib.sha256((REPO_ROOT / "src" / "run_baseline.py").read_bytes()).hexdigest()
    assert code["files"]["src/run_baseline.py"] == actual
    assert "src/metrics.py" in code["files"] and "requirements.txt" in code["files"]
    assert code["config"]["sha256"] == results["config_sha256"]
    assert len(code["combined_sha256"]) == 64
