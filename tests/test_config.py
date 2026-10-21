import copy
from datetime import datetime

import pytest
import yaml

from src.config import ConfigError, validate_baseline

BASE = {
    "experiment": {"stage": "validation", "seed": 42},
    "data": {"zip_path": "Yelp-JSON.zip", "output_dir": "data/x", "state": "PA",
             "category": "Restaurants"},
    "split": {"t_val": None, "t_test": None, "history_start": None},
    "eval": {"target_definition": None, "positive_min_stars": 4, "top_k": 10,
             "max_eval_users": None},
    "models": ["popularity", "tfidf"],
    "tfidf": {"max_features": 100, "ngram_range": [1, 1], "min_df": 1, "stop_words": None},
    "tracking": {"output_root": "artifacts/runs", "mlflow": False,
                 "mlflow_tracking_uri": "sqlite:///mlflow.db", "mlflow_experiment": "t"},
}


def _cfg(**changes):
    cfg = copy.deepcopy(BASE)
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        cfg[section][key] = value
    return cfg


def test_pending_user_decisions_block_the_run():
    with pytest.raises(ConfigError, match="split.t_val, split.t_test, eval.target_definition"):
        validate_baseline(_cfg())


def test_t_val_must_precede_t_test():
    cfg = _cfg(split__t_val="2021-07-01", split__t_test="2021-01-01",
               eval__target_definition="visit")
    with pytest.raises(ConfigError, match="earlier"):
        validate_baseline(cfg)


def test_unknown_target_definition_is_rejected():
    cfg = _cfg(split__t_val="2021-01-01", split__t_test="2021-07-01",
               eval__target_definition="liked")
    with pytest.raises(ConfigError, match="target_definition"):
        validate_baseline(cfg)


def test_valid_config_with_yaml_dates(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    for name in ("businesses.jsonl", "reviews.jsonl.gz", "prepare_manifest.json"):
        (data_dir / name).write_text("", encoding="utf-8")
    text = yaml.safe_dump(_cfg(eval__target_definition="visit", data__output_dir=str(data_dir)))
    text = text.replace("t_val: null", "t_val: 2021-01-01").replace(
        "t_test: null", "t_test: 2021-07-01"
    )  # 따옴표 없는 날짜는 YAML 이 date 로 읽는다
    settings = validate_baseline(yaml.safe_load(text))
    assert settings.t_val == datetime(2021, 1, 1)
    assert settings.t_test == datetime(2021, 7, 1)
    assert settings.models == ("popularity", "tfidf")


def test_missing_prepared_files_are_reported(tmp_path):
    cfg = _cfg(split__t_val="2021-01-01", split__t_test="2021-07-01",
               eval__target_definition="visit", data__output_dir=str(tmp_path / "none"))
    with pytest.raises(ConfigError, match="prepared file missing"):
        validate_baseline(cfg)
