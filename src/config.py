"""YAML 설정 읽기와 실행 전 검증."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

STAGES = ("validation", "test")
TARGET_DEFINITIONS = ("visit", "positive_visit")
MODELS = ("popularity", "tfidf", "random")


class ConfigError(ValueError):
    """설정이 비었거나 잘못되어 실행하면 안 되는 경우."""


@dataclass(frozen=True)
class BaselineSettings:
    stage: str
    seed: int
    data_dir: Path
    t_val: datetime
    t_test: datetime
    history_start: datetime | None
    target_definition: str
    positive_min_stars: int
    top_k: int
    max_eval_users: int | None
    models: tuple[str, ...]
    tfidf_max_features: int
    tfidf_ngram_range: tuple[int, int]
    tfidf_min_df: int
    tfidf_stop_words: str | None
    output_root: Path
    mlflow_enabled: bool
    mlflow_tracking_uri: str
    mlflow_experiment: str


def read_config(path: str | Path) -> tuple[dict, str]:
    """설정 dict와 원문(사본·해시용)을 함께 돌려준다."""
    raw = Path(path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        raise ConfigError("config root must be a mapping")
    return cfg, raw


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get(cfg: dict, dotted: str):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing config key: {dotted}")
        node = node[key]
    return node


def parse_day(value, key: str) -> datetime:
    """'YYYY-MM-DD' 문자열 또는 YAML이 만든 date를 그날 00:00:00으로 바꾼다."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ConfigError(f"{key} must be YYYY-MM-DD, got {value!r}") from exc
    raise ConfigError(f"{key} must be YYYY-MM-DD, got {value!r}")


def _positive_int(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer, got {value!r}")
    return value


def validate_prepare(cfg: dict) -> dict:
    """prepare_data 단계에 필요한 값만 검사한다."""
    zip_path = Path(_get(cfg, "data.zip_path"))
    if not zip_path.is_file():
        raise ConfigError(f"data.zip_path not found: {zip_path}")
    out = {
        "zip_path": zip_path,
        "output_dir": Path(_get(cfg, "data.output_dir")),
        "state": _get(cfg, "data.state"),
        "category": _get(cfg, "data.category"),
    }
    for key in ("state", "category"):
        if not isinstance(out[key], str) or not out[key].strip():
            raise ConfigError(f"data.{key} must be a non-empty string")
    return out


def validate_baseline(cfg: dict) -> BaselineSettings:
    """기준선 실행 전 검증. 사용자 확인 전이라 비워 둔 값(null)이 있으면 멈춘다."""
    pending = [
        key
        for key in ("split.t_val", "split.t_test", "eval.target_definition")
        if _get(cfg, key) is None
    ]
    if pending:
        raise ConfigError(
            "values awaiting user decision are null: " + ", ".join(pending)
            + " (fill them in after reviewing data_report.json)"
        )

    stage = _get(cfg, "experiment.stage")
    if stage not in STAGES:
        raise ConfigError(f"experiment.stage must be one of {STAGES}, got {stage!r}")
    t_val = parse_day(_get(cfg, "split.t_val"), "split.t_val")
    t_test = parse_day(_get(cfg, "split.t_test"), "split.t_test")
    if not t_val < t_test:
        raise ConfigError("split.t_val must be earlier than split.t_test")
    raw_start = _get(cfg, "split.history_start")
    history_start = None if raw_start is None else parse_day(raw_start, "split.history_start")
    if history_start is not None and not history_start < t_val:
        raise ConfigError("split.history_start must be earlier than split.t_val")

    target_definition = _get(cfg, "eval.target_definition")
    if target_definition not in TARGET_DEFINITIONS:
        raise ConfigError(
            f"eval.target_definition must be one of {TARGET_DEFINITIONS}, "
            f"got {target_definition!r}"
        )
    positive_min_stars = _get(cfg, "eval.positive_min_stars")
    if positive_min_stars not in (1, 2, 3, 4, 5):
        raise ConfigError("eval.positive_min_stars must be 1..5")
    raw_max_users = _get(cfg, "eval.max_eval_users")
    max_eval_users = (
        None if raw_max_users is None else _positive_int(raw_max_users, "eval.max_eval_users")
    )

    models = _get(cfg, "models")
    if not isinstance(models, list) or not models or any(m not in MODELS for m in models):
        raise ConfigError(f"models must be a non-empty list from {MODELS}, got {models!r}")
    if len(set(models)) != len(models):
        raise ConfigError("models must not contain duplicates")

    ngram = _get(cfg, "tfidf.ngram_range")
    if (
        not isinstance(ngram, list) or len(ngram) != 2
        or not all(isinstance(n, int) and n >= 1 for n in ngram) or ngram[0] > ngram[1]
    ):
        raise ConfigError("tfidf.ngram_range must be [min_n, max_n] with 1 <= min_n <= max_n")
    stop_words = _get(cfg, "tfidf.stop_words")
    if stop_words not in (None, "english"):
        raise ConfigError("tfidf.stop_words must be 'english' or null")

    data_dir = Path(_get(cfg, "data.output_dir"))
    for name in ("businesses.jsonl", "reviews.jsonl.gz", "prepare_manifest.json"):
        if not (data_dir / name).is_file():
            raise ConfigError(f"prepared file missing: {data_dir / name} (run src.prepare_data)")

    return BaselineSettings(
        stage=stage,
        seed=_positive_int(_get(cfg, "experiment.seed"), "experiment.seed"),
        data_dir=data_dir,
        t_val=t_val,
        t_test=t_test,
        history_start=history_start,
        target_definition=target_definition,
        positive_min_stars=positive_min_stars,
        top_k=_positive_int(_get(cfg, "eval.top_k"), "eval.top_k"),
        max_eval_users=max_eval_users,
        models=tuple(models),
        tfidf_max_features=_positive_int(_get(cfg, "tfidf.max_features"), "tfidf.max_features"),
        tfidf_ngram_range=(ngram[0], ngram[1]),
        tfidf_min_df=_positive_int(_get(cfg, "tfidf.min_df"), "tfidf.min_df"),
        tfidf_stop_words=stop_words,
        output_root=Path(_get(cfg, "tracking.output_root")),
        mlflow_enabled=bool(_get(cfg, "tracking.mlflow")),
        mlflow_tracking_uri=str(_get(cfg, "tracking.mlflow_tracking_uri")),
        mlflow_experiment=str(_get(cfg, "tracking.mlflow_experiment")),
    )
