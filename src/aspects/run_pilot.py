"""리뷰 속성 분류 파일럿 실행.

  python -m src.aspects.run_pilot --config configs/aspects_pilot.yaml --mode connection-check
  python -m src.aspects.run_pilot --config configs/aspects_pilot.yaml --mode dev --keyword-only
  python -m src.aspects.run_pilot --config configs/aspects_pilot.yaml --mode dev
  python -m src.aspects.run_pilot --config configs/aspects_pilot.yaml --mode final

실행 전 차단 조건(네트워크 호출 없이 종료):
- connection-check: approval.connection_check 가 true 여야 한다. 합성 문장만 보낸다.
- dev/final API 실행: approval.real_review_transfer 가 true 여야 한다(개발 10개도 동일).
- dev/final: 해당 split의 사람 라벨이 모두 채워져 있어야 한다(모델 결과를 보기 전 라벨링).
- final: freeze 의 질문·규칙 해시가 현재 파일과 같아야 한다(개발 후 고정).
- API: 키 환경변수가 설정돼 있고, base URL 환경변수가 설정 파일의 expected_base_url 과 같아야 한다.
  주소·경로·모델·요청 ID 헤더·중단 오류 이름은 모두 설정 파일(api)에서 읽는다.
산출물: <output_root>/<시각>_<mode>/ (덮어쓰지 않음). 키 값은 어디에도 저장하지 않는다.

종료 코드 (results.json 의 status 와 대응, 결과 파일은 모든 경우 보존):
  0 completed                 정상 완료 (keyword-only의 의도된 not_called 포함)
  1 failed                    예외로 중단 (results.json 에 status=failed, 가능한 경우)
  2 refused                   실행 전 차단, 네트워크 호출 없음, 결과 폴더 없음
  3 completed_with_failures   끝까지 돌았지만 api_error·schema_error 등이 있음
  4 halted                    할당량 소진·인증 오류·예산·시간 상한으로 중간에 멈춤
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.aspects import ASPECTS
from src.aspects import keyword_rules
from src.aspects.client import (
    BudgetLedger, CallOutcome, SystemOneClient, load_questions, requests_transport,
)
from src.aspects.evaluate import load_labels, summarize
from src.aspects.sample import CSV_ENCODING, sample_sha256
from src.config import read_config, text_sha256
from src.prepare_data import file_sha256
from src.run_baseline import environment

REPO_ROOT = Path(__file__).resolve().parents[2]
MODES = ("connection-check", "dev", "final")
EXIT_CODES = {"completed": 0, "failed": 1, "refused": 2, "completed_with_failures": 3,
              "halted": 4}
# 연결 확인용 합성 문장 (실제 리뷰 아님)
SYNTHETIC_TEXTS = (
    "The pasta was delicious, but the waiter ignored us for twenty minutes.",
    "Great coffee and very friendly staff.",
    "The burger arrived cold and nobody came to check on our table.",
)
log = logging.getLogger("aspects.run_pilot")


class PilotRefused(RuntimeError):
    """승인·라벨·고정·환경 조건이 맞지 않아 실행하지 않음 (네트워크 호출 없음)."""


def code_fingerprint(paths: list[Path]) -> dict:
    """B2 실행 근거용: src/*.py, src/aspects/*.py, requirements*.txt, 사용한 설정·질문 파일."""
    files = sorted((REPO_ROOT / "src").glob("*.py")) + sorted(
        (REPO_ROOT / "src" / "aspects").glob("*.py")
    ) + sorted(REPO_ROOT.glob("requirements*.txt")) + list(paths)
    entries = {}
    for p in files:
        resolved = Path(p).resolve()
        key = (resolved.relative_to(REPO_ROOT).as_posix()
               if resolved.is_relative_to(REPO_ROOT) else str(resolved))
        entries[key] = file_sha256(resolved)
    combined = hashlib.sha256(
        "\n".join(f"{k}\t{v}" for k, v in sorted(entries.items())).encode("utf-8")
    ).hexdigest()
    return {"files": entries, "combined_sha256": combined}


def _load_sample(sample_dir: Path, split: str) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((sample_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    sample = pd.read_csv(sample_dir / "sample.csv", encoding=CSV_ENCODING, keep_default_na=False,
                         dtype={"review_id": str, "split": str, "text": str})
    if sample_sha256(sample) != manifest["sample_sha256"]:
        raise PilotRefused("sample.csv does not match sample_manifest.json (changed after creation)")
    return sample[sample["split"] == split].sort_values("sample_order"), manifest


def _check_gates(cfg: dict, mode: str, keyword_only: bool, q_sha: str, env: dict) -> None:
    approval = cfg.get("approval") or {}
    if mode == "connection-check":
        if keyword_only:
            raise PilotRefused("connection-check has no keyword-only mode")
        if approval.get("connection_check") is not True:
            raise PilotRefused("approval.connection_check is not true")
    elif not keyword_only and approval.get("real_review_transfer") is not True:
        raise PilotRefused("approval.real_review_transfer is not true (dev and final both need it)")
    if mode == "final":
        freeze = cfg.get("freeze") or {}
        if freeze.get("questions_sha256") != q_sha:
            raise PilotRefused("freeze.questions_sha256 is missing or differs from questions file")
        if freeze.get("keyword_rules_sha256") != keyword_rules.rules_fingerprint():
            raise PilotRefused("freeze.keyword_rules_sha256 is missing or differs from rules")
    if not keyword_only:
        api = cfg["api"]
        if not env.get(api["api_key_env"]):
            raise PilotRefused(f"environment variable {api['api_key_env']} is not set")
        base = (env.get(api["base_url_env"]) or "").rstrip("/")
        if base != api["expected_base_url"].rstrip("/"):
            raise PilotRefused(
                f"{api['base_url_env']} must equal {api['expected_base_url']} "
                "(no silent switch to another provider)"
            )


def run(config_path: str, mode: str, keyword_only: bool = False, transport=None,
        env: dict | None = None, sleep=time.sleep, clock=time.monotonic) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    env = dict(os.environ) if env is None else env
    cfg, raw = read_config(config_path)
    questions_path = Path(cfg["questions_file"])
    questions, q_version, q_sha = load_questions(questions_path)
    _check_gates(cfg, mode, keyword_only, q_sha, env)

    items: list[dict] = []
    labels: dict = {}
    sample_info: dict = {}
    if mode == "connection-check":
        items = [{"item": i, "review_id": None, "text": t} for i, t in enumerate(SYNTHETIC_TEXTS)]
    else:
        sample_dir = Path(cfg["sample"]["output_dir"])
        part, manifest = _load_sample(sample_dir, mode)
        labels_path = sample_dir / f"labels_{mode}.csv"
        try:
            labels = load_labels(labels_path)  # 빈칸이 있으면 여기서 멈춘다
        except ValueError as exc:
            raise PilotRefused(str(exc)) from exc
        if set(labels) != set(part["sample_order"].astype(int)):
            raise PilotRefused(f"{labels_path.name} items differ from the {mode} sample")
        items = [{"item": int(r.sample_order), "review_id": r.review_id, "text": r.text}
                 for r in part.itertuples(index=False)]
        sample_info = {
            "sample_sha256": manifest["sample_sha256"],
            "labels_file": labels_path.name,
            "labels_sha256": file_sha256(labels_path),
            "review_id_cells_differing_from_sample": sum(
                1 for it in items if labels[it["item"]]["review_id"] != it["review_id"]
            ),
        }

    suffix = "_keyword" if keyword_only else ""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{mode}{suffix}"
    run_dir = Path(cfg["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(raw, encoding="utf-8")
    (run_dir / "questions.yaml").write_text(
        questions_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()], force=True)

    api = cfg["api"]
    scope = "connection_check" if mode == "connection-check" else "pilot"
    limits = cfg["budget"][scope]
    ledger = BudgetLedger(cfg["budget"]["ledger_path"], scope, limits["max_requests"],
                          limits["max_estimated_input_tokens"])
    results: dict = {
        "status": "running",
        "run_id": run_id,
        "mode": mode,
        "keyword_only": keyword_only,
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": text_sha256(raw),
        "questions": {"file": questions_path.as_posix(), "version": q_version, "sha256": q_sha},
        "keyword_rules": {"version": keyword_rules.RULES_VERSION,
                          "sha256": keyword_rules.rules_fingerprint()},
        "sample": sample_info,
        "approval": cfg.get("approval"),
        "api": {"provider": api["provider"], "base_url": api["expected_base_url"],
                "endpoint_path": api["endpoint_path"],
                "request_id_header": api.get("request_id_header"),
                "halt_error_types": list(api.get("halt_error_types") or []),
                "model_requested": api["model"], "timeout_seconds": api["timeout_seconds"],
                "min_interval_seconds": api["min_interval_seconds"],
                "max_retries_after_first": api["max_retries"],
                "fields_sent": ["model", "state (processed review text)", "questions"]},
        "budget_before": ledger.totals(),
        "budget_limits": {"scope": scope, **limits},
        "code": code_fingerprint([Path(config_path), questions_path]),
        "environment": environment(),
        "cost": {"basis": "Codiv free tier per public docs (2026-10-22)",
                 "billed_amount": "unverified", "paid_fallback": "disabled"},
        "confidence_note": "confidence is spread of model probabilities, not calibrated accuracy",
    }
    try:
        client = None
        if not keyword_only:
            client = SystemOneClient(
                endpoint_url=env[api["base_url_env"]].rstrip("/") + api["endpoint_path"],
                api_key=env[api["api_key_env"]],
                request_id_header=api.get("request_id_header"),
                halt_error_types=api.get("halt_error_types") or (),
                model=api["model"], questions=questions, ledger=ledger, run_id=run_id,
                transport=transport or requests_transport, timeout=api["timeout_seconds"],
                min_interval=api["min_interval_seconds"], max_retries=api["max_retries"],
                deadline=clock() + limits["max_run_seconds"], sleep=sleep, clock=clock,
            )
        rows, records = [], []
        with open(run_dir / "responses.jsonl", "w", encoding="utf-8") as out:
            for it in items:
                outcome = (client.classify(it["text"]) if client
                           else CallOutcome(status="not_called", error="keyword_only"))
                record = {
                    "item": it["item"], "review_id": it["review_id"],
                    "text_sha256": text_sha256(it["text"]), "text_chars": len(it["text"]),
                    "status": outcome.status, "error": outcome.error, "model": outcome.model,
                    "usage": outcome.usage, "request_id": outcome.request_id,
                    "estimated_input_tokens": outcome.estimated_input_tokens,
                    "attempts": outcome.attempts, "answers": outcome.answers,
                    "raw_response": outcome.raw_response,
                }
                out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                records.append(record)
                log.info("item=%s status=%s error=%s", it["item"], outcome.status, outcome.error)
                if mode != "connection-check":
                    rows.append({
                        "item": it["item"], "review_id": it["review_id"], "text": it["text"],
                        "human": {a: labels[it["item"]][a] for a in ASPECTS},
                        "keyword": keyword_rules.classify(it["text"]),
                        "openjev_status": outcome.status,
                        "openjev": ({a: outcome.answers[a]["choice"] for a in ASPECTS}
                                    if outcome.status == "ok" else None),
                        "confidence": ({a: outcome.answers[a]["confidence"] for a in ASPECTS}
                                       if outcome.status == "ok" else None),
                    })
        latencies = [a["latency_ms"] for r in records for a in r["attempts"]
                     if a.get("http_status") is not None]
        known_usage = [r["usage"]["input_tokens"] for r in records
                       if isinstance(r["usage"], dict)
                       and isinstance(r["usage"].get("input_tokens"), int)]
        results["calls"] = {
            "items": len(records),
            "status_counts": {s: sum(1 for r in records if r["status"] == s)
                              for s in ("ok", "api_error", "schema_error", "not_called")},
            "attempts": sum(len(r["attempts"]) for r in records),
            "models_returned": sorted({r["model"] for r in records if r["model"]}),
            "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "usage_input_tokens_known_sum": sum(known_usage),
            "items_with_unknown_usage": len(records) - len(known_usage),
            # 시도마다 같은 추정치를 예약하므로 추정치 × 시도 수
            "estimated_input_tokens_sum": sum(r["estimated_input_tokens"] * len(r["attempts"])
                                              for r in records),
            "halted_reason": client.halted_reason if client else None,
        }
        if rows:
            _write_comparison(run_dir, rows)
            results["metrics"] = summarize(rows)
        results["budget_after"] = ledger.totals()
        halted = client.halted_reason if client else None
        results["status"] = "halted" if halted else (
            "completed" if all(r["status"] == "ok" for r in records) or keyword_only
            else "completed_with_failures"
        )
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = repr(exc)
        results["traceback"] = traceback.format_exc()
        results["budget_after"] = ledger.totals()
        _write_json(run_dir / "results.json", results)
        _close_log(handler)
        raise
    _write_json(run_dir / "results.json", results)
    _close_log(handler)
    return run_dir


def _write_comparison(run_dir: Path, rows: list[dict]) -> None:
    """예측 표와 실패·불일치 사례(로컬 review_id·처리 후 텍스트 포함)."""
    fields = ["item", "review_id", "aspect", "human", "keyword", "openjev_status", "openjev",
              "openjev_confidence"]
    with open(run_dir / "predictions.csv", "w", newline="", encoding=CSV_ENCODING) as pf, \
         open(run_dir / "failures.csv", "w", newline="", encoding=CSV_ENCODING) as ff:
        pw = csv.DictWriter(pf, fieldnames=fields)
        fw = csv.DictWriter(ff, fieldnames=fields + ["text"])
        pw.writeheader()
        fw.writeheader()
        for r in rows:
            for a in ASPECTS:
                line = {
                    "item": r["item"], "review_id": r["review_id"], "aspect": a,
                    "human": r["human"][a], "keyword": r["keyword"][a],
                    "openjev_status": r["openjev_status"],
                    "openjev": r["openjev"][a] if r["openjev"] else "",
                    "openjev_confidence": r["confidence"][a] if r["confidence"] else "",
                }
                pw.writerow(line)
                if (line["keyword"] != line["human"] or r["openjev_status"] != "ok"
                        or line["openjev"] != line["human"]):
                    fw.writerow({**line, "text": r["text"]})


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _close_log(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def _score(metrics: dict) -> str:
    """점수 표시. 표본이 0개면 0으로 바꾸지 않고 not_computable."""
    if metrics.get("status") == "not_computable":
        return "not_computable"
    return f"{metrics['macro_f1']:.4f}"


def format_summary(results: dict) -> list[str]:
    """콘솔 요약. OpenJev와 키워드의 직접 비교는 같은 ok 응답 표본(keyword_compared_rows)으로 한다."""
    calls = results["calls"]
    budget = results["budget_after"]
    lines = [
        f"[pilot] status={results['status']} calls={calls['status_counts']} "
        f"attempts={calls['attempts']} halted={calls['halted_reason']}",
        f"[pilot] models={calls['models_returned']} latency_p50_ms={calls['latency_ms_p50']} "
        f"usage_known={calls['usage_input_tokens_known_sum']} "
        f"usage_unknown_items={calls['items_with_unknown_usage']}",
        f"[pilot] budget {budget['scope']}: requests {budget['requests_reserved']}"
        f"/{budget['max_requests']}, estimated tokens {budget['estimated_input_tokens_reserved']}"
        f"/{budget['max_estimated_input_tokens']}",
    ]
    metrics = results.get("metrics")
    if metrics:
        ratio = metrics["compared_ratio"]
        lines.append(
            f"[pilot] compared rows (OpenJev ok only): {metrics['n_compared']}/{metrics['n_total']}"
            f" ({'n/a' if ratio is None else f'{ratio:.2f}'}) "
            f"status={metrics['openjev_status_counts']}"
        )
        for aspect, m in metrics["aspects"].items():
            kw_cmp, oj_cmp, kw_all = (m["keyword_compared_rows"], m["openjev_compared_rows"],
                                      m["keyword_all_rows"])
            lines.append(
                f"[pilot] {aspect} same rows n={oj_cmp['n']}: keyword macro_f1={_score(kw_cmp)} "
                f"openjev macro_f1={_score(oj_cmp)}"
            )
            lines.append(
                f"[pilot] {aspect} keyword on all labelled rows n={kw_all['n']}: "
                f"macro_f1={_score(kw_all)} (separate, not a comparison)"
            )
    return lines


def main(argv: list[str] | None = None, *, transport=None, env: dict | None = None,
         sleep=time.sleep, clock=time.monotonic) -> int:
    """종료 코드는 EXIT_CODES 참고. transport·env·sleep·clock은 시험에서만 바꾼다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--keyword-only", action="store_true",
                        help="dev/final: run keyword rules and labels only, no API call")
    args = parser.parse_args(argv)
    try:
        run_dir = run(args.config, args.mode, args.keyword_only, transport=transport, env=env,
                      sleep=sleep, clock=clock)
    except PilotRefused as exc:
        print(f"[pilot] refused (no request sent): {exc}")
        return EXIT_CODES["refused"]
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    print(f"[pilot] run_dir={run_dir}")
    for line in format_summary(results):
        print(line)
    code = EXIT_CODES.get(results["status"], EXIT_CODES["failed"])
    print(f"[pilot] exit_code={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
