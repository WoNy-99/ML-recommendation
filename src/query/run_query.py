"""사용자 질문 해석(OpenJev) -> 로컬 추천 실행.

  python -m src.query.run_query --config configs/query_demo.yaml --mode connection-check
  python -m src.query.run_query --config configs/query_demo.yaml --mode local \
      --question "Looking for a quiet italian dinner in Philly" --condition italian

- connection-check: 설정의 직접 작성 질문(최대 3개)만 API로 보낸다. 재시도 포함 최대 3요청.
  approval.query_connection_check 가 true 이고 키·주소 환경변수가 맞아야 한다(아니면 호출 없이 거부).
- local: API 없이, 사람이 고른 조건(--condition)으로 로컬 추천만 확인한다.
- 설정에 등록된 미지원 지역 별칭(예: New York)이 있는 질문은 API로 보내지 않는다.
  등록되지 않은 지명(예: Boston)은 알아보지 못해 region_unspecified(PA 전체 후보)로 처리된다(한계).
- 외부로 보내는 필드: model, state(사용자 질문), questions(고정 질문 정의).
  질문에 사용자가 직접 쓴 지명(Philly 등)은 질문의 일부로 전송된다. 로컬 Yelp 자료에서 나온
  도시·음식점·카테고리 목록, 리뷰, 후보, 추천 결과는 보내지 않는다.
- 실행 설정·질문 정의·baseline 설정은 시작 시점의 바이트를 실행 폴더에 사본으로 저장하고,
  로컬 추천은 그 baseline 사본으로 실행한다(원본을 나중에 바꿔도 과거 사본은 그대로).
- 콘솔에는 음식점 이름을 출력하지 않는다(로컬 CSV에만 저장).

종료 코드: 0 completed / 1 failed / 2 refused / 3 completed_with_failures / 4 halted
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

from src.aspects.client import BudgetLedger, CallOutcome, SystemOneClient, requests_transport
from src.config import read_config, text_sha256
from src.prepare_data import file_sha256
from src.query.interpret import QUESTION_ID, check_region, load_query_spec, plan_local_query
from src.query.recommend import load_local_index, recommend
from src.run_baseline import environment

REPO_ROOT = Path(__file__).resolve().parents[2]
MODES = ("connection-check", "local")
EXIT_CODES = {"completed": 0, "failed": 1, "refused": 2, "completed_with_failures": 3,
              "halted": 4}
CSV_ENCODING = "utf-8-sig"
log = logging.getLogger("query.run_query")


class QueryRefused(RuntimeError):
    """승인·환경·입력 조건이 맞지 않아 실행하지 않음 (네트워크 호출 없음)."""


def _key(path: Path) -> str:
    resolved = Path(path).resolve()
    return (resolved.relative_to(REPO_ROOT).as_posix()
            if resolved.is_relative_to(REPO_ROOT) else str(resolved))


def _normalized_text(data: bytes) -> str:
    """read_text 와 같은 줄바꿈 정규화로 바이트를 텍스트로 (일치 확인용)."""
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def code_fingerprint(config_snapshots: dict[Path, bytes]) -> dict:
    """src/**/*.py·requirements 파일 해시 + 실행 시작 때 읽어 둔 설정 바이트(실행·질문·baseline)의 해시."""
    files = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted(REPO_ROOT.glob("requirements*.txt"))
    entries = {_key(p): file_sha256(p) for p in files}
    for path, content in config_snapshots.items():
        entries[_key(path)] = hashlib.sha256(content).hexdigest()
    combined = hashlib.sha256(
        "\n".join(f"{k}\t{v}" for k, v in sorted(entries.items())).encode("utf-8")
    ).hexdigest()
    return {"files": entries, "combined_sha256": combined}


def _new_run_dir(root: Path, mode: str) -> tuple[str, Path]:
    """실행별 새 폴더. 같은 초에 다시 실행해도 기존 폴더를 덮지 않도록 번호를 붙인다."""
    base = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{mode}"
    root.mkdir(parents=True, exist_ok=True)
    for n in range(1000):
        run_id = base if n == 0 else f"{base}-{n}"
        try:
            (root / run_id).mkdir(exist_ok=False)
            return run_id, root / run_id
        except FileExistsError:
            continue
    raise RuntimeError("could not create a new run directory")


def _check_api_gates(cfg: dict, env: dict) -> None:
    if (cfg.get("approval") or {}).get("query_connection_check") is not True:
        raise QueryRefused("approval.query_connection_check is not true")
    api = cfg["api"]
    if not env.get(api["api_key_env"]):
        raise QueryRefused(f"environment variable {api['api_key_env']} is not set")
    if (env.get(api["base_url_env"]) or "").rstrip("/") != api["expected_base_url"].rstrip("/"):
        raise QueryRefused(f"{api['base_url_env']} must equal {api['expected_base_url']} "
                           "(no silent switch to another provider)")


def run(config_path: str, mode: str, question: str | None = None, condition: str | None = None,
        transport=None, env: dict | None = None, sleep=time.sleep, clock=time.monotonic,
        index_loader=load_local_index) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    env = dict(os.environ) if env is None else env
    cfg, raw = read_config(config_path)
    spec = load_query_spec(cfg)
    # 실행 설정·질문 정의·baseline 설정을 시작 시점에 바이트로 한 번 읽어 둔다.
    # 이 바이트를 실행 폴더에 그대로 저장하고, 로컬 추천은 저장한 baseline 사본으로 실행한다.
    snapshots = {
        "run_config": (Path(config_path), "config.yaml"),
        "questions": (Path(cfg["questions_file"]), "questions.yaml"),
        "baseline": (Path(cfg["baseline_config"]), "baseline.yaml"),
    }
    data = {name: src.read_bytes() for name, (src, _) in snapshots.items()}
    if _normalized_text(data["run_config"]) != raw or text_sha256(
        _normalized_text(data["questions"])
    ) != spec.sha256:
        raise QueryRefused("a config file changed while starting the run; run again")

    if mode == "connection-check":
        _check_api_gates(cfg, env)
        items = [{"n": i, "question": q["text"], "expected": q.get("expected")}
                 for i, q in enumerate(cfg["connection_check_questions"])]
        if not items:
            raise QueryRefused("connection_check_questions is empty")
    else:
        if not question or condition not in spec.labels:
            raise QueryRefused(f"local mode needs --question and --condition in {spec.labels}")
        items = [{"n": 0, "question": question, "expected": None}]

    run_id, run_dir = _new_run_dir(Path(cfg["output_root"]), mode)
    config_copies = {}
    for name, (src, copy_name) in snapshots.items():
        (run_dir / copy_name).write_bytes(data[name])   # 원본이 나중에 바뀌어도 이 사본은 그대로
        config_copies[name] = {"source": str(src), "copy": copy_name,
                               "sha256": hashlib.sha256(data[name]).hexdigest()}
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()], force=True)

    api = cfg["api"]
    limits = cfg["budget"]["connection_check"]
    ledger = BudgetLedger(cfg["budget"]["ledger_path"], "query_connection_check",
                          limits["max_requests"], limits["max_estimated_input_tokens"])
    results: dict = {
        "status": "running", "run_id": run_id, "mode": mode, "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": text_sha256(raw),
        # 사본 파일 바이트의 SHA-256. code.files 의 같은 원본 항목과 같은 값이다
        "config_copies": config_copies,
        "baseline_used": str(run_dir / "baseline.yaml"),
        "questions": {"file": cfg["questions_file"], "copy": "questions.yaml",
                      "version": spec.version, "text_sha256": spec.sha256,
                      "labels_in_order": spec.labels},
        "approval": cfg.get("approval"),
        "external_transfer": {
            "fields_sent": ["model", "state (user question only)",
                            "questions (fixed definition from questions_file)"],
            "note": "place names typed in the user question (e.g. Philly) are sent as part "
                    "of the question",
            "not_sent": ["city/business/category lists from local Yelp data", "Yelp reviews",
                         "user ids/history", "candidates", "recommendations", "scores"],
            "api_key": "used for authentication only; not written to logs/results/copies",
        },
        "api": {k: api[k] for k in ("provider", "expected_base_url", "endpoint_path", "model",
                                    "timeout_seconds", "min_interval_seconds", "max_retries")},
        "budget_limits": {"scope": "query_connection_check", **limits},
        "budget_before": ledger.totals(),
        "code": code_fingerprint({src: data[name] for name, (src, _) in snapshots.items()}),
        "environment": environment(),
        "confidence_note": "confidence is spread of model probabilities, not calibrated accuracy",
    }
    client = None
    try:
        if mode == "connection-check":
            client = SystemOneClient(
                endpoint_url=env[api["base_url_env"]].rstrip("/") + api["endpoint_path"],
                api_key=env[api["api_key_env"]], request_id_header=api.get("request_id_header"),
                halt_error_types=api.get("halt_error_types") or (), model=api["model"],
                questions=spec.questions, ledger=ledger, run_id=run_id,
                transport=transport or requests_transport, timeout=api["timeout_seconds"],
                min_interval=api["min_interval_seconds"], max_retries=api["max_retries"],
                deadline=clock() + limits["max_run_seconds"], sleep=sleep, clock=clock,
            )
        # 1단계: 지역 확인(로컬) + 질문 해석(API 또는 사람이 고른 조건)
        with open(run_dir / "responses.jsonl", "w", encoding="utf-8") as out:
            for it in items:
                it["region"] = check_region(spec, it["question"])
                if it["region"]["status"] == "unsupported_region":
                    outcome = CallOutcome(status="not_called", error="unsupported_region")
                elif client is not None:
                    outcome = client.classify(it["question"])
                else:
                    outcome = CallOutcome(status="not_called", error="local_mode")
                it["api_status"] = outcome.status
                it["label"] = (outcome.answers[QUESTION_ID]["choice"] if outcome.status == "ok"
                               else condition if mode == "local" else None)
                it["confidence"] = (outcome.answers[QUESTION_ID]["confidence"]
                                    if outcome.status == "ok" else None)
                out.write(json.dumps({
                    "n": it["n"], "question": it["question"], "expected": it["expected"],
                    "region": it["region"], "status": outcome.status, "error": outcome.error,
                    "model": outcome.model, "usage": outcome.usage,
                    "request_id": outcome.request_id,
                    "estimated_input_tokens": outcome.estimated_input_tokens,
                    "attempts": outcome.attempts, "answers": outcome.answers,
                    "raw_response": outcome.raw_response,
                }, ensure_ascii=False, default=str) + "\n")
                log.info("question=%s region=%s api_status=%s label=%s", it["n"],
                         it["region"]["status"], outcome.status, it["label"])

        # 2단계: 로컬 조건 적용과 TF-IDF 순위 (데이터는 필요할 때만 한 번 적재)
        usable = [it for it in items if it["label"] is not None
                  and it["region"]["status"] != "unsupported_region"]
        # 시작 시점에 저장한 baseline 사본으로 로컬 인덱스를 만든다(사본 = 실제 사용 설정)
        index = index_loader(str(run_dir / "baseline.yaml")) if usable else None
        if index is not None:
            results["local_index"] = index.info
        for it in items:
            it["local"] = _local_result(spec, index, it, cfg["top_k"], run_dir)

        results["items"] = [{
            "n": it["n"], "question": it["question"], "expected": it["expected"],
            "region": it["region"], "api_status": it["api_status"], "label": it["label"],
            "matches_expected": (None if it["expected"] is None or it["label"] is None
                                 else it["label"] == it["expected"]),
            "confidence": it["confidence"], "local": it["local"],
        } for it in items]
        results["budget_after"] = ledger.totals()
        results["halted_reason"] = client.halted_reason if client else None
        failed = any(it["api_status"] in ("api_error", "schema_error") for it in items)
        results["status"] = ("halted" if results["halted_reason"]
                             else "completed_with_failures" if failed else "completed")
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


def _local_result(spec, index, it: dict, k: int, run_dir: Path) -> dict:
    """로컬 조건 적용. 추천 목록은 로컬 CSV에만 쓰고 results.json에는 수·상태만 남긴다."""
    if it["region"]["status"] == "unsupported_region":
        return {"status": "unsupported_region",
                "message": "this demo only covers PA restaurants (supported city: Philadelphia)"}
    if it["label"] is None:
        return {"status": "not_interpreted", "message": f"api_status={it['api_status']}"}
    plan = plan_local_query(spec, it["question"], it["label"])
    if plan["status"] != "ok":
        return plan
    rec = recommend(index, plan["query_text"], it["region"]["cities"], plan["categories"], k)
    summary = {"status": rec["status"], "n_candidates": rec["n_candidates"],
               "n_recommended": len(rec["items"]),
               "filters": {"cities": it["region"]["cities"], "categories": plan["categories"]},
               "query_text_sha256": rec["query_text_sha256"]}
    if rec["items"]:
        path = run_dir / f"recommendations_{it['n']}.csv"
        with open(path, "w", newline="", encoding=CSV_ENCODING) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rec["items"][0]))
            writer.writeheader()
            writer.writerows(rec["items"])
        summary.update(file=path.name, all_top_scores_zero=rec["all_top_scores_zero"])
    elif "message" in rec:
        summary["message"] = rec["message"]
    return summary


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _close_log(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def format_summary(results: dict) -> list[str]:
    """콘솔 요약. 음식점 이름·ID는 출력하지 않는다."""
    budget = results["budget_after"]
    lines = [f"[query] status={results['status']} halted={results.get('halted_reason')} "
             f"budget requests {budget['requests_reserved']}/{budget['max_requests']}"]
    for it in results["items"]:
        local = it["local"]
        lines.append(
            f"[query] q{it['n']}: region={it['region']['status']} api={it['api_status']} "
            f"label={it['label']} expected={it['expected']} "
            f"confidence={it['confidence']} local={local['status']} "
            f"candidates={local.get('n_candidates')} recommended={local.get('n_recommended')}"
        )
    return lines


def main(argv: list[str] | None = None, *, transport=None, env: dict | None = None,
         sleep=time.sleep, clock=time.monotonic, index_loader=load_local_index) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--question", help="local mode: the user question")
    parser.add_argument("--condition", help="local mode: a label from the questions file")
    args = parser.parse_args(argv)
    try:
        run_dir = run(args.config, args.mode, args.question, args.condition, transport=transport,
                      env=env, sleep=sleep, clock=clock, index_loader=index_loader)
    except QueryRefused as exc:
        print(f"[query] refused (no request sent): {exc}")
        return EXIT_CODES["refused"]
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    print(f"[query] run_dir={run_dir}")
    for line in format_summary(results):
        print(line)
    code = EXIT_CODES.get(results["status"], EXIT_CODES["failed"])
    print(f"[query] exit_code={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
