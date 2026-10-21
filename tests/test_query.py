"""질문 해석 -> 로컬 추천 흐름. 합성 음식점 자료와 가짜 네트워크만 쓴다(실제 Yelp 자료·API 없음)."""
import copy
import csv
import hashlib
import json

import pandas as pd
import pytest
import yaml

from aspects_fakes import FAKE_KEY, REPO, FakeClock, FakeTransport, error
from src.aspects.client import TransportResponse
from src.config import read_config
from src.query.interpret import QUESTION_ID, check_region, load_query_spec, plan_local_query
from src.query.recommend import build_index, recommend
from src.query.run_query import main

BASE_CFG, _ = read_config(REPO / "configs" / "query_demo.yaml")
BASE_CFG["questions_file"] = str(REPO / "configs" / "questions" / "query_v1.yaml")
SPEC = load_query_spec(BASE_CFG)
ENV = {"TYPESAFE_API_KEY": FAKE_KEY, "TYPESAFE_BASE_URL": "https://example.invalid"}

BUSINESSES = pd.DataFrame([
    {"business_id": "b1", "name": "Trattoria Uno", "city": "Philadelphia",
     "categories": "Restaurants, Italian", "state": "PA"},
    {"business_id": "b2", "name": "Pasta Due", "city": "Philadelphia",
     "categories": "Italian, Restaurants", "state": "PA"},
    {"business_id": "b3", "name": "Sushi Tre", "city": "Philadelphia",
     "categories": "Japanese, Sushi Bars, Restaurants", "state": "PA"},
    {"business_id": "b4", "name": "Pizza Quattro", "city": "Pittsburgh",
     "categories": "Italian, Restaurants", "state": "PA"},
    {"business_id": "b5", "name": "Taco Cinque", "city": "Philadelphia",
     "categories": "Mexican, Restaurants", "state": "PA"},
])
HISTORY = pd.DataFrame([
    {"review_id": f"r{i}", "user_id": f"u{i}", "business_id": b, "stars": 5, "text": t}
    for i, (b, t) in enumerate([
        ("b1", "great pizza and loud music"),
        ("b2", "quiet romantic dinner with fresh pasta"),
        ("b3", "fresh sushi and quiet atmosphere"),
        ("b4", "pizza pizza"),
        ("b5", "tacos and margaritas"),
    ])
])
TFIDF = {"max_features": 1000, "ngram_range": [1, 1], "min_df": 1, "stop_words": None}


def synthetic_index():
    return build_index(HISTORY, BUSINESSES, TFIDF)


def answer(choice):
    labels = SPEC.labels
    probs = {label: (0.95 if label == choice else 0.05 / (len(labels) - 1)) for label in labels}
    return TransportResponse(200, {}, {
        "model": "openjev-0.1",
        "answers": {QUESTION_ID: {"type": "choice", "choice": choice, "probabilities": probs,
                                  "confidence": 0.8}},
        "usage": {"input_tokens": 200, "output_tokens": 0},
    })


# ---------- 로컬 규칙 ----------

def test_spec_matches_questions_file_and_rejects_reordered_labels(tmp_path):
    assert SPEC.labels == ["italian", "japanese", "mexican", "other_cuisine", "unspecified",
                           "ambiguous"]
    doc = yaml.safe_load((REPO / "configs" / "questions" / "query_v1.yaml").read_text("utf-8"))
    criteria = doc["questions"]["cuisine"]["criteria"]
    doc["questions"]["cuisine"]["criteria"] = dict(reversed(list(criteria.items())))
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="in this order"):
        load_query_spec({**BASE_CFG, "questions_file": str(bad)})


@pytest.mark.parametrize("question, status, region", [
    ("필라델피아에서 이탈리안 음식점을 추천해줘", "supported", "philadelphia"),
    ("Looking for sushi in Philly", "supported", "philadelphia"),
    ("Best ramen in New York", "unsupported_region", None),
    ("뉴욕에서 피자 추천해줘", "unsupported_region", None),
    ("Looking for tacos", "region_unspecified", None),
    ("phillyx cheesesteak", "region_unspecified", None),   # 단어 경계
    # 한계: 등록되지 않은 지명은 알아보지 못해 '지역 언급 없음'과 같게 처리된다(PA 전체 후보)
    ("Best pizza in Boston", "region_unspecified", None),
])
def test_region_rules(question, status, region):
    result = check_region(SPEC, question)
    assert result["status"] == status and result["region"] == region


def test_labels_map_to_local_conditions_without_guessing():
    assert plan_local_query(SPEC, "q", "ambiguous")["status"] == "ambiguous_condition"
    assert plan_local_query(SPEC, "q", "other_cuisine")["status"] == "unsupported_condition"
    unspecified = plan_local_query(SPEC, "quiet dinner", "unspecified")
    assert unspecified["categories"] is None and unspecified["query_text"] == "quiet dinner"
    italian = plan_local_query(SPEC, "조용한 곳", "italian")
    assert italian["categories"] == ["Italian"] and "italian" in italian["query_text"]


def test_local_filter_and_ranking_come_from_local_data():
    index = synthetic_index()
    rec = recommend(index, "quiet dinner italian", ["Philadelphia"], ["Italian"], 10)
    assert rec["status"] == "recommended" and rec["n_candidates"] == 2   # b4는 Pittsburgh
    assert [item["business_id"] for item in rec["items"]] == ["b2", "b1"]  # b2 리뷰에 quiet dinner
    assert rec["items"][0]["score"] > rec["items"][1]["score"]


def test_empty_candidates_and_empty_query_vector():
    index = synthetic_index()
    assert recommend(index, "mexican", ["Pittsburgh"], ["Mexican"], 10)["status"] == \
        "no_candidates"
    empty = recommend(index, "아무 데나 추천해줘", ["Philadelphia"], None, 10)
    assert empty["status"] == "empty_query_vector" and empty["items"] == []


# ---------- 실행 흐름 ----------

def write_config(tmp_path, approval=True, questions=None):
    """시험용 설정 일체를 임시 폴더에 만든다(저장소 설정 파일은 건드리지 않음)."""
    cfg = copy.deepcopy(BASE_CFG)
    questions_copy = tmp_path / "questions_src.yaml"
    questions_copy.write_bytes((REPO / "configs" / "questions" / "query_v1.yaml").read_bytes())
    baseline_copy = tmp_path / "baseline_src.yaml"
    baseline_copy.write_bytes((REPO / "configs" / "baseline.yaml").read_bytes())
    cfg["questions_file"] = str(questions_copy)
    cfg["baseline_config"] = str(baseline_copy)
    cfg["approval"] = {"query_connection_check": approval}
    cfg["api"].update(expected_base_url="https://example.invalid", endpoint_path="/v9/query")
    cfg["budget"]["ledger_path"] = str(tmp_path / "ledger.jsonl")
    cfg["output_root"] = str(tmp_path / "runs")
    cfg["connection_check_questions"] = questions or [
        {"text": "필라델피아에서 이탈리안 음식점을 추천해줘", "expected": "italian"},
        {"text": "Best ramen in New York", "expected": "japanese"},
        {"text": "Looking for a quiet place for dinner in Philly", "expected": "unspecified"},
    ]
    path = tmp_path / "query.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


class Loader:
    def __init__(self):
        self.calls = 0
        self.paths = []

    def __call__(self, baseline_config_path):
        self.calls += 1
        self.paths.append(baseline_config_path)
        return synthetic_index()


def _main(config, transport=None, env=ENV, loader=None, extra=()):
    clock = FakeClock()
    return main(["--config", config, "--mode", "connection-check", *extra] if not extra
                else ["--config", config, *extra],
                transport=transport, env=env, sleep=clock.sleep, clock=clock,
                index_loader=loader or Loader())


def _results(tmp_path):
    run_dir = sorted((tmp_path / "runs").iterdir())[-1]
    return run_dir, json.loads((run_dir / "results.json").read_text("utf-8"))


def test_connection_check_sends_only_question_and_fixed_schema(tmp_path):
    config = write_config(tmp_path)
    fake = FakeTransport([answer("italian"), answer("unspecified")])
    loader = Loader()
    assert _main(config, fake, loader=loader) == 0
    # 뉴욕 질문은 로컬 지역 규칙으로 걸러져 보내지 않음 -> 호출 2회
    assert [c["payload"]["state"] for c in fake.calls] == [
        "필라델피아에서 이탈리안 음식점을 추천해줘",
        "Looking for a quiet place for dinner in Philly",
    ]
    for call in fake.calls:
        assert set(call["payload"]) == {"model", "state", "questions"}
        assert call["payload"]["questions"] == SPEC.questions
        sent = json.dumps(call["payload"], ensure_ascii=False)
        for local_only in ("Trattoria", "romantic", "b1", "Sushi Bars", "Philadelphia"):
            assert local_only not in sent                 # 로컬 자료·카테고리 매핑 미전송
        assert call["url"] == "https://example.invalid/v9/query"
    assert loader.calls == 1
    run_dir, results = _results(tmp_path)
    q0, q1, q2 = results["items"]
    assert q0["label"] == "italian" and q0["matches_expected"] is True
    assert q0["local"]["status"] == "recommended"
    assert q0["local"]["filters"] == {"cities": ["Philadelphia"], "categories": ["Italian"]}
    with open(run_dir / "recommendations_0.csv", newline="", encoding="utf-8-sig") as fh:
        assert {r["business_id"] for r in csv.DictReader(fh)} == {"b1", "b2"}  # 해석이 필터에 반영
    assert q1["api_status"] == "not_called" and q1["local"]["status"] == "unsupported_region"
    assert q2["label"] == "unspecified" and q2["local"]["n_candidates"] == 4
    assert results["budget_after"]["requests_reserved"] == 2
    for path in list(run_dir.iterdir()) + [tmp_path / "ledger.jsonl"]:
        assert FAKE_KEY not in path.read_text(encoding="utf-8-sig")


def test_ambiguous_and_unsupported_cuisine_are_not_recommended(tmp_path):
    config = write_config(tmp_path, questions=[
        {"text": "이탈리안이나 멕시칸 아무거나", "expected": "ambiguous"},
        {"text": "Korean BBQ in Philly", "expected": "other_cuisine"},
    ])
    fake = FakeTransport([answer("ambiguous"), answer("other_cuisine")])
    assert _main(config, fake) == 0
    _, results = _results(tmp_path)
    assert [it["local"]["status"] for it in results["items"]] == [
        "ambiguous_condition", "unsupported_condition"]
    assert all(it["local"].get("n_recommended") is None for it in results["items"])


def test_refused_without_approval_makes_no_call(tmp_path):
    config = write_config(tmp_path, approval=False)
    fake = FakeTransport([])
    assert _main(config, fake) == 2
    assert fake.calls == [] and not (tmp_path / "runs").exists()


def test_api_failures_keep_results_and_skip_local_data(tmp_path):
    config = write_config(tmp_path)
    loader = Loader()
    fake = FakeTransport([], default=error(400, "invalid_request_error"))
    assert _main(config, fake, loader=loader) == 3
    _, results = _results(tmp_path)
    assert results["status"] == "completed_with_failures"
    assert [it["local"]["status"] for it in results["items"]] == [
        "not_interpreted", "unsupported_region", "not_interpreted"]
    assert loader.calls == 0


def test_schema_error_is_not_turned_into_a_condition(tmp_path):
    config = write_config(tmp_path)
    bad = answer("italian")
    bad.body["answers"][QUESTION_ID]["choice"] = "french"
    fake = FakeTransport([bad, answer("unspecified")])
    assert _main(config, fake) == 3
    _, results = _results(tmp_path)
    assert results["items"][0]["api_status"] == "schema_error"
    assert results["items"][0]["label"] is None


def test_quota_exceeded_halts_and_later_questions_are_not_called(tmp_path):
    config = write_config(tmp_path)
    fake = FakeTransport([error(429, "quota_exceeded_error")])
    assert _main(config, fake) == 4
    _, results = _results(tmp_path)
    assert results["status"] == "halted" and len(fake.calls) == 1
    assert results["items"][2]["api_status"] == "not_called"


def test_budget_limit_is_shared_across_runs(tmp_path):
    config = write_config(tmp_path)
    first = FakeTransport([answer("italian"), answer("unspecified")])
    assert _main(config, first) == 0                 # 2요청 사용
    second = FakeTransport([answer("italian"), answer("unspecified")])
    assert _main(config, second) == 4                # 3번째 요청 후 상한 도달
    assert len(second.calls) == 1


def test_local_mode_needs_no_api_and_no_key(tmp_path):
    config = write_config(tmp_path, approval=False)
    code = _main(config, env={}, extra=("--mode", "local", "--question",
                                        "quiet dinner in Philly", "--condition", "italian"))
    assert code == 0
    _, results = _results(tmp_path)
    item = results["items"][0]
    assert item["api_status"] == "not_called" and item["local"]["status"] == "recommended"
    assert results["budget_after"]["requests_reserved"] == 0


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_run_folder_keeps_exact_config_copies_linked_by_hash(tmp_path):
    config = write_config(tmp_path)
    sources = {"run_config": tmp_path / "query.yaml",
               "questions": tmp_path / "questions_src.yaml",
               "baseline": tmp_path / "baseline_src.yaml"}
    originals = {k: p.read_bytes() for k, p in sources.items()}
    loader = Loader()
    fake = FakeTransport([answer("italian"), answer("unspecified")])
    assert _main(config, fake, loader=loader) == 0
    run_dir, results = _results(tmp_path)
    copies = results["config_copies"]
    for name, src in sources.items():
        copy_bytes = (run_dir / copies[name]["copy"]).read_bytes()
        assert copy_bytes == originals[name]                       # 사용한 내용 그대로
        assert copies[name]["sha256"] == _sha(copy_bytes)
        assert results["code"]["files"][str(src.resolve())] == _sha(copy_bytes)
    assert loader.paths == [str(run_dir / "baseline.yaml")]       # 로컬 추천은 사본으로 실행
    assert results["baseline_used"] == str(run_dir / "baseline.yaml")
    for call in fake.calls:                                       # 전송 내용은 그대로
        assert set(call["payload"]) == {"model", "state", "questions"}
    for name in ("config.yaml", "questions.yaml", "baseline.yaml", "results.json", "run.log"):
        assert FAKE_KEY not in (run_dir / name).read_text(encoding="utf-8-sig")


def test_copies_survive_later_edits_and_baseline_change_changes_identity(tmp_path):
    config = write_config(tmp_path)
    assert _main(config, FakeTransport([answer("italian"), answer("unspecified")])) == 0
    first_dir, first = _results(tmp_path)
    kept = {n: (first_dir / n).read_bytes() for n in ("questions.yaml", "baseline.yaml")}
    baseline_src = tmp_path / "baseline_src.yaml"
    baseline_src.write_bytes(baseline_src.read_bytes() + b"\n# edited after the first run\n")
    (tmp_path / "questions_src.yaml").write_bytes(
        (tmp_path / "questions_src.yaml").read_bytes() + b"\n# edited\n")
    assert {n: (first_dir / n).read_bytes() for n in kept} == kept   # 과거 사본은 그대로
    # 누적 예산(3요청)에서 1요청만 남아 두 번째 실행은 중간에 멈추지만, 식별 정보는 기록된다
    assert _main(config, FakeTransport([answer("italian")])) == 4
    second_dir, second = _results(tmp_path)
    assert second_dir != first_dir
    assert second["config_copies"]["baseline"]["sha256"] != \
        first["config_copies"]["baseline"]["sha256"]
    assert second["code"]["combined_sha256"] != first["code"]["combined_sha256"]
    assert (second_dir / "baseline.yaml").read_bytes() == baseline_src.read_bytes()


def test_local_mode_rejects_unknown_condition(tmp_path):
    config = write_config(tmp_path)
    code = _main(config, env={}, extra=("--mode", "local", "--question", "q",
                                        "--condition", "french"))
    assert code == 2
