"""질문 정의 로더와 로컬 해석 규칙(지역 확인, 해석 라벨 -> 로컬 조건).

- API로 보내는 것은 질문 정의 YAML(고정)뿐이다. 카테고리 매핑·지역 규칙은 로컬에서만 쓴다.
- 모호하거나 지원하지 않는 조건을 임의의 음식 종류로 바꾸지 않는다.
- confidence는 기록만 하고 조건 결정에 쓰지 않는다.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

QUESTION_ID = "cuisine"


@dataclass(frozen=True)
class QuerySpec:
    questions: dict                       # API로 보내는 고정 질문 정의 {cuisine: {...}}
    version: str
    sha256: str
    cuisine_categories: dict[str, list[str]]
    cuisine_query_terms: dict[str, str]
    other_label: str
    unspecified_label: str
    ambiguous_label: str
    supported_regions: dict[str, dict]
    unsupported_aliases: list[str]

    @property
    def labels(self) -> list[str]:
        return list(self.questions[QUESTION_ID]["criteria"])


def load_query_spec(cfg: dict) -> QuerySpec:
    """질문 YAML과 설정의 조건 매핑이 같은 라벨·순서인지 확인하고 묶는다."""
    path = Path(cfg["questions_file"])
    raw = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    questions = doc.get("questions") if isinstance(doc, dict) else None
    if not isinstance(questions, dict) or list(questions) != [QUESTION_ID]:
        raise ValueError(f"{path}: questions must define exactly [{QUESTION_ID!r}]")
    q = questions[QUESTION_ID]
    if q.get("type") != "choice" or not q.get("instructions") or not q.get("criteria"):
        raise ValueError(f"{path}: {QUESTION_ID} must be a choice with instructions and criteria")
    cond = cfg["conditions"]
    expected = list(cond["cuisine_categories"]) + [
        cond["other_label"], cond["unspecified_label"], cond["ambiguous_label"]
    ]
    if list(q["criteria"]) != expected:
        raise ValueError(f"{path}: criteria must be {expected} in this order")
    if set(cond["cuisine_query_terms"]) != set(cond["cuisine_categories"]):
        raise ValueError("conditions.cuisine_query_terms must cover every cuisine label")
    regions = cfg["regions"]
    return QuerySpec(
        questions=questions,
        version=str(doc.get("version")),
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        cuisine_categories={k: list(v) for k, v in cond["cuisine_categories"].items()},
        cuisine_query_terms=dict(cond["cuisine_query_terms"]),
        other_label=cond["other_label"],
        unspecified_label=cond["unspecified_label"],
        ambiguous_label=cond["ambiguous_label"],
        supported_regions=dict(regions["supported"]),
        unsupported_aliases=list(regions.get("unsupported_aliases") or []),
    )


def _mentions(text: str, alias: str) -> bool:
    """영문 별칭은 단어 경계로, 한글 별칭은 포함 여부로 찾는다 ('philly' 가 'phillyx' 안에서 잡히지 않게)."""
    pattern = r"(?<![a-z])" + re.escape(alias.lower()) + r"(?![a-z])"
    return re.search(pattern, text.lower()) is not None


def check_region(spec: QuerySpec, question: str) -> dict:
    """지역 규칙. 설정에 등록된 별칭만 인식한다 (API 호출 전 로컬 판단).

    - 등록된 미지원 별칭이 하나라도 있으면 unsupported_region
    - 등록된 지원 별칭이 있으면 그 도시로 필터
    - 그 외는 region_unspecified (PA 전체 후보). 지역 언급이 정말 없는 경우와 Boston처럼
      등록되지 않은 지명을 알아보지 못한 경우를 구분하지 못한다(한계).
    """
    unsupported = [a for a in spec.unsupported_aliases if _mentions(question, a)]
    if unsupported:
        return {"status": "unsupported_region", "region": None, "cities": None,
                "matched": unsupported}
    for key, region in spec.supported_regions.items():
        matched = [a for a in region["aliases"] if _mentions(question, a)]
        if matched:
            return {"status": "supported", "region": key, "cities": list(region["cities"]),
                    "matched": matched}
    return {"status": "region_unspecified", "region": None, "cities": None, "matched": []}


def plan_local_query(spec: QuerySpec, question: str, label: str) -> dict:
    """해석 라벨 -> 로컬 조건. 추천하지 않는 경우에도 이유를 상태로 돌려준다."""
    if label == spec.ambiguous_label:
        return {"status": "ambiguous_condition",
                "message": "multiple or unclear cuisines; ask the user to name one"}
    if label == spec.other_label:
        return {"status": "unsupported_condition",
                "message": f"supported cuisines: {list(spec.cuisine_categories)}"}
    if label == spec.unspecified_label:
        return {"status": "ok", "categories": None, "query_text": question}
    if label in spec.cuisine_categories:
        return {"status": "ok", "categories": spec.cuisine_categories[label],
                "query_text": f"{question} {spec.cuisine_query_terms[label]}"}
    raise ValueError(f"unknown label {label!r}")  # 응답 검사(schema_error)를 통과했다면 오지 않는다
