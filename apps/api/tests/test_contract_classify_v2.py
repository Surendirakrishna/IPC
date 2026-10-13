"""Contract tests for the classify response v2 (neighbour vote), response contract 2.0.0.

The contract lives in four places that must not drift apart:
  1. app/schemas/classify.py          (source of truth)
  2. docs/api.md                      (worked example in the `contract-example` marked JSON blocks)
  3. apps/web/lib/types.ts            (hand-written mirror, parsed here as text)
  4. apps/web/tests/fixtures.ts       (Playwright fixtures, transpiled and evaluated with the web repo's own
                                       TypeScript, so the test sees exactly the objects the browser tests send)

Prose: vault note 02-Contracts/classify-response-v2.md. Every negative case asserts the validator's error
text, not just "it raised", so a case cannot pass for the wrong reason.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.schemas import classify as C
from app.schemas.classify import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorBody,
    PatentCorpusUnavailableDetails,
)

REPO = Path(__file__).resolve().parents[3]
DOCS_API = REPO / "docs" / "api.md"
WEB = REPO / "apps" / "web"
TYPES_TS = WEB / "lib" / "types.ts"
FIXTURES_TS = WEB / "tests" / "fixtures.ts"

# The contract's value sets, written out here on purpose: removing or renaming a value in the schema must fail.
CLASSIFIERS = {"neighbour_vote", "definition_fallback"}
FALLBACK_REASONS = {
    "neighbour_vote_disabled",
    "no_serving_corpus",
    "patent_encoder_unavailable",
    "vote_empty",
    "vote_below_minimum",
    "vote_failed",
}
CONFIDENCE_STATUSES = {
    "calibrated",
    "vote_share_uncalibrated",
    "not_calibrated",
    "incompatible_artifact",
    "unsupported_input_mode",
}
VOTE_CODE_LEVELS = {"subclass", "maingroup", "subgroup"}


# ----------------------------------------------------------------------------------------------- helpers


def _docs_block(marker: str) -> dict:
    text = DOCS_API.read_text(encoding="utf-8")
    m = re.search(rf"<!-- contract-example: {marker} -->\s*```json\n(.*?)```", text, re.S)
    assert m, f"docs/api.md has no `<!-- contract-example: {marker} -->` JSON block"
    return json.loads(m.group(1))


@pytest.fixture(scope="module")
def docs_response() -> dict:
    return _docs_block("classify-response")


def ex() -> dict:
    """A fresh copy of the docs/api.md worked example (vote first code H04L 9/08, subgroup)."""
    return copy.deepcopy(_docs_block("classify-response"))


def _unwrap(annotation: Any) -> list[tuple[type[BaseModel], bool]]:
    """BaseModel classes reachable from an annotation, with whether they sit inside a list."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return [x for a in typing.get_args(annotation) if a is not type(None) for x in _unwrap(a)]
    if origin is list:
        return [(m, True) for m, _ in _unwrap(typing.get_args(annotation)[0])]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [(annotation, False)]
    return []


def key_drift(model: type[BaseModel], data: Any, path: str = "$") -> list[str]:
    """Differences between the JSON keys in `data` and the Pydantic field names, at every nesting level.

    Dict-typed fields (source_snapshot, retrieval_settings, extracted_terms items, ...) have free-form keys
    and are not descended into.
    """
    if not isinstance(data, dict):
        return [f"{path}: expected an object for {model.__name__}, got {type(data).__name__}"]
    problems: list[str] = []
    fields = set(model.model_fields)
    for missing in sorted(fields - data.keys()):
        problems.append(f"{path}: missing field {missing!r} ({model.__name__})")
    for extra in sorted(data.keys() - fields):
        problems.append(f"{path}: unexpected field {extra!r} ({model.__name__})")
    for name, info in model.model_fields.items():
        if name not in data or data[name] is None:
            continue
        nested = _unwrap(info.annotation)
        if len(nested) != 1:
            continue
        sub, is_list = nested[0]
        if is_list:
            for i, item in enumerate(data[name]):
                problems += key_drift(sub, item, f"{path}.{name}[{i}]")
        else:
            problems += key_drift(sub, data[name], f"{path}.{name}")
    return problems


def model_dicts(model: type[BaseModel], data: Any, path: tuple = ()) -> list[tuple[tuple, dict]]:
    """Every (path, dict) in `data` that corresponds to a Pydantic model, for mutation tests."""
    out: list[tuple[tuple, dict]] = [(path, data)]
    for name, info in model.model_fields.items():
        if data.get(name) is None:
            continue
        nested = _unwrap(info.annotation)
        if len(nested) != 1:
            continue
        sub, is_list = nested[0]
        if is_list:
            for i, item in enumerate(data[name]):
                out += model_dicts(sub, item, (*path, name, i))
        else:
            out += model_dicts(sub, data[name], (*path, name))
    return out


def at(data: Any, path: tuple) -> Any:
    for p in path:
        data = data[p]
    return data


def rename_key(data: dict, path: tuple, old: str, new: str) -> dict:
    """Return a deep COPY of `data` with `old` renamed to `new` at `path`. The input is never modified."""
    clone = copy.deepcopy(data)
    holder = at(clone, path)
    holder[new] = holder.pop(old)
    return clone


def rejected(payload: dict, message_part: str) -> None:
    with pytest.raises(ValidationError) as err:
        ClassifyResponse.model_validate(payload)
    assert message_part in str(err.value), f"rejected, but for a different reason:\n{err.value}"


def accepted(payload: dict) -> ClassifyResponse:
    return ClassifyResponse.model_validate(payload)


def _as_alternative(candidate: dict, reason: str) -> dict:
    keys = ("code", "display_code", "title", "level", "confidence", "vote_share", "scores")
    return {
        **{k: copy.deepcopy(candidate[k]) for k in keys},
        "reason": reason,
        "unresolved_ambiguity": [],
        "supporting_passages": [],
    }


def _uncal(score: float) -> dict:
    return {
        "score": score,
        "status": "vote_share_uncalibrated",
        "calibrator_id": None,
        "interpretation": None,
        "detail": None,
    }


def with_lead(level: str, display: str, share: dict, score: float, *, code: str | None = None) -> dict:
    """The docs example with candidates[0] replaced by a coarser (or other) code, as in TASK-0001 follow-up 2."""
    d = ex()
    lead = d["candidates"][0]
    lead["code"] = code or display
    lead["display_code"] = display
    lead["level"] = level
    lead["vote_share"] = share
    lead["confidence"] = _uncal(score)
    return d


def fallback(reason: str) -> dict:
    """A coherent definition_fallback response for `reason`, derived from the docs example."""
    d = ex()
    d["classifier"] = "definition_fallback"
    d["fallback_reason"] = reason
    for c in d["candidates"]:
        c["vote_share"] = {"subclass": None, "maingroup": None, "exact": None}
        c["confidence"] = {
            "score": None,
            "status": "not_calibrated",
            "calibrator_id": None,
            "interpretation": None,
            "detail": None,
        }
        c["origin"] = ["reranker"]
    for a in d["alternatives"]:
        a["vote_share"] = {"subclass": None, "maingroup": None, "exact": None}
        a["confidence"] = copy.deepcopy(d["candidates"][0]["confidence"])
    if reason == "vote_below_minimum":
        d["provenance"]["vote_settings"]["min_vote_share"] = 0.3
        d["vote"]["first_code_share"] = {"subclass": 0.2, "maingroup": 0.15, "exact": 0.1}
    else:
        d["vote"] = None
        if reason == "no_serving_corpus":
            d["provenance"]["patent_corpus"] = None
        if reason in ("no_serving_corpus", "neighbour_vote_disabled"):
            d["provenance"]["vote_settings"] = None
        if reason == "neighbour_vote_disabled":
            d["provenance"]["patent_encoder"] = None
    return d


# ------------------------------------------------------------------------------- 1. docs/api.md example


def test_docs_request_example_validates():
    ClassifyRequest.model_validate(_docs_block("classify-request"))


def test_docs_response_example_validates(docs_response):
    resp = ClassifyResponse.model_validate(docs_response)
    assert resp.classifier == "neighbour_vote"
    assert resp.fallback_reason is None
    assert resp.vote is not None and resp.vote.first_code == "H04L 9/08"
    assert [c.display_code for c in resp.candidates] == ["H04L 9/08", "H04L 9/32"]


def test_docs_response_example_round_trips_exactly(docs_response):
    """The example uses every field and no extra one: dumping the parsed model reproduces the JSON."""
    resp = ClassifyResponse.model_validate(docs_response)
    assert json.loads(resp.model_dump_json()) == docs_response


def test_docs_response_example_keys_match_pydantic_at_every_level(docs_response):
    assert key_drift(ClassifyResponse, docs_response) == []


def test_docs_example_exercises_the_vote_fields(docs_response):
    """Guard against the example being trimmed to something that proves nothing."""
    assert docs_response["vote"]["neighbours"], "example must list neighbours"
    assert len(docs_response["vote"]["neighbours"]) <= C.MAX_NEIGHBOURS
    assert all(c["confidence"]["status"] == "vote_share_uncalibrated" for c in docs_response["candidates"])
    p = docs_response["provenance"]
    assert p["patent_encoder"] and p["patent_corpus"] and p["vote_settings"]


# ------------------------------------------------------------------ 2. schema value sets and constants


def test_contract_version_and_caps():
    assert C.CLASSIFY_CONTRACT_VERSION == "2.0.0"
    assert C.MAX_NEIGHBOURS == 10
    assert C.MAX_TOP_CODES_PER_LEVEL == 10


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        (C.Classifier, CLASSIFIERS),
        (C.FallbackReason, FALLBACK_REASONS),
        (C.ConfidenceStatus, CONFIDENCE_STATUSES),
        (C.VoteCodeLevel, VOTE_CODE_LEVELS),
    ],
    ids=["classifier", "fallback_reason", "confidence_status", "vote_code_level"],
)
def test_schema_value_sets_match_contract(literal, expected):
    assert set(typing.get_args(literal)) == expected


@pytest.mark.parametrize("model", [C.ClassifyResponse, C.Candidate, C.Alternative, C.Provenance, C.VoteResult])
def test_openapi_serialization_schema_marks_every_field_required(model):
    """json_schema_serialization_defaults_required: fields with defaults are still always on the wire."""
    schema = model.model_json_schema(mode="serialization")
    assert set(schema["required"]) == set(model.model_fields)


# --------------------------------------------------------------------- 3. every classifier value / status


def test_classifier_neighbour_vote_valid_and_invalid():
    ok = accepted(ex())
    assert ok.classifier == "neighbour_vote"
    bad = ex()
    bad["candidates"][0]["vote_share"]["exact"] = 1.01  # vote share above 1
    bad["candidates"][0]["confidence"]["score"] = 1.01
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        ClassifyResponse.model_validate(bad)


@pytest.mark.parametrize("reason", sorted(FALLBACK_REASONS - {"vote_below_minimum"}))
def test_classifier_definition_fallback_valid_without_vote(reason):
    resp = accepted(fallback(reason))
    assert resp.classifier == "definition_fallback"
    assert resp.fallback_reason == reason
    assert resp.vote is None


def test_fallback_vote_below_minimum_keeps_the_vote():
    resp = accepted(fallback("vote_below_minimum"))
    assert resp.classifier == "definition_fallback"
    assert resp.vote is not None
    assert resp.vote.first_code_share.subclass < resp.provenance.vote_settings.min_vote_share


@pytest.mark.parametrize("reason", sorted(FALLBACK_REASONS - {"vote_below_minimum"}))
def test_fallback_reasons_without_a_vote_reject_a_vote(reason):
    d = fallback(reason)
    d["vote"] = ex()["vote"]
    rejected(d, f"vote must be null when fallback_reason is {reason!r}")


def test_fallback_vote_below_minimum_requires_the_vote():
    d = fallback("vote_below_minimum")
    d["vote"] = None
    rejected(d, "vote is required when the neighbour vote ran")


def test_calibrated_status_valid_and_null_score_invalid():
    d = ex()
    for c in d["candidates"] + d["alternatives"]:
        c["confidence"] = {
            "score": 0.81,
            "status": "calibrated",
            "calibrator_id": "vote-cal-1",
            "interpretation": None,
            "detail": None,
        }
    resp = accepted(d)
    assert resp.candidates[0].confidence.status == "calibrated"
    d["candidates"][0]["confidence"]["score"] = None
    rejected(d, "confidence.score is required when status is 'calibrated'")


def test_vote_share_uncalibrated_status_valid_and_invalid_forms():
    resp = accepted(ex())
    assert {c.confidence.status for c in resp.candidates} == {"vote_share_uncalibrated"}
    d = ex()
    d["candidates"][0]["confidence"]["score"] = None
    rejected(d, "confidence.score is required when status is 'vote_share_uncalibrated'")
    d = ex()
    d["candidates"][0]["confidence"]["calibrator_id"] = "vote-cal-1"
    rejected(d, "calibrator_id must be null for an uncalibrated vote share")


@pytest.mark.parametrize("status", ["not_calibrated", "incompatible_artifact", "unsupported_input_mode"])
def test_null_score_statuses_valid_in_fallback_and_invalid_with_a_score(status):
    d = fallback("no_serving_corpus")
    for c in d["candidates"] + d["alternatives"]:
        c["confidence"]["status"] = status
    assert accepted(d).candidates[0].confidence.status == status
    d["candidates"][0]["confidence"]["score"] = 0.5
    rejected(d, f"confidence.score must be null when status is {status!r}")


@pytest.mark.parametrize("status", ["not_calibrated", "incompatible_artifact", "unsupported_input_mode"])
def test_null_score_statuses_rejected_with_the_neighbour_vote(status):
    """With the vote the status is always calibrated or vote_share_uncalibrated and the score is never null."""
    d = ex()
    d["candidates"][0]["confidence"] = {
        "score": None,
        "status": status,
        "calibrator_id": None,
        "interpretation": None,
        "detail": None,
    }
    rejected(d, "confidence.status must be 'calibrated' or 'vote_share_uncalibrated'")


def test_vote_share_uncalibrated_rejected_in_fallback():
    d = fallback("no_serving_corpus")
    d["candidates"][0]["confidence"] = _uncal(0.5)
    rejected(d, "'vote_share_uncalibrated' is only valid with the neighbour vote")


def test_unknown_confidence_status_rejected():
    d = ex()
    d["candidates"][0]["confidence"]["status"] = "probably"
    with pytest.raises(ValidationError, match="calibrated"):
        ClassifyResponse.model_validate(d)


# ---------------------------------------------- 4. negative validator cases ported from TASK-0001 Evidence


def test_rejects_vote_share_above_one():
    d = ex()
    d["candidates"][0]["vote_share"]["subclass"] = 1.2
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        ClassifyResponse.model_validate(d)


def test_rejects_negative_vote_share():
    d = ex()
    d["candidates"][0]["vote_share"]["exact"] = -0.1
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ClassifyResponse.model_validate(d)


def test_rejects_exact_greater_than_maingroup():
    d = ex()
    d["candidates"][0]["vote_share"] = {"subclass": 0.9551, "maingroup": 0.5, "exact": 0.7907}
    rejected(d, "vote_share.exact cannot exceed vote_share.maingroup")


def test_rejects_maingroup_greater_than_subclass():
    d = ex()
    d["candidates"][0]["vote_share"] = {"subclass": 0.5, "maingroup": 0.8865, "exact": 0.4}
    rejected(d, "vote_share.maingroup cannot exceed vote_share.subclass")


def test_rejects_exact_without_maingroup_and_maingroup_without_subclass():
    d = ex()
    d["candidates"][0]["vote_share"] = {"subclass": 0.9, "maingroup": None, "exact": 0.5}
    rejected(d, "vote_share.exact requires vote_share.maingroup")
    d["candidates"][0]["vote_share"] = {"subclass": None, "maingroup": 0.5, "exact": None}
    rejected(d, "vote_share.maingroup requires vote_share.subclass")


def test_rejects_missing_fallback_reason_for_definition_fallback():
    d = fallback("no_serving_corpus")
    d["fallback_reason"] = None
    rejected(d, "fallback_reason is required when classifier is 'definition_fallback'")


def test_rejects_fallback_reason_with_neighbour_vote():
    d = ex()
    d["fallback_reason"] = "vote_failed"
    rejected(d, "fallback_reason must be null when classifier is 'neighbour_vote'")


def test_rejects_vote_with_fallback_no_serving_corpus():
    d = ex()
    d["classifier"] = "definition_fallback"
    d["fallback_reason"] = "no_serving_corpus"
    rejected(d, "vote must be null when fallback_reason is 'no_serving_corpus'")


def test_rejects_neighbour_vote_without_vote():
    d = ex()
    d["vote"] = None
    rejected(d, "vote is required when the neighbour vote ran")


def test_rejects_score_not_equal_to_own_level_share():
    d = ex()
    d["candidates"][0]["confidence"]["score"] = 0.5  # own-level share (exact) is 0.7907
    rejected(d, "an uncalibrated confidence.score must equal its vote share at its own level")


def test_rejects_uncalibrated_with_null_score():
    d = ex()
    d["candidates"][1]["confidence"]["score"] = None
    rejected(d, "confidence.score is required when status is 'vote_share_uncalibrated'")


def test_rejects_not_calibrated_with_a_score():
    d = fallback("neighbour_vote_disabled")
    d["candidates"][0]["confidence"]["score"] = 0.4
    rejected(d, "confidence.score must be null when status is 'not_calibrated'")


def test_rejects_eleven_neighbours():
    d = ex()
    d["vote"]["neighbours"] = [copy.deepcopy(d["vote"]["neighbours"][0]) for _ in range(11)]
    with pytest.raises(ValidationError, match="at most 10 items"):
        ClassifyResponse.model_validate(d)


def test_accepts_exactly_ten_neighbours():
    d = ex()
    d["vote"]["neighbours"] = [copy.deepcopy(d["vote"]["neighbours"][0]) for _ in range(10)]
    accepted(d)


def test_rejects_more_neighbours_than_neighbour_count():
    d = ex()
    d["vote"]["neighbour_count"] = 3
    rejected(d, "vote.neighbours cannot exceed vote.neighbour_count")


def test_rejects_eleven_top_codes_in_a_level():
    d = ex()
    d["vote"]["top_codes"]["subclass"] = [{"display_code": f"H{i:02d}L", "share": 0.01} for i in range(11)]
    with pytest.raises(ValidationError, match="at most 10 items"):
        ClassifyResponse.model_validate(d)


@pytest.mark.parametrize("missing", ["patent_corpus", "patent_encoder", "vote_settings"])
def test_rejects_vote_without_provenance_patent_info(missing):
    d = ex()
    d["provenance"][missing] = None
    rejected(d, "provenance.patent_encoder, patent_corpus and vote_settings are required with a vote")


def test_rejects_fallback_with_vote_shares():
    d = fallback("no_serving_corpus")
    d["candidates"][0]["vote_share"] = {"subclass": 0.9, "maingroup": 0.8, "exact": 0.7}
    rejected(d, "vote_share must be all-null in the fallback")


def test_rejects_fallback_alternative_with_vote_shares():
    d = fallback("no_serving_corpus")
    d["alternatives"][0]["vote_share"] = {"subclass": 0.9, "maingroup": None, "exact": None}
    rejected(d, "no vote shares in the fallback")


def test_rejects_unknown_classifier():
    d = ex()
    d["classifier"] = "magic"
    with pytest.raises(ValidationError, match="neighbour_vote"):
        ClassifyResponse.model_validate(d)


def test_rejects_unknown_fallback_reason():
    d = fallback("no_serving_corpus")
    d["fallback_reason"] = "because"
    with pytest.raises(ValidationError, match="no_serving_corpus"):
        ClassifyResponse.model_validate(d)


def test_rejects_neighbour_count_greater_than_k():
    d = ex()
    d["provenance"]["vote_settings"]["k"] = 50  # neighbour_count is 100
    rejected(d, "vote.neighbour_count cannot exceed provenance.vote_settings.k")


def test_rejects_neighbour_count_zero():
    d = ex()
    d["vote"]["neighbour_count"] = 0
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ClassifyResponse.model_validate(d)


def test_rejects_neighbour_vote_candidate_without_subclass_share():
    d = ex()
    d["candidates"][1]["vote_share"] = {"subclass": None, "maingroup": None, "exact": None}
    rejected(d, "vote_share.subclass is required with the vote")


def test_rejects_subgroup_first_code_with_exact_null():
    d = ex()
    d["vote"]["first_code_share"]["exact"] = None
    rejected(d, "vote.first_code_share needs a share at first_code_level (subgroup)")


def test_rejects_maingroup_first_code_with_maingroup_null():
    d = ex()
    d["vote"]["first_code"] = "H04L 9/00"
    d["vote"]["first_code_level"] = "maingroup"
    d["vote"]["first_code_share"] = {"subclass": 0.9551, "maingroup": None, "exact": None}
    rejected(d, "vote.first_code_share needs a share at first_code_level (maingroup)")


def test_rejects_first_code_share_without_subclass():
    d = ex()
    d["vote"]["first_code_share"] = {"subclass": None, "maingroup": None, "exact": None}
    rejected(d, "vote.first_code_share.subclass is required")


def test_rejects_neighbour_vote_with_empty_candidates():
    d = ex()
    d["candidates"] = []
    rejected(d, "candidates must not be empty when classifier is 'neighbour_vote'")


def test_rejects_first_code_neither_lead_nor_demoted():
    d = ex()
    # H04L 9/08 is nowhere: lead is the 9/32 sibling, alternatives hold only H02J 13/00.
    d["candidates"] = d["candidates"][1:]
    rejected(d, "candidates[0] (H04L 9/32, subgroup) must be vote.first_code H04L 9/08 or its ancestor")


# ------------------------------------------- 5. first-code rule with target_level (amendment 2, positives)


def test_accepts_first_code_demoted_to_alternatives_with_next_candidate_leading():
    d = ex()
    first = d["candidates"].pop(0)
    d["alternatives"].append(_as_alternative(first, "Precedence rule: H04L 9/32 takes precedence."))
    resp = accepted(d)
    assert resp.candidates[0].display_code == "H04L 9/32"
    assert resp.vote.first_code == "H04L 9/08"
    assert any(a.display_code == "H04L 9/08" for a in resp.alternatives)


def test_accepts_subclass_ancestor_as_lead():
    resp = accepted(with_lead("subclass", "H04L", {"subclass": 0.9551, "maingroup": None, "exact": None}, 0.9551))
    assert resp.candidates[0].display_code == "H04L"


def test_accepts_maingroup_ancestor_as_lead():
    resp = accepted(
        with_lead(
            "maingroup",
            "H04L 9/00",
            {"subclass": 0.9551, "maingroup": 0.8865, "exact": 0.0},
            0.8865,
            code="H04L0009000000",
        )
    )
    assert resp.candidates[0].level == "maingroup"


def test_accepts_class_ancestor_with_all_null_vote_share_and_class_score():
    resp = accepted(with_lead("class", "H04", {"subclass": None, "maingroup": None, "exact": None}, 0.9603))
    assert resp.candidates[0].level == "class"
    assert resp.candidates[0].vote_share.is_empty()
    assert resp.candidates[0].confidence.score == 0.9603


def test_accepts_demoted_ancestor_in_alternatives_with_other_code_leading():
    d = with_lead("subclass", "H02J", {"subclass": 0.1809, "maingroup": None, "exact": None}, 0.1809)
    ancestor = with_lead("subclass", "H04L", {"subclass": 0.9551, "maingroup": None, "exact": None}, 0.9551)
    d["alternatives"].append(_as_alternative(ancestor["candidates"][0], "Precedence rule: H02J takes precedence."))
    resp = accepted(d)
    assert resp.candidates[0].display_code == "H02J"
    assert {a.display_code for a in resp.alternatives} >= {"H04L"}


# -------------------------------------------- 6. first-code rule with target_level (amendment 2, negatives)


def test_rejects_non_ancestor_subclass_lead_H04W():
    d = with_lead("subclass", "H04W", {"subclass": 0.0459, "maingroup": None, "exact": None}, 0.0459)
    rejected(d, "candidates[0] (H04W, subclass) must be vote.first_code H04L 9/08 or its ancestor")


def test_rejects_ancestor_display_with_wrong_level():
    """Right code, wrong level: H04L labelled maingroup."""
    d = with_lead("maingroup", "H04L", {"subclass": 0.9551, "maingroup": 0.8865, "exact": 0.0}, 0.8865)
    rejected(d, "candidates[0] (H04L, maingroup) must be vote.first_code H04L 9/08 or its ancestor")


def test_rejects_sibling_maingroup_H04L_67_00():
    d = with_lead("maingroup", "H04L 67/00", {"subclass": 0.9551, "maingroup": 0.1068, "exact": 0.1068}, 0.1068)
    rejected(d, "candidates[0] (H04L 67/00, maingroup) must be vote.first_code H04L 9/08 or its ancestor")


def test_rejects_non_ancestor_class_H02():
    d = with_lead("class", "H02", {"subclass": None, "maingroup": None, "exact": None}, 0.1809)
    rejected(d, "candidates[0] (H02, class) must be vote.first_code H04L 9/08 or its ancestor")


def test_rejects_class_candidate_with_a_vote_share():
    d = with_lead("class", "H04", {"subclass": 0.9551, "maingroup": None, "exact": None}, 0.9603)
    rejected(d, "candidate H04: a class candidate has an all-null vote_share")


def test_rejects_first_code_shape_not_matching_first_code_level():
    d = ex()
    d["vote"]["first_code_level"] = "maingroup"  # first_code is still 'H04L 9/08'
    rejected(d, "vote.first_code 'H04L 9/08' is not a maingroup display code")


@pytest.mark.parametrize(
    ("first_code", "level"),
    [("H04L 9/00", "subgroup"), ("H04L 9/08", "subclass"), ("H04L", "maingroup"), ("H04", "subclass")],
)
def test_rejects_first_code_whose_shape_contradicts_its_level(first_code, level):
    d = ex()
    d["vote"]["first_code"] = first_code
    d["vote"]["first_code_level"] = level
    d["vote"]["first_code_share"] = {"subclass": 0.9551, "maingroup": 0.8865, "exact": 0.7907}
    rejected(d, f"is not a {level} display code")


def test_rejects_lead_h02j_without_demoted_first_code_or_ancestor():
    d = with_lead("subclass", "H02J", {"subclass": 0.1809, "maingroup": None, "exact": None}, 0.1809)
    rejected(d, "candidates[0] (H02J, subclass) must be vote.first_code H04L 9/08 or its ancestor")


# -------------------------------------------------- 7. v1-shaped payload, error bodies, request validation


def test_v1_shaped_payload_validates_with_defaults():
    """A payload without any v2 field (the pre-TASK-0010 code path) is a neighbour_vote_disabled fallback."""
    d = fallback("neighbour_vote_disabled")
    for k in ("classifier", "fallback_reason", "vote"):
        d.pop(k)
    for k in ("patent_encoder", "patent_corpus", "vote_settings"):
        d["provenance"].pop(k)
    for c in d["candidates"] + d["alternatives"]:
        c.pop("vote_share")
    resp = accepted(d)
    assert resp.classifier == "definition_fallback"
    assert resp.fallback_reason == "neighbour_vote_disabled"
    assert resp.vote is None


def test_503_patent_corpus_unavailable_error_body():
    body = {
        "error": "patent_corpus_unavailable",
        "message": "No serving patent corpus, and the definition fallback is unavailable.",
        "details": {"reason": "no_serving_corpus", "fallback_error": "embeddings_unavailable"},
    }
    err = ErrorBody.model_validate(body)
    PatentCorpusUnavailableDetails.model_validate(err.details)


@pytest.mark.parametrize("reason", ["no_serving_corpus", "patent_encoder_unavailable", "vote_empty", "vote_failed"])
def test_503_details_reasons_valid(reason):
    PatentCorpusUnavailableDetails.model_validate({"reason": reason, "fallback_error": "database_unavailable"})


def test_503_details_reject_unknown_reason_and_missing_fallback_error():
    with pytest.raises(ValidationError):
        PatentCorpusUnavailableDetails.model_validate({"reason": "vote_below_minimum", "fallback_error": "x"})
    with pytest.raises(ValidationError):
        PatentCorpusUnavailableDetails.model_validate({"reason": "vote_failed"})


def test_vote_sub_models_reject_out_of_range_values():
    n = ex()["vote"]["neighbours"][0]
    for field, bad in (("similarity", 1.5), ("vote_weight", 1.5), ("vote_weight", -0.1)):
        with pytest.raises(ValidationError):
            C.NeighbourPatent.model_validate({**n, field: bad})
    with pytest.raises(ValidationError):
        C.CodeShare.model_validate({"display_code": "H04L", "share": 1.0001})
    s = ex()["provenance"]["vote_settings"]
    for field, bad in (("k", 0), ("temperature", 0.0), ("beta", 1.5), ("min_vote_share", -0.1), ("hnsw_ef_search", 0)):
        with pytest.raises(ValidationError):
            C.VoteSettings.model_validate({**s, field: bad})
    c = ex()["provenance"]["patent_corpus"]
    with pytest.raises(ValidationError):
        C.PatentCorpusInfo.model_validate({**c, "document_count": -1})


# ------------------------------------------------- 8. apps/web/lib/types.ts (parsed as text) vs Pydantic

TS_TO_MODEL: dict[str, type[BaseModel]] = {
    "ClassifyRequest": C.ClassifyRequest,
    "SupportingPassage": C.SupportingPassage,
    "Confidence": C.Confidence,
    "VoteShare": C.VoteShare,
    "PathEntry": C.PathEntry,
    "ReferenceInfo": C.ReferenceInfo,
    "TechnicalScores": C.TechnicalScores,
    "Candidate": C.Candidate,
    "Alternative": C.Alternative,
    "AdditionalField": C.AdditionalField,
    "FieldOfInvention": C.FieldOfInvention,
    "SummaryLine": C.SummaryLine,
    "Summary": C.Summary,
    "InputCoverage": C.InputCoverage,
    "CodeShare": C.CodeShare,
    "VoteTopCodes": C.VoteTopCodes,
    "NeighbourPatent": C.NeighbourPatent,
    "VoteResult": C.VoteResult,
    "PatentEncoderInfo": C.PatentEncoderInfo,
    "PatentCorpusInfo": C.PatentCorpusInfo,
    "VoteSettings": C.VoteSettings,
    "Provenance": C.Provenance,
    "RankedDebug": C.RankedDebug,
    "ClassifyResponse": C.ClassifyResponse,
    "ApiErrorBody": C.ErrorBody,  # types.ts name for the backend ErrorBody
    "PatentCorpusUnavailableDetails": C.PatentCorpusUnavailableDetails,
}


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _balanced(src: str, open_at: int) -> str:
    depth = 0
    for i in range(open_at, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_at + 1 : i]
    raise AssertionError("unbalanced braces in types.ts")


def _split_top_level(text: str, seps: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "{<[(":
            depth += 1
        elif ch in "}>])":
            depth -= 1
        if ch in seps and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_ts_interfaces(path: Path) -> dict[str, dict[str, tuple[bool, str]]]:
    """interface name -> {field: (optional, type text)}."""
    src = _strip_comments(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, tuple[bool, str]]] = {}
    for m in re.finditer(r"export interface (\w+)\s*\{", src):
        body = _balanced(src, m.end() - 1)
        fields: dict[str, tuple[bool, str]] = {}
        # split on ';' or newline only at depth 0 (a type such as `{ term: string; ... }[]` nests)
        for decl in _split_top_level(body.replace("\n", ";"), ";"):
            fm = re.match(r"(\w+)(\?)?\s*:\s*(.+)$", decl, re.S)
            assert fm, f"cannot parse field declaration {decl!r} in interface {m.group(1)}"
            fields[fm.group(1)] = (bool(fm.group(2)), fm.group(3).strip())
        out[m.group(1)] = fields
    return out


def parse_ts_union(path: Path, name: str) -> set[str]:
    src = _strip_comments(path.read_text(encoding="utf-8"))
    m = re.search(rf"export type {name}\s*=\s*(.*?);", src, re.S)
    assert m, f"types.ts has no `export type {name}`"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _pydantic_nullable(annotation: Any) -> bool:
    return typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in typing.get_args(
        annotation
    )


@pytest.fixture(scope="module")
def ts_interfaces() -> dict[str, dict[str, tuple[bool, str]]]:
    return parse_ts_interfaces(TYPES_TS)


def ts_field_drift(name: str, ts_fields: set[str], model: type[BaseModel]) -> list[str]:
    fields = set(model.model_fields)
    out = [f"{name}: only in types.ts: {f}" for f in sorted(ts_fields - fields)]
    return out + [f"{name}: only in Pydantic: {f}" for f in sorted(fields - ts_fields)]


@pytest.mark.parametrize("name", sorted(TS_TO_MODEL))
def test_types_ts_interface_fields_match_pydantic(ts_interfaces, name):
    assert name in ts_interfaces, f"apps/web/lib/types.ts has no interface {name}"
    assert ts_field_drift(name, set(ts_interfaces[name]), TS_TO_MODEL[name]) == []


@pytest.mark.parametrize("name", sorted(set(TS_TO_MODEL) - {"ClassifyRequest", "ApiErrorBody"}))
def test_types_ts_response_fields_are_never_optional_and_nullability_matches(ts_interfaces, name):
    model = TS_TO_MODEL[name]
    for field, (optional, type_text) in ts_interfaces[name].items():
        assert not optional, f"{name}.{field}: response fields are never optional (`?`) in types.ts"
        ts_nullable = "null" in _split_top_level(type_text, "|")
        assert ts_nullable == _pydantic_nullable(model.model_fields[field].annotation), (
            f"{name}.{field}: nullability differs (types.ts `{type_text}`, "
            f"Pydantic {model.model_fields[field].annotation})"
        )


def test_types_ts_request_optionality_matches_pydantic_defaults(ts_interfaces):
    for field, (optional, _) in ts_interfaces["ClassifyRequest"].items():
        assert optional == (not C.ClassifyRequest.model_fields[field].is_required()), field


@pytest.mark.parametrize(
    ("ts_name", "literal"),
    [
        ("Status", C.Status),
        ("ConfidenceStatus", C.ConfidenceStatus),
        ("Coverage", C.Coverage),
        ("TargetLevel", C.TargetLevel),
        ("SourceField", C.SourceField),
        ("Classifier", C.Classifier),
        ("FallbackReason", C.FallbackReason),
        ("VoteCodeLevel", C.VoteCodeLevel),
    ],
)
def test_types_ts_literal_unions_match_pydantic(ts_name, literal):
    assert parse_ts_union(TYPES_TS, ts_name) == set(typing.get_args(literal))


def test_types_ts_constants_match_backend():
    src = TYPES_TS.read_text(encoding="utf-8")
    assert re.search(r'CLASSIFY_CONTRACT_VERSION\s*=\s*"([^"]+)"', src).group(1) == C.CLASSIFY_CONTRACT_VERSION
    assert int(re.search(r"MAX_NEIGHBOURS\s*=\s*(\d+)", src).group(1)) == C.MAX_NEIGHBOURS
    assert int(re.search(r"MAX_TOP_CODES_PER_LEVEL\s*=\s*(\d+)", src).group(1)) == C.MAX_TOP_CODES_PER_LEVEL


# ------------------------------------------- 9. apps/web/tests/fixtures.ts (evaluated) vs Pydantic

_DUMP_JS = r"""
// Evaluate a TypeScript module with the web repo's own `typescript` and print every exported object /
// zero-argument function result as JSON. Relative imports are transpiled the same way.
const fs = require("fs"), path = require("path"), vm = require("vm");
const [webDir, entry, outFile] = process.argv.slice(2);
const ts = require(path.join(webDir, "node_modules", "typescript"));
const cache = new Map();
function load(file) {
  if (cache.has(file)) return cache.get(file).exports;
  const src = fs.readFileSync(file, "utf8");
  const js = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
  const mod = { exports: {} };
  cache.set(file, mod);
  const req = (spec) => {
    if (!spec.startsWith(".")) return require(spec);
    const base = path.resolve(path.dirname(file), spec);
    for (const ext of [".ts", ".tsx", "/index.ts"]) if (fs.existsSync(base + ext)) return load(base + ext);
    throw new Error("cannot resolve " + spec + " from " + file);
  };
  vm.runInNewContext(js, { module: mod, exports: mod.exports, require: req, console }, { filename: file });
  return mod.exports;
}
const exp = load(path.resolve(entry));
const out = {};
for (const [name, value] of Object.entries(exp)) {
  if (typeof value === "function" && value.length === 0) out[name] = JSON.parse(JSON.stringify(value()));
  else if (value && typeof value === "object") out[name] = JSON.parse(JSON.stringify(value));
}
fs.writeFileSync(outFile, JSON.stringify(out));
"""


@pytest.fixture(scope="session")
def web_fixtures(tmp_path_factory) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH: cannot evaluate apps/web/tests/fixtures.ts")
    if not (WEB / "node_modules" / "typescript" / "lib" / "typescript.js").exists():
        pytest.skip(
            f"{WEB / 'node_modules' / 'typescript'} is missing (run npm ci in apps/web): cannot evaluate fixtures"
        )
    tmp = tmp_path_factory.mktemp("fixtures_dump")
    script, out = tmp / "dump.cjs", tmp / "fixtures.json"
    script.write_text(_DUMP_JS, encoding="utf-8")
    proc = subprocess.run(
        [node, str(script), str(WEB), str(FIXTURES_TS), str(out)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"fixtures.ts evaluation failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(out.read_text(encoding="utf-8"))


EXPECTED_FIXTURES = {
    "fixtureResponse",
    "fixtureFallbackResponse",
    "fixtureVoteResponse",
    "fixtureWeakVoteResponse",
    "FIXTURE_PATENT_CORPUS_UNAVAILABLE",
}


def _response_fixtures(web_fixtures: dict[str, Any]) -> dict[str, dict]:
    return {k: v for k, v in web_fixtures.items() if isinstance(v, dict) and "classifier" in v}


def test_fixtures_ts_exports_the_expected_fixtures(web_fixtures):
    missing = EXPECTED_FIXTURES - web_fixtures.keys()
    assert not missing, f"fixtures.ts no longer exports {sorted(missing)}"


def test_fixtures_ts_response_fixtures_validate_and_cover_the_classifier_values(web_fixtures):
    resp = _response_fixtures(web_fixtures)
    assert set(resp) >= {"fixtureResponse", "fixtureVoteResponse", "fixtureWeakVoteResponse"}
    parsed = {name: ClassifyResponse.model_validate(body) for name, body in resp.items()}
    assert parsed["fixtureResponse"].classifier == "definition_fallback"
    assert parsed["fixtureResponse"].fallback_reason == "no_serving_corpus"
    assert parsed["fixtureVoteResponse"].classifier == "neighbour_vote"
    assert parsed["fixtureVoteResponse"].vote is not None
    assert parsed["fixtureWeakVoteResponse"].fallback_reason == "vote_below_minimum"
    assert parsed["fixtureWeakVoteResponse"].vote is not None


@pytest.mark.parametrize("name", ["fixtureResponse", "fixtureVoteResponse", "fixtureWeakVoteResponse"])
def test_fixtures_ts_keys_match_pydantic_at_every_level(web_fixtures, name):
    assert name in web_fixtures, f"fixtures.ts no longer exports {name}"
    assert key_drift(ClassifyResponse, web_fixtures[name]) == []


def test_fixtures_ts_vote_fixture_is_the_docs_example_except_the_known_differences(web_fixtures, docs_response):
    """fixtureVoteResponse() is documented as the docs/api.md worked example with the emoji abstract. The only
    intended differences: an extra "Illustrative contract fixture." warning, and the passage offsets (the
    leading emoji shifts them by 2 code points' worth)."""
    fx, doc = copy.deepcopy(web_fixtures["fixtureVoteResponse"]), copy.deepcopy(docs_response)
    assert fx["warnings"][: len(doc["warnings"])] == doc["warnings"]
    for d in (fx, doc):
        d.pop("warnings")
        for p in d["candidates"][0]["supporting_passages"]:
            p.pop("start"), p.pop("end")
    assert fx == doc


def test_fixtures_ts_503_body_matches_the_error_contract(web_fixtures):
    fx = web_fixtures["FIXTURE_PATENT_CORPUS_UNAVAILABLE"]
    assert set(fx) == {"status", "body"} and fx["status"] == 503
    body = ErrorBody.model_validate(fx["body"])
    assert body.error == "patent_corpus_unavailable"
    PatentCorpusUnavailableDetails.model_validate(body.details)
    assert set(fx["body"]) == set(ErrorBody.model_fields)


# ------------------------------------------------ 10. prove the drift detection detects (copies only)


def test_key_drift_is_clean_on_the_unmodified_example(docs_response):
    assert key_drift(ClassifyResponse, docs_response) == []


def test_renamed_nested_field_in_a_copy_is_caught_with_both_sides_reported(docs_response):
    """Rename vote.neighbours[0].vote_weight in a COPY (the real example and fixtures are untouched)."""
    mutated = rename_key(docs_response, ("vote", "neighbours", 0), "vote_weight", "voteWeight")
    problems = key_drift(ClassifyResponse, mutated)
    assert "$.vote.neighbours[0]: missing field 'vote_weight' (NeighbourPatent)" in problems
    assert "$.vote.neighbours[0]: unexpected field 'voteWeight' (NeighbourPatent)" in problems
    assert key_drift(ClassifyResponse, docs_response) == [], "the mutation must not leak into the original"


def test_extra_and_missing_field_in_a_copy_are_caught(docs_response):
    extra = copy.deepcopy(docs_response)
    extra["candidates"][0]["vote_share"]["class"] = 0.5
    assert any("unexpected field 'class' (VoteShare)" in p for p in key_drift(ClassifyResponse, extra))
    gone = copy.deepcopy(docs_response)
    del gone["provenance"]["vote_settings"]["hnsw_ef_search"]
    assert any("missing field 'hnsw_ef_search' (VoteSettings)" in p for p in key_drift(ClassifyResponse, gone))


def test_renaming_any_field_at_any_level_is_caught_in_the_docs_example(docs_response):
    levels, escaped = set(), []
    for path, holder in model_dicts(ClassifyResponse, docs_response):
        levels.add(len([p for p in path if isinstance(p, str)]))
        for key in holder:
            mutated = rename_key(docs_response, path, key, key + "_renamed")
            if not key_drift(ClassifyResponse, mutated):
                escaped.append((path, key))
    assert not escaped, f"renames not detected: {escaped}"
    assert max(levels) >= 3, "the example must reach at least three nesting levels"


@pytest.mark.parametrize("name", ["fixtureResponse", "fixtureVoteResponse", "fixtureWeakVoteResponse"])
def test_renaming_any_field_at_any_level_is_caught_in_each_web_fixture(web_fixtures, name):
    fixture = web_fixtures[name]
    count, escaped = 0, []
    for path, holder in model_dicts(ClassifyResponse, fixture):
        for key in holder:
            count += 1
            mutated = rename_key(fixture, path, key, key + "_renamed")
            if not key_drift(ClassifyResponse, mutated):
                escaped.append((path, key))
    assert count > 20
    assert not escaped, f"renames not detected: {escaped}"


def test_renamed_fixture_field_would_also_fail_validation_not_only_the_key_check(web_fixtures):
    """Belt and braces: the renamed copy is rejected by Pydantic itself (a required field disappears)."""
    mutated = rename_key(web_fixtures["fixtureVoteResponse"], ("vote",), "first_code", "firstCode")
    with pytest.raises(ValidationError, match="first_code"):
        ClassifyResponse.model_validate(mutated)


def test_types_ts_drift_detection_catches_a_renamed_field(ts_interfaces):
    """Same proof for the text parser: rename a field in a COPY of the parsed interface."""
    fields = set(ts_interfaces["VoteResult"])
    assert ts_field_drift("VoteResult", fields, C.VoteResult) == []
    fields.remove("first_code")
    fields.add("firstCode")
    assert ts_field_drift("VoteResult", fields, C.VoteResult) == [
        "VoteResult: only in types.ts: firstCode",
        "VoteResult: only in Pydantic: first_code",
    ]
