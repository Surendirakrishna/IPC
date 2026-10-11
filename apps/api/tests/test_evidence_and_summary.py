import pytest

from app.schemas.classify import Candidate, ClassifyRequest, Confidence, TechnicalScores
from app.services.classification.evidence import (
    EvidenceError,
    Passage,
    defining_feature_check,
    split_passages,
    verify,
)
from app.services.classification.pipeline import _is_enumeration
from app.services.classification.summary import build_summary


def test_passages_have_exact_code_point_offsets(qkd_abstract):
    ps = split_passages(qkd_abstract, source="abstract")
    assert len(ps) >= 6
    for p in ps:
        assert qkd_abstract[p.start : p.end] == p.quote
        verify(p, {"abstract": qkd_abstract})


def test_offsets_with_astral_characters():
    text = "🔐 Quantum keys are distributed between nodes. Second sentence about encryption keys."
    ps = split_passages(text, source="abstract")
    for p in ps:
        assert text[p.start : p.end] == p.quote


def test_verify_rejects_wrong_offsets_and_missing_sources():
    with pytest.raises(EvidenceError):
        verify(Passage("abstract", "quantum", 0, 7), {"abstract": "classic quantum"})
    with pytest.raises(EvidenceError):
        verify(Passage("claims", "x", 0, 1), {"abstract": "x"})


def test_defining_feature_check(qkd_abstract):
    present, missing = defining_feature_check("Key distribution", qkd_abstract)
    assert set(present) == {"key", "distribution"} and not missing
    present, missing = defining_feature_check("Public key, i.e. encryption algorithm being computationally infeasible", qkd_abstract)
    assert "computationally" in missing or "infeasible" in missing


def test_device_list_is_application_context(qkd_abstract):
    device_sentence = next(p.quote for p in split_passages(qkd_abstract, source="abstract") if "smart meters" in p.quote)
    assert _is_enumeration(device_sentence)
    assert not _is_enumeration("Quantum communication channels are utilized to exchange quantum states.")


def _cand(display, conf):
    return Candidate(code="X", display_code=display, title="t", level="subgroup", confidence=conf,
                     supporting_passages=[], rationale="r", unresolved_ambiguity=[], parent_path=[], references=[],
                     scores=TechnicalScores(), origin=[])


def test_summary_uncalibrated_never_invents_percentages():
    s = build_summary("COMMUNICATION", [_cand("H04L 9/08", Confidence(score=None, status="not_calibrated"))])
    assert s.markdown == (
        "| Field Of Invention | COMMUNICATION |\n| :--- | :--- |\n"
        "| Classification (IPC) | H04L 9/08 — confidence: not calibrated |\n"
    )
    assert "%" not in s.markdown


def test_summary_calibrated_and_multiple_codes():
    s = build_summary("COMMUNICATION", [
        _cand("H04L 9/08", Confidence(score=0.8123, status="calibrated", calibrator_id="c1")),
        _cand("H04L 9/32", Confidence(score=0.4, status="calibrated", calibrator_id="c1")),
    ])
    assert s.classification_cell_text == "H04L 9/08 — 81.2%; H04L 9/32 — 40.0%"
    assert [ln.display_code for ln in s.classification_lines] == ["H04L 9/08", "H04L 9/32"]


def test_summary_empty_is_insufficient_evidence_and_escapes():
    s = build_summary("UNDETERMINED", [])
    assert "| Classification (IPC) | Insufficient evidence |" in s.markdown
    s2 = build_summary("A|B\nC", [])
    assert "A\\|B C" in s2.markdown


def test_request_contract():
    with pytest.raises(ValueError):
        ClassifyRequest(abstract="   \n ")
    r = ClassifyRequest(abstract="x", specification="  ", claims="", title=None)
    assert r.specification is None and r.claims is None
    r2 = ClassifyRequest(abstract="x", claims="1. A thing.\n2. The thing of claim 1.")
    assert r2.claims.startswith("1. A thing.")
    assert r2.target_level == "group" and r2.max_results == 5 and r2.use_llm is False
    with pytest.raises(ValueError):
        ClassifyRequest(abstract="x", claims=["1. a"])  # claims is plain text in v1


def test_subclass_scope_check(qkd_abstract):
    from app.services.classification.evidence import subclass_scope_check

    present, missing = subclass_scope_check("WIRELESS COMMUNICATION NETWORKS", qkd_abstract)
    assert present == [] and missing == ["wireless"]
    present, _ = subclass_scope_check("TRANSMISSION OF DIGITAL INFORMATION, e.g. TELEGRAPHIC COMMUNICATION", qkd_abstract)
    assert "transmission" in present
    _, missing = subclass_scope_check("TRANSMISSION OF DIGITAL INFORMATION, e.g. TELEGRAPHIC COMMUNICATION", qkd_abstract)
    assert "telegraphic" not in missing  # e.g. examples are not scope requirements


def test_provenance_accepts_disabled_thresholds():
    # Policy thresholds may be None (disabled) and lexical enforcement is a bool.
    from app.schemas.classify import Provenance

    p = Provenance(taxonomy_version="t", source_snapshot={}, embedding_provider="bge_m3", embedding_model="m",
                   model_revision="r", embedding_run_id=None, reranker_model="rr", reranker_revision="r",
                   scope_formatter="s", calibrator_id=None, field_mapping_version="2",
                   retrieval_settings={"min_reranker_score": None, "enforce_lexical_checks": False, "rrf_k": 60},
                   encoding_policy={})
    assert p.retrieval_settings["min_reranker_score"] is None
