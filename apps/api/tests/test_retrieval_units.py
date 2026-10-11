import pytest

from app.services.retrieval.fusion import ViewHits, fuse, rrf
from app.services.retrieval.keyword import build_websearch_query, extract_terms


def test_rrf_formula_ranks_start_at_one_and_absent_contributes_zero():
    s = rrf([["A", "B"], ["B"]], k=60)
    assert s["A"] == pytest.approx(1 / 61)
    assert s["B"] == pytest.approx(1 / 62 + 1 / 61)


def test_abstract_only_fusion_equals_per_view_rrf_and_preserves_components():
    vh = ViewHits("abstract", "abstract", keyword=[("K1", 1, 0.5), ("X", 2, 0.4)], vector=[("X", 1, 0.8), ("V1", 2, 0.7)])
    out = fuse([vh], k=60, per_source_cap=100)
    by = {c.code: c for c in out}
    assert out[0].code == "X"
    assert by["X"].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert by["X"].keyword_rank == 2 and by["X"].vector_rank == 1
    assert by["X"].cosine_similarity == pytest.approx(0.8)
    assert by["K1"].vector_rank is None and by["K1"].cosine_similarity is None  # lexical-only retained
    assert "keyword:abstract" in by["K1"].origin


def test_many_spec_chunks_do_not_outvote_abstract():
    abstract = ViewHits("abstract", "abstract", [("A", 1, 1.0)], [("A", 1, 0.9)])
    spec = [ViewHits(f"specification:body#{i}", "specification", [("S", 1, 1.0)], [("S", 1, 0.9)]) for i in range(20)]
    out = fuse([abstract, *spec], k=60, per_source_cap=100)
    scores = {c.code: c.rrf_score for c in out}
    assert scores["A"] == pytest.approx(scores["S"])  # equal per-source contribution


def test_terms_are_or_alternatives_not_and(qkd_abstract):
    terms = extract_terms(qkd_abstract, 24)
    texts = [t.text for t in terms]
    assert "smart grid" in " | ".join(texts)
    assert any("quantum" in t for t in texts)
    q = build_websearch_query(terms)
    assert " or " in q and " and " not in q.lower().replace("band", "")
    assert len(terms) <= 24


def test_query_builder_drops_unsafe_terms():
    from app.services.retrieval.keyword import Term

    q = build_websearch_query([Term('x" or 1=1 --', 1, 0), Term("key distribution", 1, 1)])
    assert q == '"key distribution"'


def test_unigrams_are_capped_and_phrases_lead(qkd_abstract):
    terms = extract_terms(qkd_abstract, 24)
    unigrams = [t.text for t in terms if " " not in t.text]
    assert len(unigrams) <= 4
    assert " " in terms[0].text
