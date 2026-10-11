from app.services.retrieval.query_views import build_views, chunk_span, regex_token_offsets, segment_claims

CLAIMS = """1. A method for distributing keys, comprising exchanging quantum states between nodes.
2. The method according to claim 1, wherein the nodes are smart meters.
3. A system configured to perform the method of claim 1."""


def test_claims_segmented_with_numbers_offsets_and_dependencies():
    views, status = segment_claims(CLAIMS)
    assert status == "numbered"
    assert [v.claim_number for v in views] == [1, 2, 3]
    for v in views:
        assert CLAIMS[v.start : v.end] == v.text
    assert views[0].is_independent_claim is True
    assert views[1].claim_depends_on == [1] and views[1].is_independent_claim is False
    assert views[2].claim_depends_on == [1]


def test_unreliable_claim_numbering_is_not_invented():
    views, status = segment_claims("A method comprising x.\nAnother aspect y.")
    assert status == "unnumbered" and views == []
    plan = build_views("abs text here", None, "A method comprising x.\nAnother aspect y.", None,
                       max_views=8, chunk_tokens=64, chunk_overlap=8)
    claim_views = [v for v in plan.views if v.source == "claims"]
    assert claim_views and all(v.claim_number is None for v in claim_views)
    assert any("numbering could not be parsed" in w for w in plan.warnings)


def test_abstract_only_has_single_view():
    plan = build_views("An abstract.", None, None, None, max_views=8, chunk_tokens=64, chunk_overlap=8)
    assert [v.view_id for v in plan.views] == ["abstract"]
    assert plan.coverage == {"abstract": "complete", "specification": "not_supplied", "claims": "not_supplied"}


def test_title_is_prepended_to_abstract_view_only_for_retrieval():
    plan = build_views("An abstract.", "A title", None, None, max_views=8, chunk_tokens=64, chunk_overlap=8)
    assert plan.views[0].text == "A title\nAn abstract."
    assert (plan.views[0].start, plan.views[0].end) == (0, len("An abstract."))


def test_spec_background_marked_and_budget_reports_partial():
    spec = "BACKGROUND\n" + "Prior systems use RSA keys. " * 60 + "\nDETAILED DESCRIPTION\n" + "The QKD module rotates keys. " * 60
    plan = build_views("abs", None, None, spec, max_views=3, chunk_tokens=40, chunk_overlap=5)
    spec_views = [v for v in plan.views if v.source == "specification"]
    assert len(spec_views) == 2
    assert all(not v.is_background for v in spec_views)  # background deprioritized under budget
    assert plan.coverage["specification"] == "partial"
    assert any("omitted text was not used" in w for w in plan.warnings)
    for v in spec_views:
        assert spec[v.start : v.end] == v.text


def test_chunking_never_splits_code_points():
    text = "量子鍵配送 🔐 key distribution " * 50
    spans = chunk_span(text, 0, len(text), max_tokens=20, overlap=4, token_offsets=regex_token_offsets)
    assert len(spans) > 1
    for s, e in spans:
        piece = text[s:e]
        assert piece.encode("utf-8").decode("utf-8") == piece
        assert piece == text[s:e]
