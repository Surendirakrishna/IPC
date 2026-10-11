from app.services.confidence.ipc_codes import parse_symbols
from app.services.evaluation import benchmark as bm
from app.services.evaluation.ipindia import parse_detail


def test_parse_symbols_formats():
    syms = parse_symbols("A61B 5/00, A61B 5/1455, G06N3/08, H04L0009080000 (2006.01), G16H")
    assert [s.display for s in syms] == ["H04L 9/08", "A61B 5/00", "A61B 5/1455", "G06N 3/08", "G16H"]
    assert syms[2].maingroup_display == "A61B 5/00"
    assert syms[-1].main_group is None and syms[-1].maingroup_display is None


def test_pred_key_levels():
    assert bm.pred_key("H04L 9/0838", "subclass") == "H04L"
    assert bm.pred_key("H04L 9/0838", "maingroup") == "H04L 9/00"
    assert bm.pred_key("H04L 9/0838", "exact") == "H04L 9/0838"


def test_score_document_levels_and_abstention():
    gt = bm.GroundTruth(symbols=parse_symbols("H04L 9/08, G06N 3/08"))
    row = bm.score_document(gt, ["H04L 9/14"], ["H04L 9/14", "H04L 9/08", "G06N 3/04"], "H04")
    assert row["subclass"]["top1"] and row["maingroup"]["top1"] and not row["exact"]["top1"]
    assert row["exact"]["hit@3"] and not row["exact"]["hit@1"]
    assert row["subclass"]["recall"] == 0.5 and row["field_class_match"] is True
    empty = bm.score_document(gt, [], ["G06N 3/08"], None)
    assert empty["abstained"] and not empty["subclass"]["top1"] and empty["subclass"]["hit@1"]


def test_aggregate_and_wilson():
    gt = bm.GroundTruth(symbols=parse_symbols("A01B 3/00"))
    rows = [bm.score_document(gt, ["A01B 3/00"], ["A01B 3/00"], None),
            bm.score_document(gt, [], [], None)]
    agg = bm.aggregate(rows)
    assert agg["documents"] == 2 and agg["abstained"]["k"] == 1
    assert agg["exact"]["top1_all_documents"]["rate"] == 0.5
    assert agg["exact"]["top1_answered_only"]["rate"] == 1.0
    lo, hi = bm.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8


DETAIL = """
<td style="width: 200px">Invention Title</td><td>
 Smart Key Distribution Device </td>
<td style="width: 200px">Application Number</td><td>202641000001   </td>
<td style="width: 200px">Field Of Invention</td><td>COMMUNICATION</td>
<td style="width: 200px">Classification (IPC)</td><td>H04L 9/08, H04L 9/32 </td>
<td colspan="2"><strong>Abstract:</strong><br><br>
 A device distributes keys &amp; rotates them among nodes in a network securely.</td>
<textarea id="COMPLETE_SPECIFICATION" readonly>Description:FIELD
Keys.
 , C , Claims:1. A device comprising a key store.</textarea>
"""


def test_parse_detail():
    d = parse_detail(DETAIL)
    assert d["application_number"] == "202641000001" and d["title"] == "Smart Key Distribution Device"
    assert d["field_of_invention"] == "COMMUNICATION" and d["ground_truth"] == ["H04L 9/08", "H04L 9/32"]
    assert d["abstract"].startswith("A device distributes keys & rotates")
    assert d["specification"] == "FIELD\nKeys." and d["claims"] == "1. A device comprising a key store."


def _c(code, disp, rr, ev, rt="subgroup", mg="H04L0009000000", ctx=False, sub_ok=True, terms_ok=True):
    return {"code": code, "display_code": disp, "reranker_score": rr, "best_evidence_score": ev, "record_type": rt,
            "maingroup_code": mg, "application_context_only": ctx, "subclass_scope_ok": sub_ok,
            "defining_terms_ok": terms_ok}


def test_simulate_policy_matches_rule_order():
    ranked = [
        _c("H04W0012043300", "H04W 12/0433", 1.1, -2.7, mg="H04W0012040000", sub_ok=False),  # scope fails
        _c("H04L0009080000", "H04L 9/08", -0.2, -1.9),
        _c("H04L0009140000", "H04L 9/14", -0.9, -1.6, terms_ok=False),  # defining term missing
        _c("H04L0009000000", "H04L 9/00", -1.2, -2.0, rt="maingroup", mg=None),  # parent of selected
        _c("H04L0009320000", "H04L 9/32", -1.5, -1.0),  # same main group as H04L 9/08
        _c("G06N0003080000", "G06N 3/08", -2.5, -1.0, mg="G06N0003000000"),
    ]
    live = bm.PolicyParams(min_reranker_score=-1.0, max_reranker_gap=1.5, min_evidence_score=-3.0)
    assert bm.simulate_policy(ranked, live) == ["H04L 9/08"]
    loose = bm.PolicyParams(min_reranker_score=None, max_reranker_gap=3.0, min_evidence_score=-3.0)
    assert bm.simulate_policy(ranked, loose) == ["H04L 9/08", "H04L 9/32", "G06N 3/08"]  # same main group kept
    assert bm.simulate_policy(ranked, loose, {"H04L0009320000"}) == ["H04L 9/08", "G06N 3/08"]
    assert bm.simulate_policy(ranked, bm.PolicyParams(None, 3.0, 0.0)) == []


def test_simulate_policy_main_group_is_not_its_own_subgroup():
    # WIPO records give a main group its own code as maingroup_code; it must not demote itself.
    mg = "A41D0019000000"
    ranked = [_c(mg, "A41D 19/00", 3.4, -0.6, rt="maingroup", mg=mg),
              _c("A41D0019040000", "A41D 19/04", 2.2, 0.0, mg=mg)]  # outside the 0.5 gap
    p = bm.PolicyParams(min_reranker_score=None, max_reranker_gap=0.5, min_evidence_score=-5.0, enforce_lexical=False)
    assert bm.simulate_policy(ranked, p) == ["A41D 19/00"]
    wide = bm.PolicyParams(min_reranker_score=None, max_reranker_gap=1.5, min_evidence_score=-5.0, enforce_lexical=False)
    assert bm.simulate_policy(ranked, wide) == ["A41D 19/04"]  # an eligible own subgroup still wins
