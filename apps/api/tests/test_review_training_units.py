import pytest

from app.services.confidence.evaluation import label_pairs
from app.services.evaluation import families as fam
from app.services.evaluation import review_queue as rq
from app.services.evaluation.ipindia import parse_detail
from app.services.training import hard_negatives as hn

ROW = '<td style="width: 200px">{}</td>\n<td>{}</td>'


def _doc(app_no, abstract, **kw):
    return {"application_number": app_no, "title": kw.pop("title", "T"), "abstract": abstract, **kw}


# --- families -----------------------------------------------------------------------------------

def test_parse_detail_reads_priority_claim():
    html = "\n".join(ROW.format(k, v) for k, v in [
        ("Application Number", "202638096015"), ("Priority Number", "EPO20181776.4   "),
        ("Priority Country", "EPO"), ("Priority Date", "23/06/2020 "), ("Classification (IPC)", "H03M 7/40")])
    d = parse_detail(html)
    assert (d["priority_number"], d["priority_country"], d["priority_date"]) == ("EPO20181776.4", "EPO", "23/06/2020")


def test_normalise_priority_strips_country_prefix_and_punctuation():
    assert fam.normalise_priority("EPO20181776.4", "EPO") == fam.normalise_priority("20181776.4", "EPO")
    assert fam.normalise_priority("Germany10 2023 110 262.1", "Germany") == "priority:GERMANY:1020231102621"
    assert fam.normalise_priority(" ", "EPO") is None


def test_pct_parent_needs_cross_reference_wording():
    spec = "This application is the national phase of PCT/CN2025/070668 filed in 2025."
    assert fam.pct_parents(spec) == {"pct:CN2025/070668"}
    assert fam.pct_parents("Prior art PCT/CN2025/070668 discloses a pump.") == set()


def test_group_families_by_priority_pct_and_near_duplicate():
    text = "a wireless charging pad with a coil array and a controller that selects the coil nearest the device"
    docs = [
        _doc("1", "alpha beta gamma delta epsilon", priority_number="EPO1.4", priority_country="EPO"),
        _doc("2", "zeta eta theta iota kappa", priority_number="1.4", priority_country="EPO"),
        _doc("3", text),
        _doc("4", text + " in use"),
        _doc("5", "completely different subject matter about soil tillage and ploughs"),
        _doc("6", "lambda mu nu xi omicron", specification="National stage of PCT/EP2024/83349."),
        _doc("7", "pi rho sigma tau upsilon", specification="claims priority to PCT/EP2024/083349"),
    ]
    groups = {tuple(f.members) for f in fam.group_families(docs)}
    assert groups == {("1", "2"), ("3", "4"), ("5",), ("6", "7")}


def test_assign_splits_keeps_families_together_and_honours_pins():
    docs = [_doc("1", "x", priority_number="P1", priority_country="US"),
            _doc("2", "y", priority_number="P1", priority_country="US")] + \
           [_doc(str(i), f"unique abstract number {i} " * 3) for i in range(3, 60)]
    families = fam.group_families(docs)
    a = fam.assign_splits(families, {"fit": 0.7, "dev": 0.15, "test": 0.15}, seed="s")
    assert a["1"] == a["2"] and set(a.values()) == {"fit", "dev", "test"}
    assert a == fam.assign_splits(families, {"fit": 0.7, "dev": 0.15, "test": 0.15}, seed="s")
    pinned = fam.assign_splits(families, {"fit": 1.0}, seed="s", pinned={"2": "test"})
    assert pinned["1"] == pinned["2"] == "test"
    leak = fam.split_conflicts(families, {**a, "1": "fit", "2": "test"})
    assert [c["members"] for c in leak] == [{"1": "fit", "2": "test"}]
    with pytest.raises(ValueError):
        fam.assign_splits(families, {"fit": 0.5}, seed="s")


# --- review queue -------------------------------------------------------------------------------

def _describe(codes):
    known = {"H04L0009080000": "Key distribution", "H04W0012040000": "Key management",
             "G06N0003080000": "Learning methods"}
    return {c: {"display_code": c, "title": known[c], "path": []} for c in codes if c in known}


def _item(**kw):
    item, key = rq.build_review_item(
        _doc("A1", "abstract text", claims="1. A method."), ["H04W0012040000", "G06N0003080000"],
        ["H04L0009080000", "H04L0099990000"], _describe, split="dev", family_id="fam-A1",
        taxonomy_version="20260101", **kw)
    return item, key


def test_review_item_is_blinded_and_key_records_sources():
    item, key = _item()
    codes = [c["code"] for c in item["candidates"]]
    assert sorted(codes) == ["G06N0003080000", "H04L0009080000", "H04W0012040000"]
    assert all(set(c) == {"code", "display_code", "title", "path"} for c in item["candidates"])
    assert codes == [c["code"] for c in _item()[0]["candidates"]]  # deterministic shuffle
    assert key["sources"]["H04L0009080000"] == {"model_rank": None, "declared": True}
    assert key["sources"]["H04W0012040000"] == {"model_rank": 1, "declared": False}
    assert key["declared_missing_from_catalogue"] == ["H04L0099990000"]
    assert item["claims"] == "1. A method." and item["review"]["judgements"] == {c: None for c in codes}


def _reviewed(judgements, **review):
    item, _ = _item()
    item["review"].update({"reviewer": "examiner-A", "reviewed_at": "2026-10-11", "judgements": judgements, **review})
    return item


def test_finalize_partial_review_to_expert_schema():
    item = _reviewed({"H04L0009080000": "accept", "H04W0012040000": "reject", "G06N0003080000": "unsure"})
    docs, skipped = rq.finalize_reviews([item, _item()[0]], source="unit-test")
    assert [s["reason"] for s in skipped] == ["not reviewed"]
    (d,) = docs
    assert d["judgement_completeness"] == "partial" and d["split"] == "dev"
    assert {(lb["code"], lb["judgement"]) for lb in d["labels"]} == {
        ("H04L0009080000", "accept"), ("H04W0012040000", "reject")}
    pairs = label_pairs(d, [("H04L0009080000", 1.0), ("G06N0003080000", 0.5), ("H04W0012040000", 0.1)])
    assert {(p.code, p.label) for p in pairs} == {("H04L0009080000", 1), ("H04W0012040000", 0)}


def test_finalize_complete_review_and_added_codes():
    item = _reviewed({"H04L0009080000": "accept"}, added_codes=["H04L0009140000"],
                     accepted_codes_are_exhaustive=True)
    (d,), _ = rq.finalize_reviews([item], source="unit-test")
    assert d["judgement_completeness"] == "complete"
    assert [lb["code"] for lb in d["labels"]] == ["H04L0009080000", "H04L0009140000"]


@pytest.mark.parametrize("judgements,review,message", [
    ({"H04L0009080000": "maybe"}, {}, "must be one of"),
    ({"X": "accept"}, {}, "not one of the item's candidates"),
    ({"H04L0009080000": "accept"}, {"reviewer": None}, "reviewer is required"),
    ({"H04L0009080000": "reject"}, {"accepted_codes_are_exhaustive": True}, "no code was accepted"),
])
def test_finalize_rejects_malformed_reviews(judgements, review, message):
    with pytest.raises(rq.ReviewDataError, match=message):
        rq.finalize_reviews([_reviewed(judgements, **review)], source="unit-test")


def test_finalize_checks_added_codes_exist():
    item = _reviewed({}, added_codes=["Z99Z0001000000"])
    with pytest.raises(rq.ReviewDataError, match="not in the catalogue"):
        rq.finalize_reviews([item], source="unit-test", code_exists=lambda c: False)


# --- hard negatives -----------------------------------------------------------------------------

class FakeCatalogue:
    # subclass H04L -> main groups 9/00, 12/00; 9/00 -> 9/08 -> 9/0838 ; 9/00 -> 9/14 ; class H04 -> H04L, H04W
    TREE = {"H04": ["H04L", "H04W"], "H04L": ["H04L 9/00", "H04L 12/00"], "H04L 9/00": ["H04L 9/08", "H04L 9/14"],
            "H04L 9/08": ["H04L 9/0838"], "H04W": ["H04W 12/00"]}

    def nodes(self, codes):
        parent = {c: p for p, cs in self.TREE.items() for c in cs}
        out = {}
        for c in codes:
            if c not in parent:
                continue
            chain, p = [], parent.get(c)
            while p:
                chain.insert(0, p)
                p = parent.get(p)
            rtype = "subclass" if len(c) == 4 else ("maingroup" if c.endswith("/00") else "subgroup")
            out[c] = hn.Node(c, rtype, parent[c], c[:4] if len(c) >= 4 and rtype != "subclass" else None,
                             c[:3], chain)
        return out

    def children(self, parent_codes):
        return {p: self.TREE[p] for p in parent_codes if p in self.TREE}

    def scope_texts(self, codes):
        return {c: f"scope {c}" for c in codes}


def _labelled(labels, completeness="partial"):
    return {"document_id": "D1", "patent_family_id": "F1", "split": "fit", "abstract": "abs",
            "labels": [{"code": c, "judgement": j} for c, j in labels], "judgement_completeness": completeness}


def test_mine_group_kinds_and_never_related_codes():
    g = hn.mine_group(_labelled([("H04L 9/08", "accept"), ("H04W 12/00", "reject")]), FakeCatalogue(),
                      ranked_codes=["H04L 12/00"])
    kinds = {n["code"]: n["kind"] for n in g["negatives"]}
    assert [p["code"] for p in g["positives"]] == ["H04L 9/08"]
    assert kinds["H04W 12/00"] == "explicit_reject" and kinds["H04L 9/14"] == "sibling"
    assert kinds["H04L 12/00"] == "same_subclass"  # ranked_confusable is off for a partial review
    for related in ("H04L 9/08", "H04L 9/00", "H04L 9/0838", "H04L", "H04"):
        assert related not in kinds
    assert all(n["assumed"] == (n["kind"] != "explicit_reject") for n in g["negatives"])


def test_mine_group_complete_review_uses_ranked_confusables_and_exclusions():
    g = hn.mine_group(_labelled([("H04L 9/08", "accept")], "complete"), FakeCatalogue(),
                      ranked_codes=["H04L 12/00", "H04L 9/0838"], exclude=["H04L 9/14"])
    kinds = {n["code"]: n["kind"] for n in g["negatives"]}
    assert kinds == {"H04L 12/00": "ranked_confusable", "H04W": "same_class"}
    assert not any(n["assumed"] for n in g["negatives"])
    assert hn.mine_group(_labelled([("H04W 12/00", "reject")]), FakeCatalogue()) is None


def test_flag_rows_orientation():
    g = hn.mine_group(_labelled([("H04L 9/08", "accept")]), FakeCatalogue())
    (row,) = hn.to_flag_reranker_rows(g, orientation="document_first")
    assert row["query"] == "abs" and row["pos"] == ["scope H04L 9/08"] and "scope H04L 9/14" in row["neg"]
    (row,) = hn.to_flag_reranker_rows(g, orientation="scope_first")
    assert row["query"] == "scope H04L 9/08" and row["pos"] == ["abs"]
    with pytest.raises(ValueError):
        hn.to_flag_reranker_rows(g, orientation="sideways")
