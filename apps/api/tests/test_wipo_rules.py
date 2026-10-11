"""WIPO master-file parsing (hierarchy, note ownership/scope/edition, rule meaning) and rule application."""

import zipfile
from types import SimpleNamespace

import pytest

from app.services.classification.pipeline import select_retrieval_views
from app.services.classification.rules import RuleCandidate, apply_scheme_rules
from app.services.confidence.ipc_codes import parse_symbols
from app.services.evaluation import benchmark as bm
from app.services.ingestion.wipo import (
    display_symbol,
    parse_catchwords,
    parse_definitions,
    parse_scheme,
    semantic_text,
)
from app.services.retrieval.fusion import ViewHits, fuse, rrf
from app.services.retrieval.query_views import build_views

NS = 'xmlns="http://www.wipo.int/classifications/ipc/masterfiles"'


def _entry(kind, symbol, title, body="", entry_type="K", edition="19680901"):
    return (f'<ipcEntry kind="{kind}" symbol="{symbol}" entryType="{entry_type}" edition="{edition}"><textBody><title>'
            f"<titlePart><text>{title}</text></titlePart></title></textBody>{body}</ipcEntry>")


def _note(symbol, paragraphs, end=None, edition="19900101"):
    end_attr = f' endSymbol="{end}"' if end else ""
    paras = "".join(f"<noteParagraph><text>{p}</text></noteParagraph>" for p in paragraphs)
    return (f'<ipcEntry kind="n" symbol="{symbol}"{end_attr} entryType="K" edition="{edition}"><textBody>'
            f'<note type="number">{paras}</note></textBody></ipcEntry>')


def _scheme(lang, plough_title):
    tractor = ('<ipcEntry kind="1" symbol="A01B0003240000" entryType="K" edition="19680901"><textBody><title>'
               '<titlePart><text>Tractor-drawn ploughs</text><entryReference><sref ref="A01B0003040000"/> takes '
               "precedence</entryReference></titlePart></title></textBody></ipcEntry>")
    swing = _entry("3", "A01B0003080000", "Swing ploughs")
    no_alt = _entry("2", "A01B0003060000", "without alternating possibility", swing)
    maingroup = _entry("m", "A01B0003000000", plough_title, "".join([
        _note("A01B0003020000", ['Group <sref ref="A01B0003240000"/> takes precedence over groups '
                                 '<mref ref="A01B0003020000" endRef="A01B0003040000"/>.'],
              end="A01B0003040000", edition="20000101"),
        _entry("1", "A01B0003020000", "Man-driven ploughs"),
        _entry("1", "A01B0003040000", "Animal-drawn ploughs", no_alt),
        tractor,
        _entry("1", "A01B0003300000", "Ploughs mounted on tractors"),
        _entry("1", "A01B0003990000", "Indexing scheme entry", entry_type="I"),
    ]))
    subclass_note = _note("A01B", [
        'In this subclass, group <sref ref="A01B0003300000"/>, which relates to tractors, takes precedence over '
        "other groups.",
        "In this subclass, in the absence of an indication to the contrary, classification is made in the last "
        "appropriate place.",
        'For apparatus used in drying, class <sref ref="F26"/> takes precedence over this subclass.',
        "In this main group, the last place priority rule is not applied.",
        "Any novel part must also be classified in the last appropriate place in this subclass.",
    ])
    subclass = ('<ipcEntry kind="u" symbol="A01B" entryType="K"><textBody><title><titlePart><text>SOIL WORKING'
                '</text><entryReference>making furrows <sref ref="A01C0005000000"/></entryReference></titlePart>'
                f"</title></textBody>{subclass_note}{maingroup}</ipcEntry>")
    tree = _entry("s", "A", "HUMAN NECESSITIES", _entry("c", "A01", "AGRICULTURE", subclass))
    return f'<?xml version="1.0" encoding="UTF-8"?><IPCScheme {NS} edition="20260101" lang="{lang}">{tree}</IPCScheme>'


DEFINITIONS = (
    f'<?xml version="1.0" encoding="UTF-8"?><IPCDefinitions {NS} xmlns:xhtml="http://www.w3.org/1999/xhtml" '
    'edition="20260101" lang="EN"><IPCDefinitionsSection symbol="A"><IPCDefinitionsClass symbol="A01">'
    '<IPC-DEFINITION IPC="A01B"><SPECIALRULES><xhtml:div>'
    '<xhtml:p>When a plough covered by subclass <xhtml:a href="A01B" rel="symbol"/> is also a harrow, classify in '
    '<xhtml:a href="A01C0005000000" rel="symbol"/>.</xhtml:p>'
    '<xhtml:p>Main group <xhtml:a href="A01B0003000000" rel="symbol"/> takes precedence over all other main groups '
    'of subclass <xhtml:a href="A01B" rel="symbol"/></xhtml:p></xhtml:div></SPECIALRULES>'
    '<REFERENCES><LIMITINGREFERENCES><xhtml:p>Harrows <xhtml:a href="A01B0003020000" rel="symbol"/> - '
    '<xhtml:a href="A01B0003040000" rel="symbol"/></xhtml:p></LIMITINGREFERENCES></REFERENCES>'
    "</IPC-DEFINITION></IPCDefinitionsClass></IPCDefinitionsSection></IPCDefinitions>"
)

CATCHWORDS = (
    f'<?xml version="1.0" encoding="UTF-8"?><IPCCatchwordIndex {NS} edition="20260101" lang="EN">'
    '<CWEntry id="CW1"><CWIndication>PLOUGHS</CWIndication><CWEntry id="CW2"><CWIndication>swing PLOUGHS'
    '</CWIndication><CWReferences><sref ref="A01B0003080000"/></CWReferences></CWEntry></CWEntry>'
    "</IPCCatchwordIndex>"
)


def _zip(tmp_path, name, files):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, content in files.items():
            archive.writestr(member, content)
    return path


@pytest.fixture
def scheme(tmp_path):
    path = _zip(tmp_path, "scheme.zip", {
        "EN_ipc_scheme_20260101.xml": _scheme("EN", "Ploughs with fixed plough-shares"),
        "FR_ipc_scheme_20260101.xml": _scheme("FR", "Charrues à socs fixes"),
    })
    return parse_scheme(path)


def _rules(texts, owner, kind=None):
    return [r for t in texts if t.owner_symbol == owner for r in t.rules if kind is None or r["kind"] == kind]


def test_display_symbol():
    assert display_symbol("A01B0003040000") == "A01B 3/04"
    assert display_symbol("A01B0003000000") == "A01B 3/00"
    assert display_symbol("H04W0012043300") == "H04W 12/0433"
    assert display_symbol("A01B") == "A01B"


def test_english_file_chosen_and_dotted_hierarchy_preserved(scheme):
    edition, entries, _ = scheme
    by = {e.code: e for e in entries}
    assert edition == "20260101"
    assert by["A01B0003000000"].title == "Ploughs with fixed plough-shares"
    swing = by["A01B0003080000"]
    assert swing.parent_code == "A01B0003060000"
    assert swing.ancestor_codes == ["A", "A01", "A01B", "A01B0003000000", "A01B0003040000", "A01B0003060000"]
    assert swing.maingroup_code == "A01B0003000000" and swing.subclass_code == "A01B"
    assert by["A01B0003040000"].child_codes == ["A01B0003060000"]
    assert by["A01B0003990000"].entry_type == "I"


def test_semantic_text_carries_every_dotted_parent(scheme):
    _, entries, _ = scheme
    by = {e.code: e for e in entries}
    text = semantic_text(by["A01B0003080000"], by)
    assert text.startswith("IPC A01B 3/08\n\nSection:\nHUMAN NECESSITIES")
    assert "Main group:\nPloughs with fixed plough-shares" in text
    assert "Subgroup (1 dot):\nAnimal-drawn ploughs" in text
    assert "Subgroup (2 dots):\nwithout alternating possibility" in text
    assert text.endswith("Subgroup (3 dots):\nSwing ploughs")


def test_note_keeps_owner_container_range_and_edition(scheme):
    _, _, texts = scheme
    note = next(t for t in texts if t.owner_symbol == "A01B0003020000")
    assert note.source_kind == "scheme_note" and note.note_type == "number"
    assert note.container_symbol == "A01B0003000000"
    assert note.scope_end_symbol == "A01B0003040000"
    assert note.scope_symbols == ["A01B0003020000", "A01B0003040000"]  # top-level entries of the range
    assert note.editions == "20000101"
    assert note.text == "Group A01B 3/24 takes precedence over groups A01B 3/02-A01B 3/04."
    assert note.rule_kind == "precedence"
    assert note.rules[0]["preferred"] == ["A01B0003240000"]
    assert note.rules[0]["dominated"] == ["A01B0003020000", "A01B0003040000"]
    assert note.rules[0]["conditional"] is False


def test_title_reference_precedence_prefers_cited_place_over_owner(scheme):
    _, _, texts = scheme
    ref = next(t for t in texts if t.owner_symbol == "A01B0003240000")
    assert ref.source_kind == "scheme_reference" and ref.editions == "19680901"
    assert ref.rules == [{"kind": "precedence", "preferred": ["A01B0003040000"], "dominated": ["A01B0003240000"],
                          "rest_of_scope": False, "conditional": False, "sentence": "A01B 3/04 takes precedence"}]


def test_subclass_notes_read_by_meaning(scheme):
    _, _, texts = scheme
    notes = sorted((t for t in texts if t.owner_symbol == "A01B" and t.source_kind == "scheme_note"),
                   key=lambda t: t.ordinal)
    over_other, last_place, conditional, negated, also = notes
    assert also.rules == []  # an instruction to add a code, not a choice between places
    assert over_other.rules[0]["preferred"] == ["A01B0003300000"]
    assert over_other.rules[0]["dominated"] == ["A01B"] and over_other.rules[0]["rest_of_scope"]
    assert last_place.rule_kind == "place_last" and last_place.rules[0]["scope"] == ["A01B"]
    assert conditional.rules[0]["preferred"] == ["F26"] and conditional.rules[0]["conditional"] is True
    assert negated.rules == [] and negated.rule_kind == "other"
    limiting = next(t for t in texts if t.owner_symbol == "A01B" and t.source_kind == "scheme_reference")
    assert limiting.rule_kind == "limiting_reference"
    assert limiting.rules == [{"kind": "limiting_reference", "targets": ["A01C0005000000"]}]


def test_definitions_keep_paragraphs_apart_and_merge_ranges(tmp_path, scheme):
    _, entries, _ = scheme
    path = _zip(tmp_path, "defs.zip", {"EN_ipc_definitions_20260101.xml": DEFINITIONS,
                                       "FR_ipc_definitions_20260101.xml": DEFINITIONS.replace('"EN"', '"FR"')})
    edition, texts = parse_definitions(path, entries)
    assert edition == "20260101"
    special = next(t for t in texts if t.part_type == "SPECIALRULES")
    assert "A01C 5/00. Main group A01B 3/00 takes precedence" in special.text
    assert special.rules[0]["preferred"] == ["A01B0003000000"]  # not the symbols of the previous paragraph
    assert special.rules[0]["dominated"] == ["A01B"]
    limiting = next(t for t in texts if t.part_type == "LIMITINGREFERENCES")
    assert limiting.text == "Harrows A01B 3/02-A01B 3/04"
    assert limiting.rules[0]["targets"] == ["A01B0003020000", "A01B0003040000"]


def test_catchwords_nested(tmp_path):
    path = _zip(tmp_path, "cw.zip", {"EN_ipc_catchwordindex_20260101.xml": CATCHWORDS})
    _, rows = parse_catchwords(path)
    assert [(r.phrase, r.target_symbols, r.parent_phrase) for r in rows] == [
        ("PLOUGHS", [], None), ("swing PLOUGHS", ["A01B0003080000"], "PLOUGHS")]


# --- rule application ---------------------------------------------------------------------------

def _ctx(owner, rules, source_kind="scheme_note", text="rule"):
    return {"owner_symbol": owner, "source_kind": source_kind, "part_type": "note", "editions": "20000101",
            "scope_end_symbol": None, "text": text, "rules": rules}


ANCESTORS = ["A", "A01", "A01B", "A01B0003000000"]
TITLE_RULE = _ctx("A01B0003240000", [{"kind": "precedence", "preferred": ["A01B0003040000"],
                                      "dominated": ["A01B0003240000"], "conditional": False,
                                      "sentence": "A01B 3/04 takes precedence"}], source_kind="scheme_reference")


def _cand(code, context=(), ancestors=ANCESTORS):
    return RuleCandidate(code, display_symbol(code), list(ancestors), list(context))


def test_precedence_demotes_dominated_candidate_only_when_preferred_is_supported():
    x, y = _cand("A01B0003240000", [TITLE_RULE]), _cand("A01B0003040000")
    out = apply_scheme_rules([x, y], [], precedence=True, place=True)
    assert set(out.demoted) == {"A01B0003240000"} and "A01B 3/04 takes precedence" in out.demoted["A01B0003240000"]
    alone = apply_scheme_rules([x], [y], precedence=True, place=True)
    assert alone.demoted == {}  # the rule never adds A01B 3/04
    assert "was ranked but not supported" in alone.notes["A01B0003240000"][0]
    disabled = apply_scheme_rules([x, y], [], precedence=False, place=True)
    assert disabled.demoted == {} and "not applied" in disabled.notes["A01B0003240000"][0]


def test_precedence_covers_descendants_and_conditional_rules_only_note():
    sub = ["A", "A01", "A01B", "A01B0003000000", "A01B0003040000"]
    rule = _ctx("A01B", [{"kind": "precedence", "preferred": ["A01B0003300000"], "dominated": ["A01B"],
                          "conditional": False, "sentence": "group 3/30 takes precedence over other groups"}])
    x = _cand("A01B0003060000", [rule], ancestors=sub)
    y = _cand("A01B0003300000", [rule])
    assert set(apply_scheme_rules([x, y], [], precedence=True, place=False).demoted) == {"A01B0003060000"}
    cond = _ctx("A01B", [{**rule["rules"][0], "conditional": True}])
    out = apply_scheme_rules([_cand("A01B0003060000", [cond], ancestors=sub), y], [], precedence=True, place=False)
    assert out.demoted == {} and "conditional rule" in out.notes["A01B0003060000"][0]


def test_place_rule_keeps_last_place_and_skips_ancestors():
    rule = _ctx("A01B", [{"kind": "place_last", "scope": ["A01B"], "sentence": "last appropriate place"}])
    early, late = _cand("A01B0003020000", [rule]), _cand("A01B0003300000", [rule])
    out = apply_scheme_rules([early, late], [], precedence=True, place=True)
    assert set(out.demoted) == {"A01B0003020000"}
    parent = _cand("A01B0003000000", [rule], ancestors=["A", "A01", "A01B"])
    assert apply_scheme_rules([parent, late], [], precedence=True, place=True).demoted == {}  # ancestor pair
    assert apply_scheme_rules([early, late], [], precedence=True, place=False).demoted == {}


def test_mutual_rules_keep_both_for_review_and_limiting_reference_only_notes():
    a = _cand("A01B0003020000", [_ctx("x", [{"kind": "precedence", "preferred": ["A01B0003300000"],
                                             "dominated": ["A01B0003020000"], "conditional": False}])])
    b = _cand("A01B0003300000", [_ctx("y", [{"kind": "precedence", "preferred": ["A01B0003020000"],
                                             "dominated": ["A01B0003300000"], "conditional": False}])])
    out = apply_scheme_rules([a, b], [], precedence=True, place=True)
    assert out.demoted == {} and "Conflicting" in out.notes["A01B0003020000"][-1]
    lim = _cand("A01B", [_ctx("A01B", [{"kind": "limiting_reference", "targets": ["A01C0005000000"]}])],
                ancestors=["A", "A01"])
    cited = _cand("A01C0005000000", ancestors=["A", "A01", "A01C"])
    out = apply_scheme_rules([lim, cited], [], precedence=True, place=True)
    assert out.demoted == {} and "classified in A01C 5/00" in out.notes["A01B"][0]


# --- claims weighting, fusion weights, routing recall ---------------------------------------------

def test_weighted_rrf_and_default_weights_unchanged():
    assert rrf([["A"], ["A"]], 60, [1.0, 0.5])["A"] == pytest.approx(1.5 / 61)
    abstract = ViewHits("abstract", "abstract", [("A", 1, 1.0)], [("B", 1, 0.9)])
    claim = ViewHits("claims[1]", "claims", [("C", 1, 1.0)], [("C", 1, 0.9)])
    plain = {c.code: c.rrf_score for c in fuse([abstract, claim], k=60, per_source_cap=100)}
    same = {c.code: c.rrf_score for c in fuse([abstract, claim], k=60, per_source_cap=100,
                                              source_weights={"claims": 1.0})}
    assert plain == same
    weighted = {c.code: c.rrf_score for c in fuse([abstract, claim], k=60, per_source_cap=100,
                                                  source_weights={"claims": 0.35})}
    assert weighted["C"] == pytest.approx(0.35 * plain["C"]) and weighted["A"] == pytest.approx(plain["A"])


def test_select_retrieval_views_independent_claims_only():
    plan = build_views("A plough.", None, "1. A plough comprising a share.\n2. The plough of claim 1, wherein the "
                       "share is fixed.\n3. A method of ploughing a field.", None,
                       max_views=10, chunk_tokens=400, chunk_overlap=20)
    off = SimpleNamespace(claims_retrieval_mode="off", independent_claim_weight=1.0, dependent_claim_weight=0.0)
    views, weights = select_retrieval_views(plan.views, off)
    assert [v.source for v in views] == ["abstract"] and weights == {}
    on = SimpleNamespace(claims_retrieval_mode="independent_weighted", independent_claim_weight=1.0,
                         dependent_claim_weight=0.0)
    views, weights = select_retrieval_views(plan.views, on)
    assert [v.view_id for v in views] == ["abstract", "claims[1]", "claims[3]"]
    assert weights == {"claims[1]": 1.0, "claims[3]": 1.0}
    on.dependent_claim_weight = 0.25
    assert select_retrieval_views(plan.views, on)[1]["claims[2]"] == 0.25


def test_subclass_routing_recall_at_cutoffs():
    gt = bm.GroundTruth(symbols=parse_symbols("H04L 9/08"))
    row = bm.score_document(gt, ["H04W 12/04"], ["H04W 12/04", "H04L 9/08"], "H04",
                            retrieval_debug={"stage1_subclasses": ["H04W", "H04L"],
                                             "stage2_group_candidates": ["H04W 12/04", "H04L 9/08"],
                                             "reranked_candidates": ["H04W 12/04", "H04L 9/08"]})
    stages = row["exact"]["stage_recall"]
    assert stages["subclass_routing@1"] == 0 and stages["subclass_routing@5"] == 1 and stages["subclass_routing"] == 1
    assert stages["group_candidate_generation"] == 1 and row["exact"]["hit@1"] is False and row["exact"]["hit@3"]
    agg = bm.aggregate([row])
    assert agg["exact"]["stage_recall"]["subclass_routing@1"]["mean_recall"] == 0


def test_vote_subclasses_sums_group_scores():
    from app.services.retrieval.fusion import FusedCandidate, vote_subclasses

    fused = [FusedCandidate("H04W0012040000", 0.030), FusedCandidate("H04L0009080000", 0.020),
             FusedCandidate("H04L0009140000", 0.015), FusedCandidate("G06N0003080000", 0.010)]
    assert vote_subclasses(fused, 2) == ["H04L", "H04W"]  # 0.035 beats a single stronger hit
