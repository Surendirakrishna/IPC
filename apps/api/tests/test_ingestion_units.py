from app.services.ingestion.sources import IndexLine, IndexRecord
from app.services.ingestion.tree import reference_wording, select_canonical_occurrence, traverse
from app.services.ingestion.validate import validate_sources

TREE = {
    "symbol": "IPC", "title": "International Patent Classification (CPC Scheme)", "version": "1",
    "children": [{"symbol": "A", "title": "SEC", "type": "section", "children": [
        {"symbol": "A01", "title": "CLS", "type": "class", "children": [
            {"symbol": "A01B", "title": "SUB (see A01C0005000000)", "type": "subclass", "children": [
                {"symbol": "A01B0003000000", "title": "Ploughs", "type": "maingroup", "children": []},
                {"symbol": "A01B0003000000", "title": "Ploughs with fixed plough-shares", "type": "maingroup",
                 "children": [{"symbol": "A01B0003020000", "title": "Sub", "type": "subgroup"}]},
            ]}]}]}],
}


def _line(n, **kw):
    base = dict(section="A", section_title="SEC", taxonomy_version="1", semantic_text="x", heading_text=[],
                cross_references=[], precedence_references=[], aliases=[], child_codes=[], ancestor_codes=[])
    base.update(kw)
    return IndexLine(n, base, IndexRecord.model_validate(base))


def test_traverse_preserves_repeated_occurrences():
    root, occ = traverse(TREE)
    assert root["title"].endswith("(CPC Scheme)")
    codes = [o.code for o in occ]
    assert codes.count("A01B0003000000") == 2
    assert len({o.occurrence_id for o in occ}) == len(occ)
    sub = next(o for o in occ if o.code == "A01B0003020000")
    assert sub.source_path == ["A", "A01", "A01B", "A01B0003000000", "A01B0003020000"]


def test_canonical_occurrence_selection():
    _, occ = traverse(TREE)
    mg = [o for o in occ if o.code == "A01B0003000000"]
    chosen, reason = select_canonical_occurrence("Ploughs with fixed plough-shares", "Ploughs with fixed plough-shares", mg)
    assert chosen.original_title == "Ploughs with fixed plough-shares" and reason == "normalized_title_match"


def test_reference_wording_is_verbatim_or_none():
    assert reference_wording("Hand tools (edge trimmers for lawns A01G0003060000)", "A01G0003060000") == \
        "edge trimmers for lawns A01G0003060000"
    assert reference_wording("Hand tools", "A01G0003060000") is None


def test_validation_flags_hierarchy_and_source_quality():
    lines = [
        _line(1, code="A", display_code="A", type="section", title="SEC", depth=1, child_codes=["A01"]),
        _line(2, code="A01", display_code="A01", type="class", title="CLS", depth=2, parent_code="A",
              ancestor_codes=["A"], child_codes=["A01B"]),
        _line(3, code="A01B", display_code="A01B", type="subclass", title="SUB", depth=3, parent_code="A01",
              ancestor_codes=["A", "A01"], child_codes=["A01B0003000000"], cross_references=["A01C0005000000"]),
        _line(4, code="A01B0003000000", display_code="A01B 3/00", type="maingroup", title="Ploughs with fixed plough-shares",
              depth=4, parent_code="A01B", ancestor_codes=["A", "A01", "A01B"], child_codes=["A01B0003020000"],
              is_duplicate_symbol=True),
        _line(5, code="A01B0003020000", display_code="A01B 3/02", type="subgroup", title="Sub", depth=5,
              parent_code="A01B0003000000", ancestor_codes=["A", "A01", "A01B", "A01B0003000000"]),
    ]
    root, occ = traverse(TREE)
    rep = validate_sources(lines, [], {}, root, occ, [{"code": "A01B0003000000"}], None)
    assert rep.ok
    assert rep.findings["dangling_references"]["count"] == 1  # preserved, not blocking
    assert any("CPC" in s for s in rep.source_quality)
    bad = list(lines)
    bad[4] = _line(5, code="A01B0003020000", display_code="A01B 3/00", type="subgroup", title="Sub", depth=5,
                   parent_code="A01B0003000000", ancestor_codes=["A", "A01", "A01B", "A01B0003000000"])
    rep2 = validate_sources(bad, [], {}, root, occ, [{"code": "A01B0003000000"}], None)
    assert not rep2.ok and "display_code_collisions" in rep2.blocking
