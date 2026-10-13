"""TASK-0015: WIPO revision concordance for abolished IPC symbols (H01L -> H10 and others)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from app.services.patents.concordance import (
    ConcordanceError,
    WipoConcordance,
    common_ancestor,
    internal_to_display,
    load_concordance,
    normalise_display,
    read_concordance_xml,
)
from app.services.patents.labels import ResolvedLabel

REPO = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO / "data" / "source"
MANIFEST = SOURCE_DIR / "wipo_ipc_concordance_manifest.json"
NS = "http://www.wipo.int/classifications/ipc/masterfiles"


def _xml(from_version: str, to_version: str, entries: list[tuple[str, str, list[tuple[str, str]]]]) -> bytes:
    """entries: (from-symbol, modification, [(to-symbol, default Y/N)])"""
    body = "".join(
        f'<concordance from-symbol="{f}" modification="{m}">'
        + "".join(f'<concordance-to default-reclassification="{d}" to-symbol="{t}" modification="n"/>' for t, d in tos)
        + "</concordance>"
        for f, m, tos in entries
    )
    return (f"<?xml version='1.0' encoding='UTF-8'?><IPCConcordanceList xmlns=\"{NS}\" from-version=\"{from_version}\""
            f' to-version="{to_version}">{body}</IPCConcordanceList>').encode()  # fmt: skip


def _zip(path: Path, xml: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.stem.replace("wipo_ipc_concordance", "ipc_concordancelist") + ".xml", xml)
    return path


# Two synthetic editions. 2023: A98A 1/10 is deleted into one place, A98A 1/20 into one main group, A98A 5/00
# into two main groups of one subclass, A98A 3/00 into two subclasses, A98A 2/00 into B98B 6/10 and B98B 7/00,
# A98A 4/00 changed (still exists). 2026: B98B 7/00 is itself deleted into C98C 9/10, so A98A 2/00 ends up in
# two subclasses through the chain.
ED_2023 = _xml("20220101", "20230101", [
    ("A98A0001100000", "d", [("B98B0005100000", "Y")]),
    ("A98A0001200000", "d", [("B98B0006000000", "N"), ("B98B0006100000", "Y"), ("B98B0006200000", "N")]),
    ("A98A0002000000", "d", [("B98B0006100000", "Y"), ("B98B0007000000", "N")]),
    ("A98A0003000000", "d", [("B98B0005100000", "Y"), ("D98D0001000000", "N")]),
    ("A98A0004000000", "c", [("A98A0004000000", "Y"), ("A98A0004100000", "N")]),
    ("A98A0005000000", "d", [("B98B0005100000", "Y"), ("B98B0006200000", "N")]),
])  # fmt: skip
ED_2026 = _xml("20250101", "20260101", [("B98B0007000000", "d", [("C98C0009100000", "Y")])])
CATALOGUE = {"B98B", "B98B 5/00", "B98B 5/10", "B98B 6/00", "B98B 6/10", "B98B 6/20", "C98C", "C98C 9/00",
             "C98C 9/10", "D98D", "D98D 1/00", "A98A", "A98A 4/00", "A98A 4/10"}  # fmt: skip


@pytest.fixture
def synthetic() -> WipoConcordance:
    eds = [read_concordance_xml(ED_2026)[1:], read_concordance_xml(ED_2023)[1:]]  # order does not matter
    return WipoConcordance(eds, CATALOGUE)


def test_symbol_formats() -> None:
    assert internal_to_display("H01L0021027000") == "H01L 21/027"
    assert internal_to_display("H10W0076000000") == "H10W 76/00"
    assert internal_to_display("A61L0002080000") == "A61L 2/08"
    assert internal_to_display("F24J0000000000") == "F24J"
    with pytest.raises(ConcordanceError):
        internal_to_display("H01L21027")
    assert normalise_display("h01l 21/027 ") == "H01L 21/027" and normalise_display("H01L33/00") == "H01L 33/00"
    assert normalise_display("H01L0021027000") == "H01L 21/027" and normalise_display("H01L21027") is None


def test_only_deletions_are_mapped(synthetic: WipoConcordance) -> None:
    assert set(synthetic) == {"A98A 1/10", "A98A 1/20", "A98A 2/00", "A98A 3/00", "A98A 5/00", "B98B 7/00"}  # not the "c" entry
    assert synthetic.apply("A98A 4/00") is None


def test_one_to_one_mapping(synthetic: WipoConcordance) -> None:
    assert synthetic.apply("A98A 1/10") == ["B98B 5/10"]
    assert synthetic.resolve("A98A 1/10") == ResolvedLabel("B98B", "B98B 5/00", "B98B 5/10")


def test_one_to_many_votes_for_the_common_ancestor(synthetic: WipoConcordance) -> None:
    # same main group: the main group, with no exact code (no invented precision)
    assert synthetic.apply("A98A 1/20") == ["B98B 6/00", "B98B 6/10", "B98B 6/20"]
    assert synthetic.resolve("A98A 1/20") == ResolvedLabel("B98B", "B98B 6/00", None)
    # same subclass, different main groups: the subclass only
    assert synthetic.apply("A98A 5/00") == ["B98B 5/10", "B98B 6/20"]
    assert synthetic.resolve("A98A 5/00") == ResolvedLabel("B98B", None, None)
    # through the 2026 chain one target left the subclass: no common ancestor
    assert synthetic.apply("A98A 2/00") == ["B98B 6/10", "C98C 9/10"]
    assert synthetic.resolve("A98A 2/00") is None
    # several subclasses: apply lists them, resolve gives None (no common ancestor at a vote level)
    assert synthetic.apply("A98A 3/00") == ["B98B 5/10", "D98D 1/00"]
    assert synthetic.resolve("A98A 3/00") is None


def test_chained_editions(synthetic: WipoConcordance) -> None:
    assert synthetic["B98B 7/00"] == ["C98C 9/10"]
    assert [t.edition for t in synthetic.transfers("B98B 7/00")] == ["20260101"]
    assert synthetic.transfers("A98A 2/00")[0].defaults == ("B98B 6/10",)  # default flag is kept, not used


def test_unmapped_code_returns_none(synthetic: WipoConcordance) -> None:
    for code in ("H98Z 1/00", "H98Z100", "B98B 5/10", "not a code", ""):
        assert synthetic.apply(code) is None and synthetic.resolve(code) is None


def test_input_forms_and_current_symbol_guard(synthetic: WipoConcordance) -> None:
    for form in ("A98A 1/10", "a98a1/10", "A98A0001100000", "A98A110"):
        assert synthetic.apply(form) == ["B98B 5/10"], form
    # a compact code whose split names a current symbol is never remapped
    guarded = WipoConcordance([read_concordance_xml(ED_2023)[1:]], CATALOGUE | {"A98A 1/10"})
    assert guarded.apply("A98A110") is None and guarded.apply("A98A 1/10") is None


def test_targets_outside_the_catalogue_are_dropped() -> None:
    conc = WipoConcordance([read_concordance_xml(ED_2023)[1:]], {"B98B", "B98B 6/00", "B98B 6/10"})
    assert conc.apply("A98A 1/20") == ["B98B 6/00", "B98B 6/10"]
    assert conc.apply("A98A 1/10") is None  # its only target is not in the catalogue
    unchecked = WipoConcordance([read_concordance_xml(ED_2023)[1:]], None)
    assert unchecked.apply("A98A 1/10") == ["B98B 5/10"]


def test_common_ancestor_rules() -> None:
    assert common_ancestor([]) is None
    assert common_ancestor(["H10W 76/12"]) == ResolvedLabel("H10W", "H10W 76/00", "H10W 76/12")
    assert common_ancestor(["H10W 76/00"]) == ResolvedLabel("H10W", "H10W 76/00", "H10W 76/00")
    assert common_ancestor(["H10W 76/10", "H10W 76/60"]) == ResolvedLabel("H10W", "H10W 76/00", None)
    assert common_ancestor(["H10P 10/00", "H10P 56/00"]) == ResolvedLabel("H10P", None, None)
    assert common_ancestor(["H10D 62/00", "H10W 29/00"]) is None
    assert common_ancestor(["H10W 76/12"], {"H10W", "H10W 76/12"}) == ResolvedLabel("H10W", None, "H10W 76/12")
    assert common_ancestor(["H10W 76/12"], {"H10X"}) is None


def test_load_concordance_from_a_directory_checks_the_manifest(tmp_path: Path) -> None:
    a = _zip(tmp_path / "wipo_ipc_concordance_20230101.zip", ED_2023)
    b = _zip(tmp_path / "wipo_ipc_concordance_20260101.zip", ED_2026)
    conc = load_concordance(tmp_path, CATALOGUE)  # no manifest: loads unchecked
    assert conc.sources == (a.name, b.name) and conc.apply("A98A 1/10") == ["B98B 5/10"]
    files = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (a, b)]
    (tmp_path / "wipo_ipc_concordance_manifest.json").write_text(json.dumps({"files": files}))
    assert len(load_concordance(tmp_path, CATALOGUE)) == 6
    _zip(b, ED_2026.replace(b"C98C0009100000", b"C98C0009200000"))  # tampered after the manifest was written
    with pytest.raises(ConcordanceError, match="SHA-256"):
        load_concordance(tmp_path)
    assert load_concordance(a).apply("A98A 1/10") == ["B98B 5/10"]  # a single zip
    xml = tmp_path / "list.xml"
    xml.write_bytes(ED_2023)
    assert len(load_concordance(xml)) == 5  # a single XML


def test_load_concordance_errors(tmp_path: Path) -> None:
    with pytest.raises(ConcordanceError, match="no wipo_ipc_concordance"):
        load_concordance(tmp_path)
    with pytest.raises(ConcordanceError, match="not found"):
        load_concordance(tmp_path / "missing")
    bad = tmp_path / "bad.xml"
    bad.write_bytes(b"<other/>")
    with pytest.raises(ConcordanceError, match="not a WIPO"):
        load_concordance(bad)
    _zip(tmp_path / "wipo_ipc_concordance_20230101.zip", ED_2023)
    _zip(tmp_path / "wipo_ipc_concordance_2023b.zip", ED_2023)
    with pytest.raises(ConcordanceError, match="duplicate editions"):
        load_concordance(tmp_path)


# --- the stored WIPO lists (data/source, LFS) ---------------------------------------------------------------


@pytest.fixture(scope="module")
def wipo() -> WipoConcordance:
    if not MANIFEST.is_file() or MANIFEST.read_bytes().startswith(b"version https://git-lfs"):
        pytest.skip("WIPO concordance files not present (git lfs pull)")
    catalogue = {"H10P", "H10P 76/00", "H10P 76/20", "H10W", "H10W 76/00", "H10W 76/12", "G06F", "G06F 18/00",
                 "H04L", "H04L 9/00", "H04L 9/40", "H04L 65/00", "H04L 65/60", "H04L 67/00", "H04L 67/01", "A01B", "A01B 1/00", "H10D", "H10D 62/00", "H10W 29/00",
                 *(f"G06F 18/{s}" for s in ("10", "15", "20", "21", "211", "2111"))}  # fmt: skip
    return load_concordance(SOURCE_DIR, catalogue)


def test_stored_lists_match_the_manifest() -> None:
    if not MANIFEST.is_file() or MANIFEST.read_bytes().startswith(b"version https://git-lfs"):
        pytest.skip("WIPO concordance files not present (git lfs pull)")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert [f["edition"] for f in manifest["files"]] == [f"{y}.01" for y in range(2017, 2027)]
    for f in manifest["files"]:
        path = SOURCE_DIR / f["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"]
        assert f["url"].startswith("https://www.wipo.int/classifications/data/ipc/ITSupport_and_download_area/")


def test_h01l_examples(wipo: WipoConcordance) -> None:
    # one-to-one: H01L 23/04 (deleted 2026.01) -> H10W 76/12
    assert wipo.apply("H01L2304") == ["H10W 76/12"]
    assert wipo.resolve("H01L2304") == ResolvedLabel("H10W", "H10W 76/00", "H10W 76/12")
    # one-to-many within a main group: H01L 21/027 -> H10P 76/00, H10P 76/20 -> vote for H10P 76/00 only
    assert wipo.apply("H01L21027") == ["H10P 76/00", "H10P 76/20"]
    assert wipo.resolve("H01L 21/027") == ResolvedLabel("H10P", "H10P 76/00", None)
    # one-to-many across subclasses: H01L 21/71 -> H10D 62/00 and H10W 29/00, no common vote-level ancestor
    assert wipo.apply("H01L2171") == ["H10D 62/00", "H10W 29/00"] and wipo.resolve("H01L2171") is None


def test_other_abolished_symbols(wipo: WipoConcordance) -> None:
    assert wipo.resolve("G06K900") == ResolvedLabel("G06F", "G06F 18/00", None)  # deleted 2023.01
    assert wipo.resolve("H04L2906") == ResolvedLabel("H04L", None, None)  # deleted 2022.01, 48 targets in H04L
    assert wipo.apply("A01B100") is None  # a current symbol: unmapped
