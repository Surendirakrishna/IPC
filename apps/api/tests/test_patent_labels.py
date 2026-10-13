"""TASK-0003: patent IPC label resolution against an in-memory catalogue (no database)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.patents.labels import (
    Outcome,
    ResolutionStats,
    ResolvedLabel,
    _compact_outcome,
    resolve_compact,
    resolve_labels,
    resolve_structured,
)

# A slice of the 2026.01 catalogue, as display codes. Membership mirrors the real catalogue for every
# code used below: H04L 1/218, H04L 29/xx, G06K 9/00 and H01L do not exist in 2026.01.
CATALOGUE = frozenset(
    {
        "G16H", "G16H 10/00", "G16H 10/60",
        "H04L", "H04L 1/00", "H04L 1/18", "H04L 9/00", "H04L 9/08", "H04L 9/0838", "H04L 12/00", "H04L 12/18",
        "G06K", "G06K 7/00", "G06K 7/10",
        "A61B", "A61B 5/00", "A61B 5/022", "A61B 50/00", "A61B 50/22",
        "H10B", "H10B 10/00", "H10K", "H10K 50/00",
        "B65D", "B65D 51/00",  # no B65D 5/00, so 51 is the unique main group of B65D5199
    }
)

SOURCE_DIR = Path(__file__).resolve().parents[3] / "data" / "source"
CATALOGUE_FILE = SOURCE_DIR / "ipc_classification_index.jsonl"


# --- resolve_compact: the gotcha-note samples ----------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("G16H1060", ResolvedLabel("G16H", "G16H 10/00", "G16H 10/60")),
        ("H04L1218", ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18")),  # not H04L 1/218
        ("H04L100", ResolvedLabel("H04L", "H04L 1/00", "H04L 1/00")),
        ("H04L2906", ResolvedLabel("H04L", None, None)),  # H04L 29/06 abolished
        ("G06K900", ResolvedLabel("G06K", None, None)),  # G06K 9/00 abolished
        ("H04L90838", ResolvedLabel("H04L", "H04L 9/00", "H04L 9/0838")),
    ],
)
def test_compact_gotcha_samples(code: str, expected: ResolvedLabel) -> None:
    assert resolve_compact(code, CATALOGUE) == expected


def test_compact_outcomes_for_fallback_samples() -> None:
    assert _compact_outcome("H04L2906", CATALOGUE)[0] is Outcome.SUBCLASS_ONLY
    assert _compact_outcome("G06K900", CATALOGUE)[0] is Outcome.SUBCLASS_ONLY
    assert _compact_outcome("G16H1060", CATALOGUE)[0] is Outcome.EXACT


def test_compact_ambiguous_keeps_only_subclass() -> None:
    # A61B5022 is both A61B 5/022 and A61B 50/22 in 2026.01.
    outcome, label = _compact_outcome("A61B5022", CATALOGUE)
    assert outcome is Outcome.AMBIGUOUS
    assert label == ResolvedLabel("A61B", None, None)


def test_compact_unique_maingroup_fallback() -> None:
    # B65D 51/99 does not exist; of the splits 5/199 and 51/99 only main group 51 exists.
    outcome, label = _compact_outcome("B65D5199", CATALOGUE)
    assert outcome is Outcome.MAINGROUP_ONLY
    assert label == ResolvedLabel("B65D", "B65D 51/00", None)


def test_compact_two_existing_maingroups_fall_back_to_subclass() -> None:
    # Splits 5/099 and 50/99: both main groups exist, no subgroup does -> no unique main group.
    outcome, label = _compact_outcome("A61B5099", CATALOGUE)
    assert outcome is Outcome.SUBCLASS_ONLY
    assert label == ResolvedLabel("A61B", None, None)


def test_compact_h01l_is_missing() -> None:
    assert resolve_compact("H01L21027", CATALOGUE) is None
    assert _compact_outcome("H01L21027", CATALOGUE)[0] is Outcome.MISSING


@pytest.mark.parametrize("code", ["", "garbage", "H04L 12/18", "Z99Z1200", "H04"])
def test_compact_unparseable_is_none(code: str) -> None:
    assert resolve_compact(code, CATALOGUE) is None


def test_compact_normalises_case_and_whitespace() -> None:
    assert resolve_compact(" g16h1060 ", CATALOGUE) == ResolvedLabel("G16H", "G16H 10/00", "G16H 10/60")


def test_compact_bare_subclass_is_subclass_only() -> None:
    assert _compact_outcome("H10K", CATALOGUE) == (Outcome.SUBCLASS_ONLY, ResolvedLabel("H10K", None, None))


# --- resolve_structured ----------------------------------------------------------------------------


def test_structured_exact() -> None:
    assert resolve_structured("H", "04", "L", "12", "18", CATALOGUE) == ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18")


def test_structured_normalises_padding() -> None:
    # Zero-padded main group, insignificant trailing zeros in the subgroup, integer class.
    assert resolve_structured("H", 4, "L", "0009", "080000", CATALOGUE) == ResolvedLabel("H04L", "H04L 9/00", "H04L 9/08")
    assert resolve_structured("H", "04", "L", "9", "0838", CATALOGUE) == ResolvedLabel("H04L", "H04L 9/00", "H04L 9/0838")
    assert resolve_structured("h", "04", "l", 1, "00", CATALOGUE) == ResolvedLabel("H04L", "H04L 1/00", "H04L 1/00")


def test_structured_accepts_full_subclass() -> None:
    assert resolve_structured(None, None, "G16H", "10", "60", CATALOGUE) == ResolvedLabel("G16H", "G16H 10/00", "G16H 10/60")


def test_structured_abolished_subgroup_falls_back_to_maingroup_or_subclass() -> None:
    # H04L 12/99 does not exist, main group 12 does.
    assert resolve_structured("H", "04", "L", "12", "99", CATALOGUE) == ResolvedLabel("H04L", "H04L 12/00", None)
    # H04L 29/06 and its main group are gone.
    assert resolve_structured("H", "04", "L", "29", "06", CATALOGUE) == ResolvedLabel("H04L", None, None)
    # G06K 9/00 is gone.
    assert resolve_structured("G", "06", "K", "9", "00", CATALOGUE) == ResolvedLabel("G06K", None, None)


def test_structured_missing_parts() -> None:
    assert resolve_structured("H", "04", "L", None, None, CATALOGUE) == ResolvedLabel("H04L", None, None)
    assert resolve_structured("H", "04", "L", "12", None, CATALOGUE) == ResolvedLabel("H04L", "H04L 12/00", None)
    assert resolve_structured("H", "04", "L", "", "", CATALOGUE) == ResolvedLabel("H04L", None, None)


def test_structured_h01l_and_bad_subclass_are_none() -> None:
    assert resolve_structured("H", "01", "L", "21", "027", CATALOGUE) is None
    assert resolve_structured("H", "01", "L", None, None, CATALOGUE) is None
    assert resolve_structured(None, None, None, "1", "00", CATALOGUE) is None
    assert resolve_structured("X", "04", "L", "1", "00", CATALOGUE) is None


def test_structured_single_digit_subgroup_ambiguity() -> None:
    # "8" could be 9/80 or 9/08; only 9/08 exists -> exact.
    assert resolve_structured("H", "04", "L", "9", "8", CATALOGUE) == ResolvedLabel("H04L", "H04L 9/00", "H04L 9/08")
    both = CATALOGUE | {"H04L 9/80"}
    outcome = resolve_labels([("H", "04", "L", "9", "8")], "structured", both).stats
    assert outcome.ambiguous == 1


# --- resolve_labels ----------------------------------------------------------------------------------


def test_resolve_labels_order_dedup_and_stats() -> None:
    raw = ["H04L1218", "G16H1060", "h04l1218", "H01L21027", "A61B5022", "B65D5199", "G06K900", "G16H1060"]
    res = resolve_labels(raw, "compact", CATALOGUE)
    assert res.labels == [
        ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18"),  # main label stays first
        ResolvedLabel("G16H", "G16H 10/00", "G16H 10/60"),
        ResolvedLabel("A61B", None, None),
        ResolvedLabel("B65D", "B65D 51/00", None),
        ResolvedLabel("G06K", None, None),
    ]
    assert res.stats.as_dict() == {
        "exact": 2, "maingroup_only": 1, "subclass_only": 1, "ambiguous": 1, "missing": 1, "concordance": 0,
    }
    assert res.stats.total == 6  # each distinct raw label counted once


def test_resolve_labels_dedups_identical_resolutions_but_counts_both() -> None:
    # Two different raw codes both fall back to subclass H04L: one label, two counted outcomes.
    res = resolve_labels(["H04L2906", "H04L2908"], "compact", CATALOGUE)
    assert res.labels == [ResolvedLabel("H04L", None, None)]
    assert res.stats.subclass_only == 2


def test_resolve_labels_main_label_unresolvable_next_label_leads() -> None:
    res = resolve_labels(["H01L2100", "H10B1000"], "compact", CATALOGUE)
    assert res.labels == [ResolvedLabel("H10B", "H10B 10/00", "H10B 10/00")]
    assert res.stats.missing == 1 and res.stats.exact == 1


def test_resolve_labels_structured_forms() -> None:
    raw = [
        {"section": "G", "class": "16", "subclass": "H", "main_group": "10", "subgroup": "60"},
        {"ipc_section": "H", "ipc_class": "04", "ipc_subclass": "L", "ipc_main_group": "12", "ipc_subgroup": "18"},
        ("H", "04", "L", "12", "18"),  # duplicate of the previous label
        "H04L 12/18",  # same label again, printed
        ("H", 4, "L", "012", "1800"),  # same label again, padded
        "H04L 1/18",
        {"section": "H", "class": "01", "subclass": "L", "main_group": "21", "subgroup": "027"},
        42,  # unusable
    ]
    res = resolve_labels(raw, "structured", CATALOGUE)
    assert res.labels == [
        ResolvedLabel("G16H", "G16H 10/00", "G16H 10/60"),
        ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18"),
        ResolvedLabel("H04L", "H04L 1/00", "H04L 1/18"),
    ]
    assert res.stats.as_dict() == {"exact": 3, "maingroup_only": 0, "subclass_only": 0, "ambiguous": 0, "missing": 2,
                                   "concordance": 0}


def test_resolve_labels_empty() -> None:
    res = resolve_labels([], "compact", CATALOGUE)
    assert res.labels == [] and res.stats.total == 0


def test_resolve_labels_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        resolve_labels(["H04L1218"], "display", CATALOGUE)  # type: ignore[arg-type]


# --- concordance (TASK-0015 wiring, TASK-0005) --------------------------------------------------------


class FakeConcordance:
    """Records what it was asked; ``resolve`` answers from a table, ``apply`` must never be used."""

    def __init__(self, table: dict[str, ResolvedLabel | None]) -> None:
        self.table = table
        self.asked: list[str] = []

    def resolve(self, code: str) -> ResolvedLabel | None:
        self.asked.append(code)
        return self.table.get(code)

    def apply(self, code: str) -> list[str] | None:
        raise AssertionError("resolve_labels must not reduce apply() targets itself")


def test_concordance_remaps_only_missing_subclass_only_and_maingroup_only() -> None:
    conc = FakeConcordance({
        "H01L21027": ResolvedLabel("H10B", "H10B 10/00", None),  # MISSING (H01L abolished)
        "H04L2906": ResolvedLabel("H04L", "H04L 9/00", None),  # SUBCLASS_ONLY fallback
        "B65D5199": ResolvedLabel("B65D", "B65D 51/00", "B65D 51/00"),  # MAINGROUP_ONLY fallback
        "H04L1218": ResolvedLabel("G16H", None, None),  # EXACT: must not even be asked
        "A61B5022": ResolvedLabel("G16H", None, None),  # AMBIGUOUS: must not even be asked
    })  # fmt: skip
    raw = ["H04L1218", "H01L21027", "H04L2906", "B65D5199", "A61B5022", "G06K900"]
    res = resolve_labels(raw, "compact", CATALOGUE, concordance=conc)
    assert conc.asked == ["H01L21027", "H04L2906", "B65D5199", "G06K900"]
    assert res.labels == [
        ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18"),  # exact, untouched, still first
        ResolvedLabel("H10B", "H10B 10/00", None),
        ResolvedLabel("H04L", "H04L 9/00", None),
        ResolvedLabel("B65D", "B65D 51/00", "B65D 51/00"),
        ResolvedLabel("A61B", None, None),  # ambiguous, untouched
        ResolvedLabel("G06K", None, None),  # the concordance had nothing: the fallback stays
    ]
    assert res.stats.as_dict() == {"exact": 1, "maingroup_only": 0, "subclass_only": 1, "ambiguous": 1, "missing": 0,
                                   "concordance": 3}  # fmt: skip
    assert res.stats.total == 6


def test_concordance_none_leaves_a_missing_label_missing() -> None:
    res = resolve_labels(["H01L2171"], "compact", CATALOGUE, concordance=FakeConcordance({}))
    assert res.labels == [] and res.stats.missing == 1 and res.stats.concordance == 0


def test_concordance_reads_structured_labels_as_printed_symbols() -> None:
    conc = FakeConcordance({"H01L 21/027": ResolvedLabel("H10P", "H10P 76/00", None)})
    raw = [
        {"section": "H", "class": "01", "subclass": "L", "main_group": "0021", "subgroup": "0270"},
        ("H", "01", "L", "23", None),  # main group only
        ("H", "01", "L", None, None),  # subclass only
        ("H", "04", "L", "12", "18"),  # exact: not asked
    ]
    res = resolve_labels(raw, "structured", CATALOGUE, concordance=conc)
    assert conc.asked == ["H01L 21/027", "H01L 23/00", "H01L"]
    assert res.labels == [ResolvedLabel("H10P", "H10P 76/00", None), ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18")]
    assert res.stats.concordance == 1 and res.stats.missing == 2 and res.stats.exact == 1


def test_concordance_result_is_deduplicated_with_other_labels() -> None:
    conc = FakeConcordance({"H01L2100": ResolvedLabel("H10B", "H10B 10/00", "H10B 10/00")})
    res = resolve_labels(["H10B1000", "H01L2100"], "compact", CATALOGUE, concordance=conc)
    assert res.labels == [ResolvedLabel("H10B", "H10B 10/00", "H10B 10/00")]
    assert res.stats.exact == 1 and res.stats.concordance == 1


def test_catalogue_is_any_container() -> None:
    class Lookup:
        def __contains__(self, code: object) -> bool:
            return code in CATALOGUE

    assert resolve_compact("H04L1218", Lookup()) == ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18")


def test_stats_add() -> None:
    total = ResolutionStats()
    total.add(resolve_labels(["H04L1218", "H01L2100"], "compact", CATALOGUE).stats)
    total.add(resolve_labels(["A61B5022"], "compact", CATALOGUE).stats)
    assert total.as_dict() == {"exact": 1, "maingroup_only": 0, "subclass_only": 0, "ambiguous": 1, "missing": 1,
                               "concordance": 0}


# --- against the real 2026.01 catalogue, when the data file is checked out ------------------------


@pytest.fixture(scope="module")
def real_catalogue() -> frozenset[str]:
    if not CATALOGUE_FILE.exists() or CATALOGUE_FILE.stat().st_size < 1_000_000:  # absent or an LFS pointer
        pytest.skip("data/source/ipc_classification_index.jsonl not checked out")
    with CATALOGUE_FILE.open(encoding="utf-8") as fh:
        return frozenset(json.loads(line)["display_code"] for line in fh if line.strip())


@pytest.mark.parametrize(
    ("code", "outcome", "deepest"),
    [
        ("G16H1060", Outcome.EXACT, "G16H 10/60"),
        ("H04L1218", Outcome.EXACT, "H04L 12/18"),
        ("H04L100", Outcome.EXACT, "H04L 1/00"),
        ("H04L2906", Outcome.SUBCLASS_ONLY, "H04L"),
        ("G06K900", Outcome.SUBCLASS_ONLY, "G06K"),
        ("A61B5022", Outcome.AMBIGUOUS, "A61B"),
        ("H01L21027", Outcome.MISSING, None),
    ],
)
def test_samples_on_real_catalogue(real_catalogue: frozenset[str], code: str, outcome: Outcome, deepest: str | None) -> None:
    got_outcome, label = _compact_outcome(code, real_catalogue)
    assert got_outcome is outcome
    assert (label.deepest if label else None) == deepest


@pytest.fixture(scope="module")
def wipo(real_catalogue: frozenset[str]):
    from app.services.patents.concordance import load_concordance

    return load_concordance(SOURCE_DIR, real_catalogue)  # checked against the stored manifest's SHA-256


@pytest.mark.parametrize(
    ("code", "outcome", "label"),
    [
        # one-to-one: H01L 23/04 (deleted 2026.01) -> H10W 76/12
        ("H01L2304", Outcome.CONCORDANCE, ResolvedLabel("H10W", "H10W 76/00", "H10W 76/12")),
        # one-to-many in one main group: no exact code is invented
        ("H01L21027", Outcome.CONCORDANCE, ResolvedLabel("H10P", "H10P 76/00", None)),
        # targets in two subclasses: no common vote-level ancestor -> still missing
        ("H01L2171", Outcome.MISSING, None),
        # G06K 9/00 itself went to G06F 18/xx in 2023.01 (its SUBCLASS_ONLY fallback was G06K)
        ("G06K900", Outcome.CONCORDANCE, ResolvedLabel("G06F", "G06F 18/00", None)),
        # H04L 29/06 (2022.01) -> 48 targets in H04L: the subclass only
        ("H04L2906", Outcome.CONCORDANCE, ResolvedLabel("H04L", None, None)),
        # current symbols are untouched
        ("H04L1218", Outcome.EXACT, ResolvedLabel("H04L", "H04L 12/00", "H04L 12/18")),
        ("A61B5022", Outcome.AMBIGUOUS, ResolvedLabel("A61B", None, None)),
    ],
)
def test_real_wipo_lists(real_catalogue, wipo, code: str, outcome: Outcome, label: ResolvedLabel | None) -> None:
    res = resolve_labels([code], "compact", real_catalogue, concordance=wipo)
    assert res.labels == ([label] if label else [])
    assert getattr(res.stats, outcome.value) == 1 and res.stats.total == 1


def test_real_wipo_lists_structured(real_catalogue, wipo) -> None:
    res = resolve_labels([{"section": "H", "class": "01", "subclass": "L", "main_group": "23", "subgroup": "04"}],
                         "structured", real_catalogue, concordance=wipo)  # fmt: skip
    assert res.labels == [ResolvedLabel("H10W", "H10W 76/00", "H10W 76/12")] and res.stats.concordance == 1
