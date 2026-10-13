"""Resolve a patent source's IPC labels against the active taxonomy catalogue.

Every resolved label is a ``ResolvedLabel(subclass, main_group, exact)`` of catalogue display codes
("H04L", "H04L 12/00", "H04L 12/18"); ``main_group`` and ``exact`` are None when that level cannot be
determined or no longer exists. Every emitted code exists in the catalogue passed in.

Two source schemes:

- ``compact`` (HUPD): "H04L1218" carries no main group / subgroup separator, so the split is chosen
  by catalogue membership (port of ``split_compact`` in data/manifests/benchmarks/knn_hupd/knn_eval.py).
- ``structured`` (USPTO / PatentsView): section, class, subclass, main group and subgroup arrive as
  separate fields, so resolution is a normalisation.

Labels whose symbol no longer exists (H01L, G06K 9/00, H04L 29/06 in 2026.01) fall back to the
deepest level that still exists. With a WIPO concordance (``concordance.load_concordance``, TASK-0015),
a label that resolved MISSING, SUBCLASS_ONLY or MAINGROUP_ONLY is first mapped forward with
``concordance.resolve(raw)``: the common ancestor of all its current targets (never more precise than they
agree). If that gives a label, it replaces the fallback and counts as ``concordance``. EXACT and AMBIGUOUS
labels name a current symbol (or several) and are never remapped.
"""

from __future__ import annotations

import re
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal, NamedTuple, Protocol

from app.services.confidence.ipc_codes import parse_symbols

Catalogue = Container[str]
"""Anything answering ``display_code in catalogue``: a set of catalogue display codes in tests, or a
DB-backed lookup in production."""

Scheme = Literal["compact", "structured"]

SUBCLASS_RE = re.compile(r"^[A-H]\d{2}[A-Z]$")
COMPACT_RE = re.compile(r"^(?P<sub>[A-H]\d{2}[A-Z])(?P<digits>\d*)$")
MAX_MAINGROUP_DIGITS = 4
MIN_SUBGROUP_DIGITS = 2


class ResolvedLabel(NamedTuple):
    subclass: str
    main_group: str | None
    exact: str | None

    @property
    def deepest(self) -> str:
        return self.exact or self.main_group or self.subclass


class Outcome(StrEnum):
    EXACT = "exact"
    MAINGROUP_ONLY = "maingroup_only"
    SUBCLASS_ONLY = "subclass_only"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"  # subclass not in the catalogue, or the label could not be parsed
    CONCORDANCE = "concordance"  # an abolished symbol, re-homed with the WIPO concordance


# Outcomes a concordance may improve: the symbol (or its subclass) is not current. EXACT and AMBIGUOUS name
# current symbols and are never remapped.
REMAPPABLE = (Outcome.MISSING, Outcome.SUBCLASS_ONLY, Outcome.MAINGROUP_ONLY)


class Concordance(Protocol):
    """Abolished symbol -> current symbols (``concordance.WipoConcordance``, TASK-0015).

    ``resolve`` is what a label votes for: the common ancestor of all its final targets, or None. ``apply``
    only lists the in-catalogue targets; reducing those would invent precision, so resolution never uses it.
    """

    def apply(self, code: str) -> list[str] | None: ...

    def resolve(self, code: str) -> ResolvedLabel | None: ...


@dataclass
class ResolutionStats:
    exact: int = 0
    maingroup_only: int = 0
    subclass_only: int = 0
    ambiguous: int = 0
    missing: int = 0
    concordance: int = 0

    def count(self, outcome: Outcome) -> None:
        setattr(self, outcome.value, getattr(self, outcome.value) + 1)

    def add(self, other: ResolutionStats) -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)

    @property
    def total(self) -> int:
        return sum(asdict(self).values())

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class LabelResolution:
    labels: list[ResolvedLabel] = field(default_factory=list)  # label order, main label first
    stats: ResolutionStats = field(default_factory=ResolutionStats)


def _maingroup(sub: str, mg: str, catalogue: Catalogue) -> str | None:
    code = f"{sub} {mg}/00"
    return code if code in catalogue else None


def _fallback(sub: str, maingroups: set[str], catalogue: Catalogue) -> tuple[Outcome, ResolvedLabel]:
    """No exact symbol: a unique existing main group, else the subclass only."""
    existing = {code for mg in maingroups if (code := _maingroup(sub, mg, catalogue)) is not None}
    if len(existing) == 1:
        return Outcome.MAINGROUP_ONLY, ResolvedLabel(sub, existing.pop(), None)
    return Outcome.SUBCLASS_ONLY, ResolvedLabel(sub, None, None)


def _choose(
    sub: str, candidates: Sequence[tuple[str, str]], catalogue: Catalogue
) -> tuple[Outcome, ResolvedLabel | None]:
    """Shared rule for both schemes, given the subclass and candidate (main group, subgroup) pairs."""
    if sub not in catalogue:
        return Outcome.MISSING, None
    exact = list(dict.fromkeys((mg, sg) for mg, sg in candidates if f"{sub} {mg}/{sg}" in catalogue))
    if len(exact) == 1:
        mg, sg = exact[0]
        return Outcome.EXACT, ResolvedLabel(sub, _maingroup(sub, mg, catalogue), f"{sub} {mg}/{sg}")
    if len(exact) > 1:
        return Outcome.AMBIGUOUS, ResolvedLabel(sub, None, None)
    return _fallback(sub, {mg for mg, _ in candidates}, catalogue)


def _compact_outcome(code: str, catalogue: Catalogue) -> tuple[Outcome, ResolvedLabel | None]:
    m = COMPACT_RE.match(code.strip().upper())
    if m is None:
        return Outcome.MISSING, None
    sub, digits = m.group("sub"), m.group("digits")
    last = min(MAX_MAINGROUP_DIGITS, len(digits) - MIN_SUBGROUP_DIGITS)
    splits = [(str(int(digits[:i])), digits[i:]) for i in range(1, last + 1)]
    return _choose(sub, splits, catalogue)


def resolve_compact(code: str, catalogue: Catalogue) -> ResolvedLabel | None:
    """'H04L1218' -> ResolvedLabel('H04L', 'H04L 12/00', 'H04L 12/18').

    Tries every main group / subgroup split (subgroup at least 2 digits). A unique split whose symbol
    exists wins; with none, a unique existing main group; otherwise the subclass only. A code matching
    several symbols is ambiguous and keeps only its subclass. A subclass absent from the catalogue (or
    an unparseable code) gives None.
    """
    return _compact_outcome(code, catalogue)[1]


def _text(value: object) -> str:
    """Source fields may arrive as str, int or None."""
    return "" if value is None else str(value).strip().upper()


def _is_digits(value: str) -> bool:
    return value.isascii() and value.isdigit()


def _subclass(section: str | int | None, class_: str | int | None, subclass: str | None) -> str | None:
    sub = _text(subclass)
    if SUBCLASS_RE.match(sub):  # a source that already gives "H04L" as the subclass
        return sub
    sec, cls = _text(section), _text(class_)
    if _is_digits(cls):
        cls = cls.zfill(2)
    full = f"{sec}{cls}{sub}"
    return full if SUBCLASS_RE.match(full) else None


def _subgroup_candidates(subgroup: str) -> list[str]:
    """IPC subgroup digits read as a decimal fraction: trailing zeros beyond two digits carry no
    meaning ("0800" == "08"). A single digit is ambiguous between "d0" and "0d", so both are tried."""
    canonical = subgroup.rstrip("0").ljust(MIN_SUBGROUP_DIGITS, "0")
    if len(subgroup) == 1:
        return list(dict.fromkeys([canonical, subgroup.zfill(MIN_SUBGROUP_DIGITS)]))
    return [canonical]


def _structured_outcome(
    section: str | int | None,
    class_: str | int | None,
    subclass: str | None,
    main_group: str | int | None,
    subgroup: str | int | None,
    catalogue: Catalogue,
) -> tuple[Outcome, ResolvedLabel | None]:
    sub = _subclass(section, class_, subclass)
    if sub is None:
        return Outcome.MISSING, None
    mg, sg = _text(main_group), _text(subgroup)
    if not _is_digits(mg):
        return _choose(sub, [], catalogue)
    mg = str(int(mg))
    if not _is_digits(sg):  # main group known, subgroup not: never invent one
        if sub not in catalogue:
            return Outcome.MISSING, None
        return _fallback(sub, {mg}, catalogue)
    return _choose(sub, [(mg, s) for s in _subgroup_candidates(sg)], catalogue)


def resolve_structured(
    section: str | int | None,
    class_: str | int | None,
    subclass: str | None,
    main_group: str | int | None,
    subgroup: str | int | None,
    catalogue: Catalogue,
) -> ResolvedLabel | None:
    """('H', '04', 'L', '12', '18') -> ResolvedLabel('H04L', 'H04L 12/00', 'H04L 12/18').

    ``subclass`` may be the letter alone (with ``section`` and ``class_``) or the full "H04L". Leading
    zeros in the main group and insignificant trailing zeros in the subgroup are normalised. Same
    fallbacks as ``resolve_compact``: exact, else the main group, else the subclass; a missing
    subclass gives None.
    """
    return _structured_outcome(section, class_, subclass, main_group, subgroup, catalogue)[1]


STRUCTURED_KEYS = ("section", "class", "subclass", "main_group", "subgroup")
STRUCTURED_ALIASES = {"class": ("class", "class_", "ipc_class"), "section": ("section", "ipc_section"),
                      "subclass": ("subclass", "ipc_subclass"), "main_group": ("main_group", "ipc_main_group"),
                      "subgroup": ("subgroup", "ipc_subgroup")}


StructuredFields = tuple[Any, Any, Any, Any, Any]  # section, class, subclass, main group, subgroup


def _structured_fields(raw: Any) -> StructuredFields | None:
    if isinstance(raw, Mapping):
        sec, cls, sub, mg, sg = (next((raw[a] for a in STRUCTURED_ALIASES[k] if a in raw), None) for k in STRUCTURED_KEYS)
        return sec, cls, sub, mg, sg
    if isinstance(raw, str):  # a printed symbol such as "H04L 12/18"
        parsed = parse_symbols(raw)
        if len(parsed) != 1:
            return None
        p = parsed[0]
        return None, None, p.subclass, p.main_group, p.subgroup
    if isinstance(raw, Sequence) and len(raw) == len(STRUCTURED_KEYS):
        sec, cls, sub, mg, sg = raw
        return sec, cls, sub, mg, sg
    return None


def _label_key(raw: Any, scheme: Scheme) -> Any:
    """Identity of a raw label for de-duplication, after the same normalisation resolution applies."""
    if scheme == "compact":
        return str(raw).strip().upper()
    fields = _structured_fields(raw)
    if fields is None:
        return ("unparsed", repr(raw))
    sec, cls, sub, mg, sg = fields
    mg_text, sg_text = _text(mg), _text(sg)
    return (
        _subclass(sec, cls, sub),
        str(int(mg_text)) if _is_digits(mg_text) else mg_text,
        sg_text.rstrip("0").ljust(MIN_SUBGROUP_DIGITS, "0") if len(sg_text) > 1 and _is_digits(sg_text) else sg_text,
    )


def _within_catalogue(mapped: ResolvedLabel, catalogue: Catalogue) -> tuple[Outcome, ResolvedLabel | None]:
    """A concordance result limited to ``catalogue``: the concordance filters against its *own* catalogue
    (possibly none, or another release), and every label emitted here must exist in the caller's catalogue.

    Levels absent from the catalogue are dropped; an exact code is kept only under a kept main group (never
    deeper than a missing level). A mapped subclass that is not in the catalogue makes the label ``missing``.
    """
    if mapped.subclass not in catalogue:
        return Outcome.MISSING, None
    mg = mapped.main_group if mapped.main_group is not None and mapped.main_group in catalogue else None
    exact = mapped.exact if mg is not None and mapped.exact is not None and mapped.exact in catalogue else None
    return Outcome.CONCORDANCE, ResolvedLabel(mapped.subclass, mg, exact)


def _printed_symbol(fields: StructuredFields) -> str | None:
    """A structured label as the printed symbol the concordance reads ("H01L 21/027", "H01L 21/00", "H01L")."""
    sec, cls, sub, mg, sg = fields
    subclass = _subclass(sec, cls, sub)
    if subclass is None:
        return None
    mg_text, sg_text = _text(mg), _text(sg)
    if not _is_digits(mg_text):
        return subclass
    if not _is_digits(sg_text):
        return f"{subclass} {int(mg_text)}/00"
    return f"{subclass} {int(mg_text)}/{_subgroup_candidates(sg_text)[0]}"


def resolve_labels(
    raw_labels: Iterable[Any],
    scheme: Scheme,
    catalogue: Catalogue,
    concordance: Concordance | None = None,
) -> LabelResolution:
    """Resolve one document's labels, keeping their order (main label first) and dropping duplicates.

    ``compact`` labels are strings ("H04L1218"). ``structured`` labels are mappings with keys
    section / class / subclass / main_group / subgroup (``ipc_``-prefixed keys accepted), 5-tuples
    in that order, or a single printed symbol ("H04L 12/18"). Statistics count each distinct raw
    label once; labels that resolve to nothing are counted as ``missing`` and left out.

    With ``concordance``, a MISSING / SUBCLASS_ONLY / MAINGROUP_ONLY label is replaced by
    ``concordance.resolve(raw)`` when that is not None, and counted as ``concordance`` instead. The mapped label is
    cut to the levels present in ``catalogue``; if its subclass is absent, the label counts as ``missing`` and is
    dropped (the fallback label too: the symbol is known to be abolished).
    """
    if scheme not in ("compact", "structured"):
        raise ValueError(f"unknown label scheme {scheme!r}; expected 'compact' or 'structured'")
    result = LabelResolution()
    seen_raw: set[Any] = set()
    seen_labels: set[ResolvedLabel] = set()
    for raw in raw_labels:
        key = _label_key(raw, scheme)
        if key in seen_raw:
            continue
        seen_raw.add(key)
        if scheme == "compact":
            outcome, label = _compact_outcome(str(raw), catalogue)
            symbol: str | None = str(raw)
        else:
            fields = _structured_fields(raw)
            outcome, label = (Outcome.MISSING, None) if fields is None else _structured_outcome(*fields, catalogue)
            symbol = None if fields is None else _printed_symbol(fields)
        if concordance is not None and outcome in REMAPPABLE and symbol is not None:
            mapped = concordance.resolve(symbol)
            if mapped is not None:
                outcome, label = _within_catalogue(ResolvedLabel(*mapped), catalogue)
        result.stats.count(outcome)
        if label is not None and label not in seen_labels:
            seen_labels.add(label)
            result.labels.append(label)
    return result
