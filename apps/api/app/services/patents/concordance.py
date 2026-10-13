"""Map abolished IPC symbols forward with WIPO's revision concordance lists (TASK-0015, ADR-0005).

WIPO publishes one concordance list per IPC edition (``ipc_concordancelist_YYYYMMDD.xml`` in the edition's
MasterFiles): each ``<concordance from-symbol=... modification="d|c">`` lists the ``<concordance-to
to-symbol=... default-reclassification="Y|N">`` places its content went to in that edition. Only deleted
(``d``) symbols need mapping; a changed (``c``) symbol still exists. The lists stored under ``data/source/``
(``wipo_ipc_concordance_YYYYMMDD.zip``, 2017.01 to 2026.01, URLs and SHA-256 in
``wipo_ipc_concordance_manifest.json``) are chained in edition order, so a symbol deleted in 2023 whose
target was deleted again in 2026 ends at the 2026 targets.

Symbols in the lists are 14-character internal codes (``H01L0021027000`` = H01L 21/027); everything this
module returns is a catalogue display code ("H01L 21/027", "H10W 76/00", "H10B").

One-to-many transfers (about half of the H01L labels in the 2017 HUPD corpus) must not invent precision:
``apply`` returns every current target, and ``common_ancestor`` reduces them to the deepest level they all
share that exists in the catalogue (one target: that code; same main group: the main group with no exact
code; same subclass: the subclass only). Targets in several subclasses have no common ancestor at a vote
level and give None (they keep voting for nothing, as before). WIPO's ``default-reclassification`` flag is
read but not used to pick one target: it is a backlog-reclassification default, not evidence about a
particular document.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Container, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from app.services.patents.labels import ResolvedLabel

NS = "{http://www.wipo.int/classifications/ipc/masterfiles}"
FILE_GLOB = "wipo_ipc_concordance_*.zip"
MANIFEST_NAME = "wipo_ipc_concordance_manifest.json"

INTERNAL_RE = re.compile(r"^(?P<sub>[A-H]\d{2}[A-Z])(?P<mg>\d{4})(?P<sg>\d{6})$")
DISPLAY_RE = re.compile(r"^(?P<sub>[A-H]\d{2}[A-Z])(?:\s*(?P<mg>\d{1,4})\s*/\s*(?P<sg>\d{2,}))?$")
COMPACT_RE = re.compile(r"^(?P<sub>[A-H]\d{2}[A-Z])(?P<digits>\d{3,})$")
MAX_MAINGROUP_DIGITS = 4
MIN_SUBGROUP_DIGITS = 2


class ConcordanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transfer:
    """One deletion in one edition: ``from_symbol`` went to ``targets`` (display codes)."""

    edition: str  # to-version, e.g. "20260101"
    from_symbol: str
    targets: tuple[str, ...]
    defaults: tuple[str, ...]  # targets flagged default-reclassification="Y"


def internal_to_display(symbol: str) -> str:
    """'H01L0021027000' -> 'H01L 21/027'; 'H10W0076000000' -> 'H10W 76/00'; a subclass-only code -> 'H10W'."""
    m = INTERNAL_RE.match(symbol.strip().upper())
    if m is None:
        raise ConcordanceError(f"not a 14-character IPC symbol: {symbol!r}")
    mg, sg = int(m.group("mg")), m.group("sg").rstrip("0").ljust(MIN_SUBGROUP_DIGITS, "0")
    if mg == 0 and sg == "00":
        return m.group("sub")
    return f"{m.group('sub')} {mg}/{sg}"


def normalise_display(code: str) -> str | None:
    """'H01L 21/027', 'h01l21/027 ' -> 'H01L 21/027'; a 14-character code is converted; else None."""
    text = code.strip().upper()
    if INTERNAL_RE.match(text):
        return internal_to_display(text)
    m = DISPLAY_RE.match(text) if "/" in text or len(text) == 4 else None
    if m is None:
        return None
    if m.group("mg") is None:
        return m.group("sub")
    return f"{m.group('sub')} {int(m.group('mg'))}/{m.group('sg')}"


def read_concordance_xml(data: bytes) -> tuple[str, str, list[Transfer]]:
    """(from_version, to_version, deletions) of one WIPO concordance list."""
    root = ET.fromstring(data)
    if root.tag != f"{NS}IPCConcordanceList":
        raise ConcordanceError(f"not a WIPO IPC concordance list (root element {root.tag!r})")
    to_version = root.get("to-version") or ""
    transfers = []
    for entry in root.findall(f"{NS}concordance"):
        if entry.get("modification") != "d":
            continue
        targets, defaults = [], []
        for to in entry.findall(f"{NS}concordance-to"):
            display = internal_to_display(to.get("to-symbol", ""))
            if display not in targets:
                targets.append(display)
            if to.get("default-reclassification") == "Y" and display not in defaults:
                defaults.append(display)
        transfers.append(Transfer(to_version, internal_to_display(entry.get("from-symbol", "")), tuple(targets),
                                  tuple(defaults)))  # fmt: skip
    return root.get("from-version") or "", to_version, transfers


def _read_file(path: Path) -> bytes:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if len(names) != 1:
                raise ConcordanceError(f"{path} should hold exactly one XML file, has {names}")
            return z.read(names[0])
    return path.read_bytes()


def _verify_against_manifest(files: Sequence[Path], directory: Path) -> None:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {f["file"]: f["sha256"] for f in manifest.get("files", [])}
    for path in files:
        want = expected.get(path.name)
        if want is None:
            raise ConcordanceError(f"{path.name} is not listed in {MANIFEST_NAME}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise ConcordanceError(f"{path.name} has SHA-256 {got}, the manifest records {want}")


class WipoConcordance(Mapping[str, list[str]]):
    """Old display code -> current display codes, chained over the loaded editions.

    As a mapping it holds every symbol deleted in any loaded edition and its final targets (unfiltered).
    ``apply`` (the ``labels.Concordance`` protocol) takes a raw label in any form and returns only targets
    that exist in ``catalogue``.
    """

    def __init__(self, editions: Sequence[tuple[str, Sequence[Transfer]]], catalogue: Container[str] | None = None,
                 sources: Sequence[str] = ()) -> None:  # fmt: skip
        self.editions = sorted(((ed, {t.from_symbol: t for t in transfers}) for ed, transfers in editions),
                               key=lambda e: e[0])  # fmt: skip
        self.catalogue = catalogue
        self.sources = tuple(sources)
        self._old: set[str] = {s for _, by_symbol in self.editions for s in by_symbol}
        self._final: dict[str, list[str]] = {s: self._chain(s, 0) for s in sorted(self._old)}

    # --- Mapping -------------------------------------------------------------------------------------------
    def __getitem__(self, code: str) -> list[str]:
        return list(self._final[code])

    def __iter__(self) -> Iterator[str]:
        return iter(self._final)

    def __len__(self) -> int:
        return len(self._final)

    # --- chaining ---------------------------------------------------------------------------------------------
    def _chain(self, symbol: str, start: int) -> list[str]:
        for i in range(start, len(self.editions)):
            transfer = self.editions[i][1].get(symbol)
            if transfer is not None:
                out: list[str] = []
                for target in transfer.targets:
                    for final in self._chain(target, i + 1):
                        if final not in out:
                            out.append(final)
                return out
        return [symbol]

    def transfers(self, code: str) -> list[Transfer]:
        """The deletions ``code`` went through, in edition order (first hop only per edition)."""
        return [by_symbol[code] for _, by_symbol in self.editions if code in by_symbol]

    # --- lookups ---------------------------------------------------------------------------------------------
    def _current(self, code: str) -> bool:
        return self.catalogue is not None and code in self.catalogue

    def old_symbol(self, code: str) -> str | None:
        """The deleted symbol a raw label denotes, or None.

        Accepts a display code ("H01L 21/027"), a 14-character code or a compact HUPD code ("H01L21027").
        A compact code is split the way ``labels.resolve_compact`` splits it (subgroup at least 2 digits),
        but against the deleted symbols; a split that names a current catalogue symbol wins, so a label that
        resolves normally is never remapped (WIPO occasionally reuses a deleted symbol later).
        """
        text = code.strip().upper()
        display = normalise_display(text)
        if display is not None:
            if self._current(display):
                return None
            return display if display in self._old else None
        m = COMPACT_RE.match(text)
        if m is None:
            return None
        sub, digits = m.group("sub"), m.group("digits")
        last = min(MAX_MAINGROUP_DIGITS, len(digits) - MIN_SUBGROUP_DIGITS)
        splits = [f"{sub} {int(digits[:i])}/{digits[i:]}" for i in range(1, last + 1)]
        if any(self._current(s) for s in splits):
            return None
        hits = list(dict.fromkeys(s for s in splits if s in self._old))
        if len(hits) == 1:
            return hits[0]
        if not hits:  # the label names a subgroup that was never listed, but its main group was deleted
            mgs = list(dict.fromkeys(f"{sub} {int(digits[:i])}/00" for i in range(1, last + 1)))
            mg_hits = [g for g in mgs if g in self._old and not self._current(g)]
            if len(mg_hits) == 1:
                return mg_hits[0]
        return None

    def apply(self, code: str) -> list[str] | None:
        """Current symbols an abolished label's content now sits in, or None.

        None when the label is not an abolished symbol (or still exists), or when none of its final targets
        is in the catalogue. Without a catalogue every final target is returned unchecked.
        """
        old = self.old_symbol(code)
        if old is None:
            return None
        targets = [t for t in self._final[old] if self.catalogue is None or t in self.catalogue]
        return targets or None

    def resolve(self, code: str) -> ResolvedLabel | None:
        """``common_ancestor(apply(code))``: what an abolished label should vote for, or None."""
        targets = self.apply(code)
        return common_ancestor(targets, self.catalogue) if targets else None


def common_ancestor(targets: Sequence[str], catalogue: Container[str] | None = None) -> ResolvedLabel | None:
    """The deepest level all ``targets`` share, as a ``ResolvedLabel``; None if they span several subclasses.

    One target: that code (a main-group target is its own exact code, as labels.resolve_compact does for
    "H04L100"). Same main group: the main group, exact None. Same subclass: the subclass only. A level that
    is not in ``catalogue`` is dropped (None), like the resolver's fallbacks.
    """
    unique = list(dict.fromkeys(targets))
    if not unique:
        return None

    def present(code: str) -> bool:
        return catalogue is None or code in catalogue

    subs = {t[:4] for t in unique}
    if len(subs) != 1:
        return None
    sub = subs.pop()
    if not present(sub):
        return None
    mains = {t.split("/")[0] for t in unique if "/" in t}
    if len(unique) == 1 and "/" in unique[0]:
        target = unique[0]
        mg = f"{target.split('/')[0]}/00"
        return ResolvedLabel(sub, mg if present(mg) else None, target if present(target) else None)
    if len(mains) == 1 and all("/" in t for t in unique):
        mg = f"{mains.pop()}/00"
        return ResolvedLabel(sub, mg if present(mg) else None, None)
    return ResolvedLabel(sub, None, None)


def load_concordance(path: str | Path, catalogue: Container[str] | None = None) -> WipoConcordance:
    """Load WIPO concordance lists: a directory (every ``wipo_ipc_concordance_*.zip`` in it, checked against
    ``wipo_ipc_concordance_manifest.json`` when present), one ``.zip`` or one ``.xml``."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob(FILE_GLOB))
        if not files:
            raise ConcordanceError(f"no {FILE_GLOB} in {p}")
        _verify_against_manifest(files, p)
    elif p.is_file():
        files = [p]
    else:
        raise ConcordanceError(f"{p} not found")
    editions = []
    for f in files:
        _, to_version, transfers = read_concordance_xml(_read_file(f))
        if not to_version:
            raise ConcordanceError(f"{f.name} has no to-version")
        editions.append((to_version, transfers))
    versions = [e for e, _ in editions]
    if len(set(versions)) != len(versions):
        raise ConcordanceError(f"duplicate editions in {p}: {versions}")
    return WipoConcordance(editions, catalogue, sources=[f.name for f in files])
