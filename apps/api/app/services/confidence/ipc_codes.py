"""Parse human-written IPC symbols (as printed on patent documents) into catalogue lookups.

Documents print symbols like "G06N 3/08", "G06N3/08", "G06N0003080000", "H04L 9/0838" or with
version suffixes "G06N 3/08 (2006.01)". The catalogue (this snapshot) keys records by the zero-padded
internal code and a display code; it has no subgroups deeper than one dot level in its hierarchy,
but its display codes keep the source digits (e.g. "H04L 9/0838" if present).
Matching uses the catalogue's own display_code values, never an invented padding algorithm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SYMBOL_RE = re.compile(
    r"(?P<sub>[A-H]\d{2}[A-Z])\s*[-:]?\s*(?:(?P<mg>\d{1,4})\s*/\s*(?P<sg>\d{1,6}))?(?!\d)"
)
INTERNAL_RE = re.compile(r"\b([A-H]\d{2}[A-Z]\d{10})\b")


@dataclass(frozen=True)
class ParsedSymbol:
    raw: str
    subclass: str
    main_group: str | None  # "3" in G06N 3/08
    subgroup: str | None  # "08" in G06N 3/08

    @property
    def display(self) -> str:
        if self.main_group is None:
            return self.subclass
        return f"{self.subclass} {int(self.main_group)}/{self.subgroup}"

    @property
    def maingroup_display(self) -> str | None:
        return None if self.main_group is None else f"{self.subclass} {int(self.main_group)}/00"


def parse_symbols(text: str) -> list[ParsedSymbol]:
    """Extract IPC symbols from a free-text classification field, in order, de-duplicated."""
    out: list[ParsedSymbol] = []
    seen: set[str] = set()
    for m in INTERNAL_RE.finditer(text):
        code = m.group(1)
        sub, mg, sg = code[:4], str(int(code[4:8])), code[8:14].rstrip("0")
        sg = sg.ljust(2, "0") if sg else "00"
        ps = ParsedSymbol(code, sub, mg, sg)
        if ps.display not in seen:
            seen.add(ps.display)
            out.append(ps)
    cleaned = INTERNAL_RE.sub(" ", text)
    for m in SYMBOL_RE.finditer(cleaned.upper()):
        sub, mg, sg = m.group("sub"), m.group("mg"), m.group("sg")
        ps = ParsedSymbol(m.group(0).strip(), sub, mg, sg)
        if ps.display not in seen:
            seen.add(ps.display)
            out.append(ps)
    return out
