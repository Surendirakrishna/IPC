"""Traverse the original schema tree, preserving every occurrence and its wording."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PAREN_RE = re.compile(r"\s*\([^()]*\)")
WS_RE = re.compile(r"\s+")


@dataclass
class Occurrence:
    occurrence_id: int
    code: str
    node_type: str | None
    source_path: list[str]
    original_title: str
    direct_child_symbols: list[str]
    parent_symbol: str | None
    sibling_index: int
    context: dict[str, Any] = field(default_factory=dict)


def traverse(tree: dict[str, Any]) -> tuple[dict[str, Any], list[Occurrence]]:
    """Depth-first, document-order traversal. Returns (root metadata, occurrences).

    occurrence_id is the pre-order index, so identical code/path combinations remain distinct.
    The root node ("IPC") is metadata, not an occurrence.
    """
    root_meta = {k: v for k, v in tree.items() if k != "children"}
    occurrences: list[Occurrence] = []
    # stack of (node, path_of_ancestor_symbols, sibling_index, parent_symbol)
    stack: list[tuple[dict[str, Any], list[str], int, str | None]] = [
        (child, [], idx, None) for idx, child in reversed(list(enumerate(tree.get("children", []))))
    ]
    while stack:
        node, path, sibling_index, parent = stack.pop()
        symbol = node.get("symbol")
        children = node.get("children") or []
        occ = Occurrence(
            occurrence_id=len(occurrences),
            code=symbol,
            node_type=node.get("type"),
            source_path=[*path, symbol],
            original_title=node.get("title") or "",
            direct_child_symbols=[c.get("symbol") for c in children],
            parent_symbol=parent,
            sibling_index=sibling_index,
            context={k: v for k, v in node.items() if k not in ("symbol", "title", "type", "children")},
        )
        occurrences.append(occ)
        for idx, child in reversed(list(enumerate(children))):
            stack.append((child, occ.source_path, idx, symbol))
    return root_meta, occurrences


def normalize_title(title: str) -> str:
    """Comparison key only (never stored as wording): strip parentheticals and collapse spaces."""
    prev = None
    out = title
    while prev != out:
        prev, out = out, PAREN_RE.sub("", out)
    return WS_RE.sub(" ", out).strip().casefold()


def select_canonical_occurrence(
    record_title: str, canonical_title: str | None, occs: list[Occurrence]
) -> tuple[Occurrence | None, str]:
    """Pick the tree occurrence whose wording corresponds to the record's canonical title.

    Deterministic order of preference; returns (occurrence, reason).
    """
    if not occs:
        return None, "no_tree_occurrence"
    if len(occs) == 1:
        return occs[0], "single_occurrence"
    targets = [t for t in (canonical_title, record_title) if t]
    for target in targets:
        key = normalize_title(target)
        for occ in occs:
            if normalize_title(occ.original_title) == key:
                return occ, "normalized_title_match"
    for target in targets:
        key = normalize_title(target)
        for occ in occs:
            if normalize_title(occ.original_title).startswith(key):
                return occ, "normalized_title_prefix"
    return occs[-1], "fallback_last_occurrence"


REF_CODE_RE = re.compile(r"\b([A-H]\d{2}[A-Z]?\d{0,10})\b")


def reference_wording(original_title: str, target_code: str) -> str | None:
    """Return the parenthetical clause in which a target code is cited, verbatim, if present.

    Only extracts the supplied wording; it never infers a condition or exception.
    """
    for paren in re.finditer(r"\(([^()]*)\)", original_title):
        for segment in paren.group(1).split(";"):
            if re.search(rf"(?<![A-Z0-9]){re.escape(target_code)}(?![A-Z0-9])", segment):
                return segment.strip()
    return None
