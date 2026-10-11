"""Patent-family grouping and family-safe data splits.

Two applications are put in the same family when they share any family key:
- a priority claim (country + number, normalised);
- a PCT application cited as the parent in the specification's cross-reference wording
  ("national phase of PCT/CN2025/070668", "claims priority to ...");
- near-identical title + abstract (word 5-shingle Jaccard >= NEAR_DUPLICATE_JACCARD), which catches
  divisionals and re-filings that carry no priority field.

Families are the unit of splitting: every member of a family lands in the same split, and
`split_conflicts` reports any family whose members already sit in different splits/files.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

NEAR_DUPLICATE_JACCARD = 0.8
SHINGLE = 5
PCT_RE = re.compile(r"PCT\s*/\s*([A-Z]{2})\s*(\d{4})\s*/\s*(\d{4,7})", re.I)
PCT_PARENT_CUE_RE = re.compile(
    r"national\s+phase|national\s+stage|continuation|divisional|claims?\s+(?:the\s+)?(?:priority|benefit)|"
    r"priority\s+(?:to|from|of)|benefit\s+of", re.I)
PCT_CUE_WINDOW = 200
SPEC_HEAD_CHARS = 3000


def _doc_id(doc: dict[str, Any]) -> str:
    return str(doc.get("document_id") or doc["application_number"])


def normalise_priority(number: str | None, country: str | None) -> str | None:
    if not number or not number.strip():
        return None
    c = re.sub(r"[^A-Z0-9]", "", (country or "").upper())
    n = re.sub(r"[^A-Z0-9]", "", number.upper())
    if c and n.startswith(c) and len(n) > len(c):  # IP India prefixes the country: "EPO20181776.4"
        n = n[len(c):]
    return f"priority:{c}:{n}" if n else None


def pct_parents(specification: str | None) -> set[str]:
    """PCT numbers cited as this application's parent near the start of the specification."""
    head = (specification or "")[:SPEC_HEAD_CHARS]
    keys = set()
    for m in PCT_RE.finditer(head):
        window = head[max(0, m.start() - PCT_CUE_WINDOW): m.end() + PCT_CUE_WINDOW]
        if PCT_PARENT_CUE_RE.search(window):
            keys.add(f"pct:{m.group(1).upper()}{m.group(2)}/{int(m.group(3)):06d}")
    return keys


def _shingles(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < SHINGLE:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1))


def family_keys(doc: dict[str, Any]) -> set[str]:
    keys = set()
    if doc.get("patent_family_id"):
        keys.add(f"declared:{doc['patent_family_id']}")
    if p := normalise_priority(doc.get("priority_number"), doc.get("priority_country")):
        keys.add(p)
    keys |= pct_parents(doc.get("specification"))
    return keys


@dataclass
class Family:
    family_id: str
    members: list[str]
    keys: list[str] = field(default_factory=list)
    near_duplicate_links: list[tuple[str, str, float]] = field(default_factory=list)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_families(docs: Iterable[dict[str, Any]], *,
                   near_duplicate_jaccard: float | None = NEAR_DUPLICATE_JACCARD) -> list[Family]:
    """Group documents into families. Pairwise near-duplicate search is O(n^2) on shingle sets with a
    length pre-filter; fine for evaluation sets of a few thousand documents."""
    docs = list(docs)
    uf = _UnionFind()
    by_key: dict[str, list[str]] = defaultdict(list)
    keys_of: dict[str, set[str]] = {}
    for d in docs:
        did = _doc_id(d)
        uf.find(did)
        keys_of[did] = family_keys(d)
        for k in keys_of[did]:
            by_key[k].append(did)
    for members in by_key.values():
        for other in members[1:]:
            uf.union(members[0], other)

    links: list[tuple[str, str, float]] = []
    if near_duplicate_jaccard is not None:
        sh = [(_doc_id(d), _shingles(f"{d.get('title') or ''} {d.get('abstract') or ''}")) for d in docs]
        for i, (a, sa) in enumerate(sh):
            for b, sb in sh[i + 1:]:
                if not sa or not sb or min(len(sa), len(sb)) < near_duplicate_jaccard * max(len(sa), len(sb)):
                    continue
                j = len(sa & sb) / len(sa | sb)
                if j >= near_duplicate_jaccard:
                    uf.union(a, b)
                    links.append((a, b, round(j, 3)))

    groups: dict[str, list[str]] = defaultdict(list)
    for did in keys_of:
        groups[uf.find(did)].append(did)
    families = []
    for members in groups.values():
        members.sort()
        keys = sorted({k for m in members for k in keys_of[m]})
        # A stable, readable id: the smallest family key if any, else the smallest member id.
        fid = "fam-" + (keys[0] if keys else members[0])
        mset = set(members)
        families.append(Family(fid, members, keys, [x for x in links if x[0] in mset]))
    return sorted(families, key=lambda f: f.family_id)


def assign_splits(families: list[Family], ratios: dict[str, float], *, seed: str,
                  pinned: dict[str, str] | None = None) -> dict[str, str]:
    """Map document id -> split. A family's split is drawn from a hash of (seed, family_id), so it is
    stable when unrelated documents are added. A family containing a pinned document takes that
    document's split (pinned conflicts are reported by `split_conflicts`, not resolved here)."""
    if not ratios or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    pinned = pinned or {}
    names = sorted(ratios)
    out: dict[str, str] = {}
    for fam in families:
        pins = sorted({pinned[m] for m in fam.members if m in pinned})
        if pins:
            split = pins[0]
        else:
            h = int(hashlib.sha256(f"{seed}:{fam.family_id}".encode()).hexdigest()[:12], 16) / 16 ** 12
            acc, split = 0.0, names[-1]
            for name in names:
                acc += ratios[name]
                if h < acc:
                    split = name
                    break
        for m in fam.members:
            out[m] = split
    return out


def split_conflicts(families: list[Family], assignment: dict[str, str]) -> list[dict[str, Any]]:
    """Families whose members carry more than one split label (leakage)."""
    conflicts = []
    for fam in families:
        splits = {assignment[m] for m in fam.members if m in assignment}
        if len(splits) > 1:
            conflicts.append({"family_id": fam.family_id, "keys": fam.keys,
                              "members": {m: assignment.get(m) for m in fam.members}})
    return conflicts
