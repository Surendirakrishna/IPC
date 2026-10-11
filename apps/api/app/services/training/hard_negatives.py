"""Reranker training groups from expert-reviewed labels, with hard negatives near each accepted code.

For each reviewed document the group holds the accepted codes (positives) and negatives of these
kinds, strongest first:
- explicit_reject: the reviewer rejected the code;
- ranked_confusable: the pipeline ranked the code highly but it is not accepted. Used only when the
  review is "complete", because under a partial review an unjudged code may be correct;
- sibling / same_subclass / same_class: codes next to an accepted code in the scheme (same parent,
  other main groups of its subclass, other subclasses of its class). These are *assumed* negatives
  under a partial review and are flagged as such.

Never a negative: an accepted code, an ancestor or descendant of one, or any code in `exclude`
(for example codes a reviewer marked "unsure"). Scope texts are built with the same formatter the
pipeline reranks with, so training and inference inputs match; each group records the formatter
version (`scope_formatter`) and must be regenerated when it changes. Indexing codes (WIPO entry_type
'I') are excluded on both sides.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

NEGATIVE_KINDS = ("explicit_reject", "ranked_confusable", "sibling", "same_subclass", "same_class")
DEFAULT_PER_KIND = {"explicit_reject": 20, "ranked_confusable": 8, "sibling": 4, "same_subclass": 4,
                    "same_class": 2}


@dataclass
class Node:
    code: str
    record_type: str
    parent_code: str | None
    subclass_code: str | None
    class_code: str | None
    ancestor_codes: list[str] = field(default_factory=list)


class Catalogue(Protocol):
    def nodes(self, codes: list[str]) -> dict[str, Node]: ...
    def children(self, parent_codes: list[str]) -> dict[str, list[str]]: ...
    def scope_texts(self, codes: list[str]) -> dict[str, str]: ...


def _rng(*parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:16], 16))


def _related(code: str, accepted: dict[str, Node], nodes: dict[str, Node]) -> bool:
    """True when `code` is an accepted code, or an ancestor or descendant of one."""
    if code in accepted:
        return True
    node = nodes.get(code)
    anc = set(node.ancestor_codes) if node else set()
    return any(code in a.ancestor_codes or a.code in anc for a in accepted.values())


def mine_group(
    doc: dict[str, Any],
    catalogue: Catalogue,
    *,
    ranked_codes: list[str] | None = None,
    exclude: Iterable[str] = (),
    per_kind: dict[str, int] | None = None,
    seed: str = "negatives-v1",
) -> dict[str, Any] | None:
    """One training group for an expert-labelled document, or None when nothing was accepted."""
    per_kind = {**DEFAULT_PER_KIND, **(per_kind or {})}
    doc_id = str(doc["document_id"])
    accepts = list(dict.fromkeys(lb["code"] for lb in doc["labels"] if lb["judgement"] == "accept"))
    rejects = list(dict.fromkeys(lb["code"] for lb in doc["labels"] if lb["judgement"] == "reject"))
    complete = doc.get("judgement_completeness") == "complete"
    accepted = catalogue.nodes(accepts)
    if not accepted:
        return None
    excluded = set(exclude)

    # Scheme neighbourhoods of every accepted code.
    parents = {n.parent_code for n in accepted.values() if n.parent_code}
    subclasses = {n.subclass_code for n in accepted.values() if n.subclass_code}
    classes = {n.class_code for n in accepted.values() if n.class_code}
    kids = catalogue.children(sorted(parents | subclasses | classes))
    candidates: dict[str, list[str]] = {
        "explicit_reject": rejects,
        "ranked_confusable": list(ranked_codes or []) if complete else [],
        "sibling": [c for p in sorted(parents) for c in kids.get(p, [])],
        "same_subclass": [c for s in sorted(subclasses) for c in kids.get(s, [])],
        "same_class": [c for k in sorted(classes) for c in kids.get(k, [])],
    }
    all_codes = sorted({c for cs in candidates.values() for c in cs})
    nodes = {**catalogue.nodes(all_codes), **accepted}

    negatives: list[dict[str, Any]] = []
    taken: set[str] = set()
    for kind in NEGATIVE_KINDS:
        pool = [c for c in dict.fromkeys(candidates[kind])
                if c not in taken and c not in excluded and c in nodes and not _related(c, accepted, nodes)]
        if kind not in ("explicit_reject", "ranked_confusable"):  # scheme neighbours: sample, don't take the first
            _rng(seed, doc_id, kind).shuffle(pool)
        for c in pool[: per_kind[kind]]:
            taken.add(c)
            negatives.append({"code": c, "kind": kind,
                              "assumed": kind not in ("explicit_reject",) and not complete})

    texts = catalogue.scope_texts([*accepted, *(n["code"] for n in negatives)])
    return {
        "document_id": doc_id,
        "patent_family_id": doc.get("patent_family_id"),
        "split": doc["split"],
        "text": doc["abstract"],
        "judgement_completeness": doc.get("judgement_completeness"),
        # Groups are only valid for the scope formatter they were built with; regenerate when it changes.
        "scope_formatter": getattr(catalogue, "scope_version", None),
        "positives": [{"code": c, "scope": texts[c]} for c in accepted if c in texts],
        "negatives": [{**n, "scope": texts[n["code"]]} for n in negatives if n["code"] in texts],
    }


def to_flag_reranker_rows(group: dict[str, Any], *, orientation: str = "scope_first") -> list[dict[str, Any]]:
    """FlagEmbedding reranker fine-tuning rows ({"query", "pos", "neg"}); one row per positive.

    FlagEmbedding pairs (query, passage). The served pipeline puts the candidate scope first, so
    "scope_first" makes each positive scope the query, with the abstract as its positive passage and
    the abstract paired with nothing else; that orientation cannot express code-vs-code negatives, so
    "document_first" (abstract as query, scopes as passages) is the one that uses the hard negatives.
    Train and serve with the same orientation."""
    if orientation == "document_first":
        neg = [n["scope"] for n in group["negatives"]]
        return [{"query": group["text"], "pos": [p["scope"]], "neg": neg,
                 "document_id": group["document_id"], "split": group["split"]} for p in group["positives"]]
    if orientation == "scope_first":
        return [{"query": p["scope"], "pos": [group["text"]], "neg": [],
                 "document_id": group["document_id"], "split": group["split"]} for p in group["positives"]]
    raise ValueError(f"unknown orientation {orientation!r}")


def classification_codes(session: Any, version: str, codes: list[str]) -> set[str]:
    """The subset of `codes` that are classification places (WIPO entry_type 'K'); indexing codes
    (entry_type 'I') are valid symbols but never a positive or a negative."""
    from sqlalchemy import text

    if not codes:
        return set()
    rows = session.execute(text(
        "SELECT code FROM ipc_official_scheme_entries"
        " WHERE taxonomy_version = :v AND entry_type = 'K' AND code = ANY(:c)"),
        {"v": version, "c": list(dict.fromkeys(codes))}).all()
    return {r[0] for r in rows}


class DbCatalogue:
    """Catalogue backed by the active release (classification entries only; indexing codes excluded)."""

    def __init__(self, session: Any, version: str, *, official_context: bool) -> None:
        from app.services.classification.scope import SCOPE_FORMATTER_VERSION

        self.session, self.version, self.official_context = session, version, official_context
        self.scope_version = f"{SCOPE_FORMATTER_VERSION}:official={official_context}"

    def nodes(self, codes: list[str]) -> dict[str, Node]:
        from app.db import repository as repo

        keep = classification_codes(self.session, self.version, codes)
        recs = repo.get_records(self.session, self.version, [c for c in codes if c in keep])
        return {c: Node(c, r.record_type, r.parent_code, r.subclass_code, r.class_code, list(r.ancestor_codes))
                for c, r in recs.items()}

    def children(self, parent_codes: list[str]) -> dict[str, list[str]]:
        from app.db import repository as repo

        return repo.get_siblings(self.session, self.version, parent_codes)

    def scope_texts(self, codes: list[str]) -> dict[str, str]:
        from app.db import repository as repo
        from app.services.classification.scope import format_scope

        records = repo.get_records(self.session, self.version, codes)
        anc = [a for r in records.values() for a in r.ancestor_codes if a not in records]
        records.update(repo.get_records(self.session, self.version, anc))
        refs = repo.get_references(self.session, self.version, codes)
        for code, rec in records.items():
            rec.references = refs.get(code, [])
        if self.official_context:
            ctx = list(dict.fromkeys([*records, *(a for r in records.values() for a in r.ancestor_codes)]))
            official = repo.get_official_texts(self.session, self.version, ctx)
            for rec in records.values():
                seen, rec.official_context = set(), []
                for owner in [rec.code, *reversed(rec.ancestor_codes)]:
                    for item in official.get(owner, []):
                        key = (item["owner_symbol"], item["source_kind"], item["part_type"], item["ordinal"])
                        if key not in seen:
                            seen.add(key)
                            rec.official_context.append(item)
        return {c: format_scope(records[c], records, official=self.official_context) for c in codes if c in records}
