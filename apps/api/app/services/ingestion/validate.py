"""Source validation: hierarchy consistency, reference closure, tree/index matching, counts.

Findings are reported, never silently repaired. Expected counts are compared only against a
snapshot manifest whose hashes match the supplied files; they are not universal IPC rules.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.services.ingestion.sources import IndexLine
from app.services.ingestion.tree import Occurrence

MAX_EXAMPLES = 25


@dataclass
class ValidationReport:
    counts: dict[str, int] = field(default_factory=dict)
    findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_quality: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)

    def add(self, key: str, examples: list[Any], *, blocking: bool = False, note: str | None = None) -> None:
        if not examples:
            return
        self.findings[key] = {
            "count": len(examples),
            "examples": examples[:MAX_EXAMPLES],
            "blocking": blocking,
            **({"note": note} if note else {}),
        }
        if blocking:
            self.blocking.append(key)

    @property
    def ok(self) -> bool:
        return not self.blocking

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": self.counts,
            "findings": self.findings,
            "blocking": self.blocking,
            "source_quality": self.source_quality,
        }


def validate_sources(
    lines: list[IndexLine],
    read_errors: list[dict[str, Any]],
    duplicate_codes: dict[str, list[int]],
    root_meta: dict[str, Any],
    occurrences: list[Occurrence],
    duplicates: list[dict[str, Any]],
    expected: dict[str, Any] | None,
    *,
    official_hierarchy: bool = False,
) -> ValidationReport:
    rep = ValidationReport()
    recs = {ln.record.code: ln.record for ln in lines}

    type_counts = Counter(r.type for r in recs.values())
    rep.counts = {
        "records": len(recs),
        **{f"type_{t}": type_counts.get(t, 0) for t in ("section", "class", "subclass", "maingroup", "subgroup")},
        "tree_occurrences": len(occurrences),
        "duplicate_maingroup_entries": len(duplicates),
        "read_errors": len(read_errors),
        "duplicate_canonical_codes": len(duplicate_codes),
    }

    rep.add("read_errors", read_errors, blocking=True)
    rep.add(
        "duplicate_canonical_codes",
        [{"code": c, "lines": ls} for c, ls in duplicate_codes.items()],
        blocking=True,
    )

    versions = Counter(r.taxonomy_version for r in recs.values())
    if len(versions) != 1:
        rep.add("mixed_taxonomy_versions", [dict(versions)], blocking=True)
    tree_version = root_meta.get("version")
    if versions and tree_version not in versions:
        rep.add("tree_version_mismatch", [{"tree": tree_version, "index": list(versions)}], blocking=True)

    # --- hierarchy -------------------------------------------------------
    missing_parent, parent_not_last_ancestor, bad_ancestor, child_mismatch = [], [], [], []
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for code, r in recs.items():
        if r.parent_code is not None:
            children_by_parent[r.parent_code].append(code)
            if r.parent_code not in recs:
                missing_parent.append({"code": code, "parent_code": r.parent_code})
            if not r.ancestor_codes or r.ancestor_codes[-1] != r.parent_code:
                parent_not_last_ancestor.append({"code": code, "parent_code": r.parent_code})
        elif r.depth != 1:
            missing_parent.append({"code": code, "parent_code": None, "depth": r.depth})
        if len(r.ancestor_codes) != r.depth - 1:
            bad_ancestor.append({"code": code, "depth": r.depth, "ancestors": r.ancestor_codes})
        for anc in r.ancestor_codes:
            if anc not in recs:
                bad_ancestor.append({"code": code, "missing_ancestor": anc})
    for code, r in recs.items():
        declared = set(r.child_codes)
        actual = set(children_by_parent.get(code, []))
        if declared != actual:
            child_mismatch.append(
                {
                    "code": code,
                    "declared_not_parented": sorted(declared - actual)[:10],
                    "parented_not_declared": sorted(actual - declared)[:10],
                }
            )
    rep.add("missing_parent", missing_parent, blocking=True)
    rep.add("parent_not_last_ancestor", parent_not_last_ancestor, blocking=True)
    rep.add("ancestor_inconsistency", bad_ancestor, blocking=True)
    rep.add("child_codes_mismatch", child_mismatch, blocking=True)

    # depth/type consistency (observed in this snapshot: subgroups flattened at depth 5)
    depth_by_type: dict[str, Counter] = defaultdict(Counter)
    for r in recs.values():
        depth_by_type[r.type][r.depth] += 1
    rep.counts["depth_by_type"] = {t: dict(c) for t, c in depth_by_type.items()}  # type: ignore[assignment]
    if set(depth_by_type.get("subgroup", {}).keys()) == {5}:
        if official_hierarchy:
            rep.source_quality.append(
                "The supplied comparison tree flattens subgroups; serving parent paths are replaced from the "
                "version-matched WIPO master scheme."
            )
        else:
            rep.source_quality.append(
                "All subgroups are flattened at depth 5 under a main group; deeper subgroup dot hierarchy is "
                "not present in these inputs and is not reconstructed."
            )

    # --- display codes -----------------------------------------------------
    by_display: dict[str, list[str]] = defaultdict(list)
    for code, r in recs.items():
        by_display[r.display_code].append(code)
    rep.add(
        "display_code_collisions",
        [{"display_code": d, "codes": cs} for d, cs in by_display.items() if len(cs) > 1],
        blocking=True,
    )

    # --- references --------------------------------------------------------
    dangling = []
    ref_total = 0
    for code, r in recs.items():
        for kind, targets in (("cross_reference", r.cross_references), ("precedence_reference", r.precedence_references)):
            for t in targets:
                ref_total += 1
                if t not in recs:
                    dangling.append({"source": code, "target": t, "kind": kind})
    rep.counts["reference_edges"] = ref_total
    rep.add(
        "dangling_references",
        dangling,
        note="Preserved with target_exists=false; never auto-assigned.",
    )

    # --- tree vs index -------------------------------------------------------
    occ_by_code: dict[str, list[Occurrence]] = defaultdict(list)
    for o in occurrences:
        occ_by_code[o.code].append(o)
    rep.counts["tree_distinct_symbols"] = len(occ_by_code)
    rep.counts["tree_repeated_symbols"] = sum(1 for v in occ_by_code.values() if len(v) > 1)
    rep.add("index_codes_missing_from_tree", sorted(set(recs) - set(occ_by_code)))
    rep.add("tree_symbols_missing_from_index", sorted(set(occ_by_code) - set(recs)))
    type_mismatch = [
        {"code": o.code, "tree_type": o.node_type, "index_type": recs[o.code].type}
        for o in occurrences
        if o.code in recs and o.node_type and o.node_type != recs[o.code].type
    ]
    rep.add("tree_type_mismatch", type_mismatch)

    # --- duplicates -------------------------------------------------------------
    dup_codes = {d.get("code") for d in duplicates}
    flagged = {c for c, r in recs.items() if r.is_duplicate_symbol}
    rep.add("duplicates_file_not_flagged_in_index", sorted(dup_codes - flagged))
    rep.add("index_flagged_not_in_duplicates_file", sorted(flagged - dup_codes))
    rep.add("duplicates_file_codes_missing_from_index", sorted(c for c in dup_codes if c not in recs))
    rep.add(
        "duplicate_symbols_single_tree_occurrence",
        sorted(c for c in dup_codes if len(occ_by_code.get(c, [])) < 2),
    )

    # --- root naming ------------------------------------------------------------
    root_title = str(root_meta.get("title", ""))
    if "CPC" in root_title:
        if official_hierarchy:
            rep.source_quality.append(
                f"The comparison tree root is titled {root_title!r}; the version-matched WIPO IPC master files "
                "supply the serving hierarchy."
            )
        else:
            rep.source_quality.append(
                f"Tree root is titled {root_title!r} (symbol {root_meta.get('symbol')!r}): naming inconsistency "
                "between IPC and CPC. This snapshot is not certified as a complete official IPC release."
            )

    # --- expected snapshot counts -----------------------------------------------
    if expected:
        diffs = []
        for key, want in expected.get("counts", {}).items():
            got = rep.counts.get(key)
            if got != want:
                diffs.append({"count": key, "expected": want, "actual": got})
        rep.add("snapshot_count_mismatch", diffs, blocking=True)

    return rep
