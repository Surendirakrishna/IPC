"""Benchmark scoring against documents' recorded IPC symbols (e.g. IP India published applications).

Ground truth caveat: IPC symbols on published Indian applications are usually those declared at
filing (often an older IPC version, possibly revised later by the examiner). Results therefore
measure agreement with the recorded IPC, not examiner-verified accuracy. Recorded symbols that are
absent from the active catalogue are reported and scored at the deepest level that does exist.

Levels: subclass (A01B), main group (A01B 3/00), exact symbol (A01B 3/42 or the main group itself).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.services.confidence.ipc_codes import ParsedSymbol

LEVELS = ("subclass", "maingroup", "exact")
ROUTING_CUTOFFS = (1, 5, 10)


@dataclass
class GroundTruth:
    symbols: list[ParsedSymbol]
    catalogue_exact: set[str] = field(default_factory=set)  # display codes found in the catalogue
    missing_from_catalogue: list[str] = field(default_factory=list)

    def keys(self, level: str) -> set[str]:
        if level == "subclass":
            return {s.subclass for s in self.symbols}
        if level == "maingroup":
            return {s.maingroup_display for s in self.symbols if s.maingroup_display}
        return {s.display for s in self.symbols if s.main_group is not None}


def pred_key(display_code: str, level: str) -> str:
    """display_code like 'H04L 9/08' or 'H04L 9/00' -> key at level."""
    sub = display_code[:4]
    if level == "subclass":
        return sub
    rest = display_code[4:].strip()
    if "/" not in rest:
        return display_code
    mg = rest.split("/")[0]
    return f"{sub} {int(mg)}/00" if level == "maingroup" else display_code


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def score_document(
    gt: GroundTruth,
    supported: list[str],
    ranked: list[str],
    field_class: str | None,
    retrieval_debug: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """supported: display codes in the summary (may be empty = abstention).
    ranked: supported + alternatives ordered by reranker score (candidate-generation quality)."""
    out: dict[str, Any] = {"abstained": not supported}
    stages = retrieval_debug or {}
    route_truth = {s.subclass for s in gt.symbols}
    route_codes = stages.get("stage1_subclasses")

    def route_recall(k: int | None) -> float | None:
        if not route_truth or route_codes is None:
            return None
        return len(route_truth & set(route_codes[:k] if k else route_codes)) / len(route_truth)
    for level in LEVELS:
        truth = gt.keys(level)
        if not truth:
            out[level] = None
            continue
        sup = [pred_key(c, level) for c in supported]
        rk = list(dict.fromkeys(pred_key(c, level) for c in ranked))
        stage2_codes = stages.get("stage2_group_candidates")
        reranked_codes = stages.get("reranked_candidates")
        stage2 = list(dict.fromkeys(pred_key(c, level) for c in (stage2_codes or [])))
        reranked = list(dict.fromkeys(pred_key(c, level) for c in (reranked_codes or [])))
        tp = len(set(sup) & truth)
        prec = tp / len(set(sup)) if sup else 0.0
        rec = tp / len(truth)
        out[level] = {
            "f1": (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
            "top1": bool(sup) and sup[0] in truth,
            "any_supported": tp > 0,
            "precision": (tp / len(set(sup))) if sup else None,
            "recall": tp / len(truth),
            "first_truth_in_supported": gt_first_key(gt, level) in set(sup),
            **{f"hit@{k}": bool(set(rk[:k]) & truth) for k in (1, 3, 5, 10, 100)},
            "stage_recall": {
                "subclass_routing": route_recall(None),
                **{f"subclass_routing@{k}": route_recall(k) for k in ROUTING_CUTOFFS},
                "group_candidate_generation": (len(set(stage2) & truth) / len(truth)
                                               if stage2_codes is not None else None),
                "reranker_top100": (len(set(reranked[:100]) & truth) / len(truth)
                                    if reranked_codes is not None else None),
            },
        }
    first = gt.symbols[0] if gt.symbols else None
    out["field_class_match"] = (field_class == first.subclass[:3]) if (first and field_class) else None
    return out


def gt_first_key(gt: GroundTruth, level: str) -> str | None:
    if not gt.symbols:
        return None
    s = gt.symbols[0]
    return {"subclass": s.subclass, "maingroup": s.maingroup_display, "exact": s.display}[level]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    agg: dict[str, Any] = {"documents": n, "abstained": rate(sum(r["abstained"] for r in rows), n)}
    answered = [r for r in rows if not r["abstained"]]
    for level in LEVELS:
        rs = [r[level] for r in rows if r[level] is not None]
        ans = [r[level] for r in answered if r[level] is not None]
        if not rs:
            continue
        precs = [x["precision"] for x in ans if x["precision"] is not None]
        agg[level] = {
            "documents_with_truth_at_level": len(rs),
            "top1_all_documents": rate(sum(x["top1"] for x in rs), len(rs)),
            "top1_answered_only": rate(sum(x["top1"] for x in ans), len(ans)),
            "any_supported_all_documents": rate(sum(x["any_supported"] for x in rs), len(rs)),
            "first_recorded_symbol_in_supported": rate(sum(x["first_truth_in_supported"] for x in rs), len(rs)),
            "mean_precision_answered": round(sum(precs) / len(precs), 3) if precs else None,
            "mean_recall_all_documents": round(sum(x["recall"] for x in rs) / len(rs), 3),
            "mean_f1_all_documents": round(sum(x["f1"] for x in rs) / len(rs), 3),
            **{f"candidate_hit@{k}": rate(sum(x[f"hit@{k}"] for x in rs), len(rs))
               for k in (1, 3, 5, 10, 100)},
            "stage_recall": {
                stage: {
                    "mean_recall": round(sum(x["stage_recall"][stage] for x in rs
                                              if x["stage_recall"][stage] is not None)
                                           / max(1, sum(x["stage_recall"][stage] is not None for x in rs)), 3)
                    if any(x["stage_recall"][stage] is not None for x in rs) else None,
                    "any_hit": rate(sum(bool(x["stage_recall"][stage]) for x in rs
                                         if x["stage_recall"][stage] is not None),
                                    sum(x["stage_recall"][stage] is not None for x in rs)),
                }
                for stage in ("subclass_routing", *(f"subclass_routing@{k}" for k in ROUTING_CUTOFFS),
                              "group_candidate_generation", "reranker_top100")
            },
        }
    fc = [r["field_class_match"] for r in rows if r["field_class_match"] is not None]
    agg["field_of_invention_class_match"] = rate(sum(fc), len(fc))
    return agg


def section_breakdown(docs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {}
    for d, r in zip(docs, rows, strict=True):
        sec = d["ground_truth"][0][0] if d.get("ground_truth") else "?"
        by.setdefault(sec, []).append(r)
    out = {}
    for sec, rs in sorted(by.items()):
        sub = [x["subclass"] for x in rs if x["subclass"]]
        out[sec] = {"documents": len(rs),
                    "subclass_top1": rate(sum(x["top1"] for x in sub), len(sub)),
                    "subclass_candidate_hit@10": rate(sum(x["hit@10"] for x in sub), len(sub))}
    return out


@dataclass(frozen=True)
class PolicyParams:
    min_reranker_score: float | None  # None = no absolute floor
    max_reranker_gap: float
    min_evidence_score: float | None  # None = any extracted passage suffices
    max_results: int = 5
    enforce_lexical: bool = True


def simulate_policy(ranked: list[dict[str, Any]], p: PolicyParams,
                    rule_demoted: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Replay pipeline._apply_policy on recorded ranked_debug rows (same rule order).

    Pass 1 content checks, pass 2 relevance gap from the best passing candidate, then hierarchy
    (drop a main group whose own subgroup is selected). WIPO priority rules need the rule text, so they
    are replayed only as recorded: codes in rule_demoted (from a live run's outcomes) are dropped.
    Supported candidates sharing a main group are all kept, as in the live policy.
    """
    passed = []
    for c in ranked:
        if p.min_reranker_score is not None and c["reranker_score"] < p.min_reranker_score:
            continue
        ev = c["best_evidence_score"]
        if ev is None or (p.min_evidence_score is not None and ev < p.min_evidence_score):
            continue
        if p.enforce_lexical and (c["application_context_only"] or not c["subclass_scope_ok"]
                                  or not c["defining_terms_ok"]):
            continue
        passed.append(c)
    if not passed:
        return []
    best = max(c["reranker_score"] for c in passed)
    eligible = [c for c in passed if best - c["reranker_score"] <= p.max_reranker_gap]
    supported = [c for c in eligible if not (c["record_type"] == "maingroup" and any(
        o["record_type"] == "subgroup" and o["maingroup_code"] == c["code"] for o in eligible))]
    return [c["display_code"] for c in supported if c["code"] not in rule_demoted][: p.max_results]


def rule_demoted_codes(ranked: list[dict[str, Any]]) -> set[str]:
    """Codes a live run moved to alternatives because of a WIPO priority rule."""
    return {c["code"] for c in ranked if str(c.get("outcome", "")).startswith("alternative: WIPO")}


def status_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r.get("status", "error") for r in results))
