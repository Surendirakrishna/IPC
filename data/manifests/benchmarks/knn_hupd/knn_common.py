"""Shared helpers for the nearest-neighbour (kNN) vote experiment.

Scoring reuses app.services.evaluation.benchmark so the numbers are comparable with the live
benchmark reports. Catalogue presence comes from data/source/ipc_classification_index.jsonl (the
same 80,145 records the API serves), so nothing here needs the database.

Run with apps/api/.venv python, from the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "apps" / "api"))

from app.services.confidence.ipc_codes import ParsedSymbol, parse_symbols  # noqa: E402
from app.services.evaluation import benchmark as bm  # noqa: E402

EVAL = REPO / "data" / "evaluation"
DEV_DOCS = EVAL / "ipindia_2026-09.jsonl"
TEST_DOCS = EVAL / "ipindia_test_pool_2026-10.jsonl"
WORK = Path("D:/ipc_knn")  # large files live outside OneDrive


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def catalogue_codes() -> set[str]:
    out = set()
    with (REPO / "data" / "source" / "ipc_classification_index.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["type"] in ("subclass", "maingroup", "subgroup"):
                out.add(r["display_code"])
    return out


def query_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    return f"{title}. {doc['abstract'].strip()}" if title else doc["abstract"].strip()


def score_rows(docs: list[dict], preds: dict[str, dict], present: set[str]):
    """Mirror of cli._score_docs. preds[app_no] = {"supported": [...], "ranked": {level: [...]} or [...]}.

    A per-level ranked dict lets hit@k be measured on each level's own ordering.
    """
    rows, scored, errors = [], [], []
    for d in docs:
        p = preds.get(d["application_number"])
        if p is None:
            errors.append({"application_number": d["application_number"], "error": "no prediction"})
            continue
        syms = parse_symbols(", ".join(d["ground_truth"]))
        gt_sub = [s for s in syms if s.subclass in present]
        if not gt_sub:
            errors.append({"application_number": d["application_number"], "error": "no recorded symbol in catalogue"})
            continue
        ranked = p["ranked"]

        def rk(level: str) -> list[str]:
            return ranked[level] if isinstance(ranked, dict) else ranked

        row = bm.score_document(bm.GroundTruth(symbols=gt_sub), p["supported"], rk("subclass"), None)
        mg = [s for s in syms if s.maingroup_display in present]
        row["maingroup"] = (bm.score_document(bm.GroundTruth(symbols=mg), p["supported"], rk("maingroup"), None)
                            ["maingroup"] if mg else None)
        ex = [s for s in syms if s.display in present]
        row["exact"] = (bm.score_document(bm.GroundTruth(symbols=ex), p["supported"], rk("exact"), None)["exact"]
                        if ex else None)
        row["application_number"] = d["application_number"]
        rows.append(row)
        scored.append(d)
    return rows, scored, errors


def headline(rows: list[dict]) -> dict:
    agg = bm.aggregate([{k: v for k, v in r.items() if k != "application_number"} for r in rows])
    out = {"documents": agg["documents"]}
    for level in bm.LEVELS:
        a = agg.get(level)
        if a:
            out[level] = {"top1": a["top1_all_documents"]["rate"], "ci95": a["top1_all_documents"]["ci95"],
                          **{f"hit@{k}": a[f"candidate_hit@{k}"]["rate"] for k in (3, 5, 10)}}
    return out


__all__ = ["ParsedSymbol", "parse_symbols", "bm", "REPO", "WORK", "DEV_DOCS", "TEST_DOCS",
           "load_jsonl", "catalogue_codes", "query_text", "score_rows", "headline"]
