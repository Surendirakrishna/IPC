"""Offline A/B: rerank the same candidate lists with different scope-text variants.

Candidates are each document's stored `ranked` list (top TOPN) from a benchmark results.jsonl, so only the
reranker input changes. Reports hit@1/3/10 of recorded IPC at subclass / main group / exact.
usage: rerank_scope_ab.py RESULTS_JSONL DOCS_JSONL [TOPN] [OUT_JSONL]
"""
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from app.db import repository as repo
from app.db.session import api_session
from app.services.classification.reranker import get_reranker
from app.services.classification.scope import format_scope
from app.services.confidence.ipc_codes import parse_symbols
from app.services.evaluation import benchmark as bm

s = get_settings()
reranker = get_reranker(s)
results = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if x.strip()]
docs = {d["application_number"]: d for d in
        (json.loads(x) for x in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if x.strip())}
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 100
out = open(sys.argv[4], "w", encoding="utf-8") if len(sys.argv) > 4 else None
VARIANTS = {"full": (True, 6), "no_official": (False, 6), "path_only": (False, 0),
            "compact_own_notes": ("own", 0)}


def scope_text(rec, records, official, refs):
    if official == "own":  # title + path, then only this code's own WIPO notes, capped
        from app.services.classification.scope import _official_context_lines
        base = format_scope(rec, records, max_refs=0, official=False)
        own = [x for x in rec.official_context if (x.get("applies_to") or x["owner_symbol"]) == rec.code]
        saved, rec.official_context = rec.official_context, own
        try:
            extra = _official_context_lines(rec, max_chars=600)
        finally:
            rec.official_context = saved
        return "\n".join([base, *extra])
    return format_scope(rec, records, max_refs=refs, official=official)
KS = (1, 3, 10)
POLICY = bm.PolicyParams(s.min_reranker_score, s.max_reranker_gap, s.min_evidence_score,
                        enforce_lexical=s.enforce_lexical_checks)
seconds: dict[str, list[float]] = defaultdict(list)
assumed_ev: dict[str, int] = defaultdict(int)
hits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))


def keys(display: str) -> dict[str, str]:
    sub = display.replace(" ", "")[:4]
    mg = display.split("/")[0] + "/00" if "/" in display else None
    return {"subclass": sub, "maingroup": mg, "exact": display}


def hydrate(session, version, codes):
    records = repo.get_records(session, version, codes)
    anc = [a for rec in records.values() for a in rec.ancestor_codes if a not in records]
    records.update(repo.get_records(session, version, anc))
    refs = repo.get_references(session, version, codes)
    for code, rec in records.items():
        rec.references = refs.get(code, [])
    ctx = list(dict.fromkeys([*records, *(a for rec in records.values() for a in rec.ancestor_codes)]))
    official = repo.get_official_texts(session, version, ctx)
    for rec in records.values():
        seen, rec.official_context = set(), []
        for owner in [rec.code, *reversed(rec.ancestor_codes)]:
            for item in official.get(owner, []):
                key = (item["owner_symbol"], item["source_kind"], item["part_type"], item["ordinal"])
                if key not in seen:
                    seen.add(key)
                    rec.official_context.append(item)
    return records


with api_session() as session:
    version = repo.active_release(session).taxonomy_version
    for i, r in enumerate(results):
        if r.get("http") != 200 or not r.get("ranked"):
            continue
        d = docs[r["application_number"]]
        syms = parse_symbols(", ".join(d["ground_truth"]))
        truth = {"subclass": {x.subclass for x in syms},
                 "maingroup": {x.maingroup_display for x in syms if x.maingroup_display},
                 "exact": {x.display for x in syms}}
        # the pipeline reranks against views[0]: title + newline + abstract
        text = f"{d['title']}\n{d['abstract']}" if d.get("title") else d["abstract"]
        by_display = {}
        codes = []
        # ranked holds display codes; ranked_debug maps a prefix of them to internal codes
        dbg = {x["display_code"]: x["code"] for x in r.get("ranked_debug") or []}
        ev_by = {x["code"]: x["best_evidence_score"] for x in r.get("ranked_debug") or []}
        for disp in r["ranked"][:topn]:
            code = dbg.get(disp)
            if not code and (rec := repo.get_record_by_any_code(session, version, disp)):
                code = rec.code
            if code:
                codes.append(code)
                by_display[code] = disp
        records = hydrate(session, version, codes)
        codes = [c for c in codes if c in records]
        row = {"application_number": r["application_number"], "n_candidates": len(codes)}
        for name, (official, refs) in VARIANTS.items():
            pairs = [(scope_text(records[c], records, official, refs), text) for c in codes]
            t0 = time.perf_counter()
            scores = [p.score for p in reranker.score(pairs)]
            seconds[name].append(time.perf_counter() - t0)
            ranked_codes = [c for c, _ in sorted(zip(codes, scores), key=lambda kv: (-kv[1], kv[0]))]
            order = [by_display[c] for c in ranked_codes]
            row[name] = {"order": order[:20], "scores": {by_display[c]: round(sc, 4) for c, sc in zip(codes, scores)}}
            # Policy replay over the new top 20. Evidence scores are the recorded ones (scored with the full
            # scope); a candidate with none recorded is assumed to pass the evidence floor.
            sc_by = dict(zip(codes, scores))
            replay_rows = []
            for c in ranked_codes[:20]:
                old = ev_by.get(c)
                rec = records[c]
                replay_rows.append({"code": c, "display_code": by_display[c], "reranker_score": sc_by[c],
                                    "best_evidence_score": old if old is not None else 0.0,
                                    "record_type": rec.record_type, "maingroup_code": rec.maingroup_code,
                                    "application_context_only": False, "subclass_scope_ok": True,
                                    "defining_terms_ok": True, "_assumed": old is None})
            sel = bm.simulate_policy(replay_rows, POLICY)
            if sel and next(x for x in replay_rows if x["display_code"] == sel[0])["_assumed"]:
                assumed_ev[name] += 1
            for level in ("subclass", "maingroup", "exact"):
                if truth[level]:
                    hits[name][f"policy_top1_{level}"].append(
                        float(bool(sel) and keys(sel[0])[level] in truth[level]))
            for level in ("subclass", "maingroup", "exact"):
                if not truth[level]:
                    continue
                got = [keys(x)[level] for x in order]
                for k in KS:
                    hits[name][f"{level}@{k}"].append(float(bool(truth[level] & set(got[:k]))))
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"[{i + 1}/{len(results)}] {r['application_number']} {len(codes)} candidates", flush=True)

n_docs = len(next(iter(hits.values()))["subclass@1"])
print(f"\ndocuments={n_docs} candidates/doc<={topn}")
for name, by in hits.items():
    print(f"{name:18} " + " ".join(f"{k}={sum(v) / len(v):.3f}" for k, v in by.items())
          + f" median_rerank_s={statistics.median(seconds[name]):.2f} top1_on_assumed_evidence={assumed_ev[name]}")
