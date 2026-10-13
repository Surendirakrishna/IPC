"""Nearest-neighbour IPC vote: tune on the IP India dev set, score the held-out test set once.

For a query, the K most similar corpus applications vote for their IPC codes. Neighbour n has weight
exp((s_n - s_1) / T); its main IPC label counts 1 and each secondary label BETA. A code's score is
the weighted share of neighbours carrying it (at each level a neighbour counts once, with its best
label weight), so it lies in [0, 1] and reads as "share of similar patents that carry this code".

The first code is chosen top-down: best subclass, then the best main group inside it, then the best
subgroup inside that. Corpus labels absent from the 2026.01 catalogue vote at the deepest level that
exists (e.g. G06K 9/00 -> G06K).

Usage: python knn_eval.py paecter bge_m3
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from knn_common import (DEV_DOCS, TEST_DOCS, WORK, catalogue_codes, headline, load_jsonl,
                        score_rows)

HERE = Path(__file__).resolve().parent
KS = (5, 10, 20, 30, 50, 100, 200)
TS = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1.0)
BETAS = (0.0, 0.25, 0.5, 1.0)
K_MAX = max(KS)
PARTIAL = int(os.environ.get("KNN_PARTIAL_SHARDS", "0"))  # >0: early look on the first N shards, writes nothing
BASELINE_TEST = {"subclass": 0.309, "maingroup": 0.179, "exact": 0.089}  # wipo2026_test__abstract, n=123


SPLIT_STATS: dict[str, int] = defaultdict(int)


def split_compact(code: str, present: set[str]) -> tuple[str, str | None, str | None] | None:
    """'H04L1218' -> ('H04L', 'H04L 12/00', 'H04L 12/18'), choosing the main group/subgroup split
    (subgroup >= 2 digits) whose symbol exists in the catalogue. A code matching no symbol falls back
    to a unique existing main group, then to the subclass; a code matching several symbols is
    ambiguous and votes only for its subclass."""
    sub, digits = code[:4], code[4:]
    if sub not in present:
        SPLIT_STATS["subclass_missing"] += 1
        return None
    splits = [(str(int(digits[:i])), digits[i:]) for i in range(1, min(4, len(digits) - 2) + 1)]
    exact = [(mg, sg) for mg, sg in splits if f"{sub} {mg}/{sg}" in present]
    if len(exact) == 1:
        mg, sg = exact[0]
        SPLIT_STATS["exact"] += 1
        return sub, f"{sub} {mg}/00", f"{sub} {mg}/{sg}"
    if len(exact) > 1:
        SPLIT_STATS["ambiguous"] += 1
        return sub, None, None
    mgs = {mg for mg, _ in splits if f"{sub} {mg}/00" in present}
    if len(mgs) == 1:
        SPLIT_STATS["maingroup_only"] += 1
        return sub, f"{sub} {mgs.pop()}/00", None
    SPLIT_STATS["subclass_only"] += 1
    return sub, None, None


def resolve_labels(corpus: list[dict], present: set[str]) -> list[list[tuple[str, str | None, str | None]]]:
    """Per corpus doc: [(subclass, main group or None, exact or None)] in label order (main label first)."""
    out = []
    for r in corpus:
        labs = [x for x in (split_compact(c, present) for c in r["labels"]) if x is not None]
        out.append(labs)
    return out


def truth_codes(labs: list[tuple[str, str | None, str | None]]) -> list[str]:
    """Hold-out ground truth as display codes at the deepest resolved level."""
    return list(dict.fromkeys(ex or mg or sub for sub, mg, ex in labs))


def search(corpus_emb, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    q = torch.from_numpy(queries.astype(np.float16)).cuda()
    sims, idx = [], []
    for i in range(0, len(q), 256):
        s, ix = torch.topk(q[i:i + 256] @ corpus_emb.T, K_MAX, dim=1)
        sims.append(s.float().cpu().numpy())
        idx.append(ix.cpu().numpy())
    return np.concatenate(sims), np.concatenate(idx)


def vote(sims: np.ndarray, idx: np.ndarray, labels, k: int, t: float, beta: float) -> dict:
    w = np.exp((sims[:k].astype(np.float64) - float(sims[0])) / t).tolist()
    total = sum(w)
    score = {lvl: defaultdict(float) for lvl in ("subclass", "maingroup", "exact")}
    for wn, n in zip(w, idx[:k]):
        best: dict[str, dict[str, float]] = {lvl: {} for lvl in score}
        for j, (sub, mg, ex) in enumerate(labels[n]):
            lw = 1.0 if j == 0 else beta
            if lw == 0:
                continue
            for lvl, key in (("subclass", sub), ("maingroup", mg), ("exact", ex)):
                if key and lw > best[lvl].get(key, 0.0):
                    best[lvl][key] = lw
        for lvl in score:
            for key, lw in best[lvl].items():
                score[lvl][key] += wn * lw
    ranked = {lvl: sorted(score[lvl], key=score[lvl].get, reverse=True) for lvl in score}
    share = {lvl: {c: round(score[lvl][c] / total, 4) for c in ranked[lvl][:10]} for lvl in score}
    if not ranked["subclass"]:
        return {"supported": [], "ranked": ranked, "probability": {}}
    sub = ranked["subclass"][0]
    mg = next((c for c in ranked["maingroup"] if c.startswith(sub + " ")), None)
    ex = next((c for c in ranked["exact"] if mg and c.split("/")[0] == mg.split("/")[0]), None) if mg else None
    first = ex or mg or sub
    prob = {"subclass": score["subclass"][sub] / total,
            "maingroup": score["maingroup"][mg] / total if mg else None,
            "exact": score["exact"][ex] / total if ex else None}
    return {"supported": [first], "ranked": ranked, "probability": prob, "top_share": share}


def predict(docs, sims, idx, labels, k, t, beta) -> dict[str, dict]:
    return {d["application_number"]: vote(sims[i], idx[i], labels, k, t, beta) for i, d in enumerate(docs)}


def mean_top1(h: dict) -> float:
    return sum(h[lvl]["top1"] for lvl in ("subclass", "maingroup", "exact")) / 3


def reliability(rows: list[dict], preds: dict[str, dict], level: str) -> list[dict]:
    """Vote share of the chosen code vs how often it matched, in five bins."""
    bins = [[] for _ in range(5)]
    for r in rows:
        p = preds[r["application_number"]]["probability"].get(level)
        if p is None or r[level] is None:
            continue
        bins[min(4, int(p * 5))].append((p, r[level]["top1"]))
    return [{"bin": f"{b / 5:.1f}-{(b + 1) / 5:.1f}", "n": len(x),
             "mean_vote_share": round(sum(p for p, _ in x) / len(x), 3) if x else None,
             "observed_match_rate": round(sum(c for _, c in x) / len(x), 3) if x else None}
            for b, x in enumerate(bins)]


def run(model: str, corpus: list[dict], labels, present: set[str]) -> dict:
    import torch

    d = WORK / "emb" / model
    shards = sorted(d.glob("corpus_*.npy"))
    if PARTIAL:
        shards = shards[:PARTIAL]
        n = sum(np.load(p, mmap_mode="r").shape[0] for p in shards)
        corpus, labels = corpus[:n], labels[:n]
    corpus_emb = torch.from_numpy(np.concatenate([np.load(p) for p in shards])).cuda()
    assert corpus_emb.shape[0] == len(corpus), (corpus_emb.shape, len(corpus))
    holdout = load_jsonl(WORK / "holdout.jsonl")
    hold_emb = np.concatenate([np.load(p) for p in sorted(d.glob("holdout_*.npy"))])
    splits = {"dev": (load_jsonl(DEV_DOCS), np.load(d / "query_dev.npy")),
              "test": (load_jsonl(TEST_DOCS), np.load(d / "query_test.npy")),
              "holdout": ([{"application_number": r["id"], "ground_truth": truth_codes(labs)}
                           for r, labs in zip(holdout, resolve_labels(holdout, present), strict=True)], hold_emb)}
    found = {s: search(corpus_emb, q) for s, (_, q) in splits.items()}

    grid = []
    dev_docs = splits["dev"][0]
    for k, t, beta in itertools.product(KS, TS, BETAS):
        rows, *_ = score_rows(dev_docs, predict(dev_docs, *found["dev"], labels, k, t, beta), present)
        h = headline(rows)
        grid.append({"k": k, "T": t, "beta": beta, "mean_top1": round(mean_top1(h), 4),
                     **{lvl: h[lvl]["top1"] for lvl in ("subclass", "maingroup", "exact")}})
    grid.sort(key=lambda g: (-g["mean_top1"], g["k"]))
    best = grid[0]
    k, t, beta = best["k"], best["T"], best["beta"]

    out = {"model": json.loads((d / "meta.json").read_text()), "chosen_on_dev": best, "dev_grid_top10": grid[:10]}
    for split in ("dev", "test", "holdout"):
        docs, _ = splits[split]
        preds = predict(docs, *found[split], labels, k, t, beta)
        rows, _, errors = score_rows(docs, preds, present)
        out[split] = {"headline": headline(rows), "skipped": len(errors),
                      "reliability_subclass": reliability(rows, preds, "subclass"),
                      "reliability_maingroup": reliability(rows, preds, "maingroup"),
                      "nearest_similarity": {q: round(float(np.quantile(found[split][0][:, 0], q)), 3)
                                             for q in (0.5, 0.9, 0.99, 1.0)}}
        if split == "test" and not PARTIAL:
            with (HERE / f"results_test_{model}.jsonl").open("w", encoding="utf-8") as fh:
                for i, dd in enumerate(docs):
                    p = preds[dd["application_number"]]
                    fh.write(json.dumps({"application_number": dd["application_number"],
                                         "truth": dd["ground_truth"], "first": p["supported"],
                                         "probability": p["probability"],
                                         "top_subclasses": p.get("top_share", {}).get("subclass"),
                                         "nearest": [corpus[j]["id"] for j in found[split][1][i][:5]],
                                         "nearest_similarity": [round(float(x), 3) for x in found[split][0][i][:5]]},
                                        ensure_ascii=False) + "\n")
    del corpus_emb
    torch.cuda.empty_cache()
    return out


def main(models: list[str]) -> None:
    corpus = load_jsonl(WORK / "corpus.jsonl")
    present = catalogue_codes()
    labels = resolve_labels(corpus, present)
    report = {"corpus": json.loads((WORK / "corpus_meta.json").read_text()),
              "corpus_label_resolution": dict(SPLIT_STATS),
              "baseline_current_pipeline_test_top1": BASELINE_TEST, "models": {}}
    for m in models:
        report["models"][m] = run(m, corpus, labels, present)
        print(m, json.dumps(report["models"][m]["test"]["headline"]), flush=True)
    if PARTIAL:
        print(json.dumps({m: {"chosen_on_dev": r["chosen_on_dev"], **{s: r[s]["headline"] for s in ("dev", "test", "holdout")}}
                          for m, r in report["models"].items()}, indent=1))
        return
    path = HERE / "report.json"
    if path.exists():  # keep results of models not re-run
        old = json.loads(path.read_text(encoding="utf-8"))
        report["models"] = {**old.get("models", {}), **report["models"]}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["paecter", "bge_m3"])
