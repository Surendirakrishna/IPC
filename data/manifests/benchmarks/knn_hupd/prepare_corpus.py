"""Build the labelled corpus from HUPD year tarballs (one JSON per USPTO application).

Keeps applications with a title or abstract and at least one IPC label (main label first), samples
CORPUS_SIZE for the vote corpus and HOLDOUT_SIZE disjoint applications for an in-domain check
against USPTO-assigned labels. Output: D:/ipc_knn/corpus.jsonl and holdout.jsonl.

Usage: python prepare_corpus.py 2017 2016
"""

from __future__ import annotations

import json
import random
import re
import sys
import tarfile

from knn_common import WORK

CORPUS_SIZE = 300_000
HOLDOUT_SIZE = 2_000
SEED = 20261013


COMPACT = re.compile(r"^[A-H]\d{2}[A-Z]\d{3,}$")


def labels(rec: dict) -> list[str]:
    """HUPD writes IPC compactly ('G16H1060' = G16H 10/60), losing the main group/subgroup split.
    Raw codes are kept here, main label first; knn_eval.split_compact resolves them against the
    catalogue."""
    raw = [rec.get("main_ipcr_label") or ""] + list(rec.get("ipcr_labels") or [])
    out: list[str] = []
    for r in raw:
        r = r.strip().upper()
        if COMPACT.match(r) and r not in out:
            out.append(r)
    return out


def main(years: list[str]) -> None:
    pool: list[dict] = []
    stats = {"read": 0, "no_text": 0, "no_label": 0}
    for y in years:
        with tarfile.open(WORK / "raw" / "data" / f"{y}.tar.gz", mode="r|gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                rec = json.load(tf.extractfile(m))
                stats["read"] += 1
                abstract = (rec.get("abstract") or "").strip()
                if len(abstract) < 50:
                    stats["no_text"] += 1
                    continue
                lab = labels(rec)
                if not lab:
                    stats["no_label"] += 1
                    continue
                pool.append({"id": rec.get("application_number") or m.name, "year": y,
                             "filing_date": rec.get("filing_date"), "title": (rec.get("title") or "").strip(),
                             "abstract": abstract, "labels": lab})
                if stats["read"] % 50_000 == 0:
                    print(y, stats, len(pool), flush=True)
    random.Random(SEED).shuffle(pool)
    holdout, corpus = pool[:HOLDOUT_SIZE], pool[HOLDOUT_SIZE:HOLDOUT_SIZE + CORPUS_SIZE]
    for name, rows in (("corpus", corpus), ("holdout", holdout)):
        with (WORK / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"years": years, "seed": SEED, **stats, "pool": len(pool), "corpus": len(corpus), "holdout": len(holdout),
            "mean_labels": round(sum(len(r["labels"]) for r in corpus) / max(1, len(corpus)), 2)}
    (WORK / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main(sys.argv[1:] or ["2017", "2016"])
