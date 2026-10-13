"""Show what a user would see: paste an abstract, get IPC codes with vote-share probabilities.

Usage: python classify_demo.py <file with abstract text | request JSON with "abstract"> [model]
Uses the k / T / beta chosen on the dev set (report.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import embed
import knn_eval as ke
from knn_common import REPO, WORK, catalogue_codes, load_jsonl


def titles() -> dict[str, str]:
    out = {}
    with (REPO / "data" / "source" / "ipc_classification_index.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["display_code"]] = r["title"]
    return out


def main() -> None:
    src = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "paecter"
    raw = src.read_text(encoding="utf-8")
    text = json.loads(raw)["abstract"] if src.suffix == ".json" else raw
    params = json.loads((Path(__file__).parent / "report.json").read_text())["models"][model_name]["chosen_on_dev"]

    import torch
    corpus = load_jsonl(WORK / "corpus.jsonl")
    present = catalogue_codes()
    labels = ke.resolve_labels(corpus, present)
    d = WORK / "emb" / model_name
    corpus_emb = torch.from_numpy(np.concatenate([np.load(p) for p in sorted(d.glob("corpus_*.npy"))])).cuda()
    model, *_ = embed.load_model(model_name)
    q = embed.encode(model, [text.strip()], 1)
    sims, idx = ke.search(corpus_emb, q)
    p = ke.vote(sims[0], idx[0], labels, params["k"], params["T"], params["beta"])
    t = titles()
    print(f"k={params['k']} T={params['T']} beta={params['beta']}  first code: {p['supported'][0]}  {p['probability']}")
    for level in ("subclass", "maingroup", "exact"):
        print(f"\n{level}:")
        for code, share in list(p["top_share"][level].items())[:5]:
            print(f"  {share:5.0%}  {code:<14} {t.get(code, '')[:90]}")
    print("\nnearest corpus applications:", [(corpus[j]["id"], corpus[j]["title"][:60], round(float(s), 3))
                                             for j, s in zip(idx[0][:3], sims[0][:3])])


if __name__ == "__main__":
    main()
