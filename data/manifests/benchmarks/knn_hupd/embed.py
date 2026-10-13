"""Embed the corpus, the HUPD hold-out and the IP India dev/test queries with one model.

Text is "title. abstract" for every document, truncated at 512 tokens. Vectors are L2-normalised
float16, written in resumable shards under D:/ipc_knn/emb/<model>/.

Usage: python embed.py paecter|bge_m3 [--batch 32]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from knn_common import DEV_DOCS, TEST_DOCS, WORK, load_jsonl, query_text

MODELS = {
    "paecter": ("mpi-inno-comp/paecter", "1f355d9e9a34dc4859e22d05ce27e970eeacfbd2"),
    "bge_m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),  # same revision as the API
}
# PaECTER is fetched at that commit into D: (C: is nearly full; the HF downloader stalled on this
# network). BGE-M3 loads from the default HF cache the API already uses.
LOCAL = {"paecter": WORK / "models" / "paecter"}
SHARD = 20_000


def load_model(name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    model_id, revision = MODELS[name]
    kwargs = {"model_kwargs": {"torch_dtype": torch.float16}, "device": "cuda"}
    if name in LOCAL:
        model = SentenceTransformer(str(LOCAL[name]), **kwargs)
    else:
        model = SentenceTransformer(model_id, revision=revision, **kwargs)
    model.max_seq_length = 512
    model.eval()
    return model, model_id, revision


def encode(model, texts: list[str], batch: int) -> np.ndarray:
    import torch

    order = np.argsort([-len(t) for t in texts])  # longest first: less padding, early OOM
    with torch.inference_mode():
        out = model.encode([texts[i] for i in order], batch_size=batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    res = np.empty_like(out)
    res[order] = out
    return res.astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    out = WORK / "emb" / args.model
    out.mkdir(parents=True, exist_ok=True)
    model, model_id, revision = load_model(args.model)
    (out / "meta.json").write_text(json.dumps({"model_id": model_id, "revision": revision, "max_seq_length": 512,
                                               "dtype": "float16", "text": "title. abstract"}, indent=2))

    queries = {"dev": load_jsonl(DEV_DOCS), "test": load_jsonl(TEST_DOCS)}
    for split, docs in queries.items():
        np.save(out / f"query_{split}.npy", encode(model, [query_text(d) for d in docs], args.batch))
        (out / f"query_{split}_ids.json").write_text(json.dumps([d["application_number"] for d in docs]))

    for name in ("holdout", "corpus"):
        rows = load_jsonl(WORK / f"{name}.jsonl")
        texts = [query_text(r) for r in rows]
        for start in range(0, len(texts), SHARD):
            path = out / f"{name}_{start // SHARD:04d}.npy"
            if path.exists():
                continue
            t0 = time.perf_counter()
            np.save(path, encode(model, texts[start:start + SHARD], args.batch))
            dt = time.perf_counter() - t0
            print(f"{args.model} {name} {start + SHARD}/{len(texts)} {dt:.0f}s ({SHARD / dt:.0f} docs/s)", flush=True)
    print("done", args.model)


if __name__ == "__main__":
    main()
