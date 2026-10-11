# Retrieval evaluation

Two separate questions are measured here, and neither result is a measure of classification
accuracy:

1. **ANN fidelity:** how closely HNSW reproduces exact nearest-neighbour search (`ipc eval-retrieval`).
2. **Storage-precision fidelity:** how closely `halfvec(1024)` reproduces float32 exact top-k
   (`ipc compare-precision`).

Classification correctness needs expert-labelled data (`data/evaluation/README.md`). None has been
supplied, so **no accuracy figure exists and none is claimed**.

## Commands

```bash
uv run ipc --target local eval-retrieval --queries 200 --k 100 --ef-search 500
uv run ipc --target local compare-precision --queries 100 --k 100 --extra-query-file ../../data/evaluation/qkd_contract_request.json
uv run ipc --target local explain-vector --ef-search 200
uv run ipc --target local storage-probe            # synthetic vectors, scratch tables, local only
```

Probe queries are randomly sampled catalogue titles (seed 13), re-encoded as queries. Results are
written to `data/manifests/`.

## Results (local PG17 + pgvector 0.8.0, BGE-M3 run `emb-bge_m3-72c8b3da25a4`)

### HNSW versus exact recall@100

200 title-probe queries, record types `maingroup` and `subgroup`.

| Build (m / ef_construction) | ef_search 200 | ef_search 500 | Queries < 0.5 (ef 500) | Build time | Index size |
| --- | --- | --- | --- | --- | --- |
| 32 / 200 (spec starting values) | 0.615 | 0.765 | 37 | 26 s | 626 MiB |
| 32 / 500 | 0.802 | 0.902 | 11 | 51 s | 626 MiB |
| 48 / 500 | 0.800 | 0.906 | 11 | 64 s | 626 MiB |
| **64 / 800 (new default)** | 0.872 | **0.948**, re-measured as **0.955** (min 0.22) by `ipc eval-retrieval` | 4 | 124–127 s | 626 MiB |

- Builds ran in parallel on local Docker with `maintenance_work_mem=1GB` and `shm_size=2gb`.
- An exact sequential scan takes about 155 ms per query; HNSW takes about 48–53 ms (p50).

**Finding.** The catalogue's `semantic_text` vectors are tightly clustered: every record shares the
template and its long ancestor headings. The spec's starting values (m=32, ef_construction=200,
ef_search=200) therefore reached only 0.62 mean recall@100, and some queries had zero overlap. The
nearest result for those queries was at cosine distance 0.58, against an exact nearest of 0.29.

- Index size does not depend on `m` here, because the vectors dominate it.
- The defaults are now m=64, ef_construction=800 and ef_search=500.
- `VECTOR_SEARCH_MODE=exact` gives full recall for about 100 ms more per query view.
- Keyword retrieval, expansion and reranking reduce, but do not remove, the effect of a missed
  neighbour.

### halfvec(1024) versus float32 exact top-k

| Queries | overlap@10 (mean / min) | overlap@100 (mean / min) |
| --- | --- | --- |
| 101 (100 title probes + the QKD abstract) | 1.00 / 1.00 | 1.00 / 1.00 |

The exact top-100 under halfvec was identical to float32 for every probe. On this data, half
precision does not change exact ranking. It remains a tested storage alternative. It is not used, because the sakthi DB stores float32 as
specified.

### QKD smoke check (not calibration, not accuracy)

`data/manifests/acceptance_qkd_local.json` holds the live response:

- `needs_review`, field COMMUNICATION (H04), supported H04L 9/08 "Key distribution".
- Its quote is verified at `abstract[0:261]`.
- H04W key-management codes are alternatives, because the subclass scope term "wireless" is not
  evidenced.
- Modulation "keying" codes are alternatives, with no supporting passage.

It is one example and says nothing about general accuracy.

## Candidate recall on real applications (2026-10-11)

The two checks above measure index fidelity. Whether the right code reaches the candidate pool at
all is measured against the codes recorded on IP India applications, by the `stage_recall` and
`candidate_hit@100` fields of `ipc benchmark`. Per-stage figures, the routing comparison and the
reranker-input A/B are in
[benchmark-ipindia.md](benchmark-ipindia.md#dev-set-results-after-the-wipo-202601-rebuild-2026-10-11).
In short, on the 130-application dev set:

- Subclass-first (group-vote) routing keeps the same recall before reranking as flat retrieval
  (subclass 85.4%, main group 71.5%, exact 50.0% in the fused top 100).
- The WIPO 2026.01 vectors (`emb-bge_m3-61c53c85a23d`) and the previous vectors are within one or two
  documents of each other.
- The gap between finding the code and ranking it first is in the reranker, not in retrieval.

## Settings

- Defaults are `HNSW_M=64`, `HNSW_EF_CONSTRUCTION=800` and `HNSW_EF_SEARCH=500`, chosen from the table
  above. The spec's m=32/200/200 were the benchmark starting point. These are measured choices, not
  claimed optima.
- `ef_search` is set with transaction-local `set_config` and is capped at `HNSW_EF_SEARCH_MAX`.
- For filtered searches (section or subclass), `hnsw.iterative_scan=relaxed_order` is set and
  results are re-sorted by distance. The number of rows actually returned is reported when it
  falls below the requested pool.
- `record_type` and `subclass_code` are denormalized onto the embedding table, so filters apply in
  the same scan.

## EmbeddingGemma comparison run (agreement only)

| Item | Value |
| --- | --- |
| Run | `emb-gemma_api-e1b53a3633ed`, model `embeddinggemma-2` from the self-hosted endpoint |
| Vectors | 80,145 / 80,145 validated, 768-dim, in `ipc_embeddings_gemma` (local only, no HNSW) |
| BGE-M3 vs Gemma, exact top-100 overlap (101 queries) | mean 0.298, min 0.01, max 0.81 (`data/manifests/embedding_agreement_bge_vs_gemma.json`) |
| QKD abstract, exact rank of H04L 9/08 | BGE-M3: 64 · Gemma: 9 |
| QKD abstract, exact rank of H04W 12/0433 | BGE-M3: 24 · Gemma: 5 |

The two models retrieve very different neighbourhoods. One example cannot show which one serves
classification better. Answering that needs the expert-labelled evaluation set: run the pipeline
with `EMBEDDING_PROVIDER=gemma_api` against the Gemma run, then compare candidate recall on the
labelled codes. BGE-M3 remains the served index, as the spec requires (ADR-0001).
