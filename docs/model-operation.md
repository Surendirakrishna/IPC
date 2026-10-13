# Model operation

## Models (pinned)

| Role | Model | Revision | Loaded by |
| --- | --- | --- | --- |
| Dense embeddings (served) | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | `SentenceTransformer(..., revision=...)`; fp16 on CUDA, fp32 on CPU |
| Dense embeddings (comparison) | `embeddinggemma-2` via the self-hosted OpenAI-compatible endpoint | served model id + created stamp (the endpoint exposes no weight hash) | `GemmaApiProvider` (httpx) |
| Reranker | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `huggingface_hub.snapshot_download(revision=...)`, then `FlagReranker(local_path)`, because FlagReranker has no revision argument |
| Patent encoder (neighbour vote) | `mpi-inno-comp/paecter` | `1f355d9e9a34dc4859e22d05ce27e970eeacfbd2` | `PaecterEncoder` ([paecter.py](../apps/api/app/services/embeddings/paecter.py)) from `PATENT_ENCODER_LOCAL_PATH`, with the weights' SHA-256 checked against the revision; else the Hugging Face snapshot at that revision, download bounded by `PATENT_ENCODER_DOWNLOAD_TIMEOUT_S` (300 s). fp16 on CUDA. Loaded **only** with `NEIGHBOUR_VOTE_ENABLED=true` |
| Optional LLM | `gemma-4-26b` (self-hosted) | n/a | `OpenAICompatibleClient`; **off** unless `LLM_ENABLED=true` |

The BGE revisions were checked against the Hugging Face API on 2026-10-08. The PaECTER revision is
pinned in `config.py` (`PATENT_ENCODER_MODEL_REVISION`) and `paecter.py`, and every corpus records the
encoder revision it was embedded with. The vote refuses a corpus embedded with another revision
(`fallback_reason: patent_encoder_unavailable`).

- **PaECTER input:** "title. abstract" (the abstract alone without a title), truncated at 512 tokens,
  1,024 dimensions, L2-normalized. Corpus documents and queries use the same text, so vectors imported
  from the offline experiment (`--from-npy`) are comparable.
- Models load **once per API process**, in a background thread at startup. `/ready` reports progress.
  PaECTER loads after the API is `ready`, in its own thread. Until it is loaded, and for good if it
  fails, the vote answers `patent_encoder_unavailable` and the definition pipeline serves. A PaECTER
  failure never makes `/ready` 503; it shows in `checks.neighbour_vote.patent_encoder_error`.
- Models are never downloaded per request, and never run in the database, the browser or a
  serverless function.
- Inference is serialized by a lock per model. The API bounds concurrent classify calls with
  `INFERENCE_QUEUE_SIZE` and returns 503 `busy` beyond that.
- One model-owning process (`--workers 1`). To scale out, move the encoder and reranker into a
  dedicated inference service.

## Embedding policy

- **Catalogue entries:** exactly the supplied `semantic_text`, with normalized dense vectors and no
  prefix. BGE-M3 measured 1,024 dimensions. Tokenizer lengths for all 80,145 records:
  - median 111, 99th percentile 234, maximum 331 tokens (`C23C0022480000`);
  - so `TAXONOMY_MAX_TOKENS=512` truncates nothing.
  - Recorded by `ipc check-token-lengths`; manifest in `data/manifests/token_lengths_*`.
- **Queries:**
  - the abstract view is allowed up to 2,048 tokens and is otherwise rejected with 422;
  - optional-document chunks are 1,024 tokens with 128 overlap;
  - the model limit is 8,192.
- **Validation:** every vector must have the expected dimension, all finite values and a nonzero
  norm. `finalize_run` also checks that every norm is within 0.99–1.01.
- **Gemma:** uses the EmbeddingGemma retrieval prompts (`task: search result | query: …` and
  `title: none | text: …`). Vectors are L2-normalized client-side and stored in
  `ipc_embeddings_gemma` (`vector(768)`). They are **never** mixed with BGE-M3 vectors.

## Embedding runs (resumable)

```bash
uv run ipc --target local embed --provider bge_m3          # resumes automatically
uv run ipc --target local build-index --table ipc_embeddings --m 32 --ef-construction 200
uv run ipc --target local activate-embeddings <run_id>     # requires validated run + HNSW index
```

- A run is keyed by release, model id, revision, input hash and storage table.
- **Resume:** rows whose vector is missing, belongs to another run or has a stale
  `semantic_text_sha256` are re-embedded. Progress is committed per batch of 512.
- **Model change:** starting a run for a different model or revision supersedes the old run and
  stops it serving. `/ready` stays 503 until the new run is validated, indexed and activated. Mixed
  outputs are never served. The API also refuses (503 `embeddings_incompatible`) if the loaded query
  encoder differs from the model and revision of the serving run.
- **GPU out of memory:** the batch size is halved, up to 4 retries.

## Neighbour vote

The vote is **off by default** (`NEIGHBOUR_VOTE_ENABLED=false` in `config.py`). With it off, the API
answers with the definition pipeline alone (`classifier: definition_fallback`,
`fallback_reason: neighbour_vote_disabled`). The selection rule is in
[classification-policy.md](classification-policy.md#neighbour-vote-primary-classifier); the reasons
are in ADR-0005 ("A PaECTER neighbour vote is the primary classifier").

### Switching it on

1. A validated patent corpus must be serving (`ipc patent-status` shows `"is_serving": true`). Today
   that is `hupd-2017`.
2. Add to `.env` (repository root):

   ```
   NEIGHBOUR_VOTE_ENABLED=true
   PATENT_ENCODER_LOCAL_PATH=D:/ipc_knn/models/paecter
   ```

   Set the local path on this network: the Hugging Face downloader stalls here. The directory must be
   PaECTER at the pinned revision; its weights hash is checked at load.
3. Restart the API. The setting is read at startup.
4. Check `/ready`: `checks.neighbour_vote.state` must be `serving` and `patent_encoder_state` must be
   `ready`. The other states and their meaning are in [api.md](api.md). None of them makes `/ready`
   return 503, because the definition pipeline can still answer, so check the state itself.

To switch it off, set `NEIGHBOUR_VOTE_ENABLED=false` (or remove the line) and restart.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `NEIGHBOUR_K` | 100 | nearest corpus patents that vote |
| `NEIGHBOUR_TEMPERATURE` | 0.005 | neighbour weight `exp((s − s₁)/T)` |
| `NEIGHBOUR_BETA` | 1.0 | weight of secondary labels (the main label counts 1); at most 1 |
| `NEIGHBOUR_MIN_VOTE_SHARE` | 0.2 | below this first-code subclass share, the definition pipeline answers. **Provisional** |
| `PATENT_HNSW_EF_SEARCH` | 200 | `hnsw.ef_search` for the patent search; must be ≥ `NEIGHBOUR_K` (checked at startup), at most 1000 |
| `PATENT_ENCODER_LOCAL_PATH` | unset | local PaECTER copy at the pinned revision |
| `PATENT_ENCODER_DOWNLOAD_TIMEOUT_S` | 300 | bound on a Hugging Face download when there is no local copy |

k, T and β were chosen on the IP India dev set in the offline experiment
(`data/manifests/benchmarks/knn_hupd/report.json`). The minimum share has not been tuned on the served
vote yet.

### Memory

With the vote on, the API holds three models. Measured on the dev machine (RTX 2050 4 GB, 15.7 GiB
RAM) on 2026-10-14 with all three on CUDA:

| | Value |
| --- | --- |
| Private commit after load | 7.3 GiB |
| Peak commit during load | **about 8.3 GiB** (8.34) |
| VRAM, without / with PaECTER | 1,159 / 1,819 MiB |

- **OS error 1455** ("The paging file is too small for this operation to complete") on a cold start:
  BGE-M3 and the reranker load, then PaECTER fails. This is the Windows commit limit (RAM plus page
  file), not VRAM or disk. The API keeps serving with the fallback, and `/ready` shows the error in
  `checks.neighbour_vote.patent_encoder_error`. **Restart the API once**: a system-managed page file
  grows after the failed allocation, and the retry has worked. A larger fixed page file, or fewer open
  applications, avoids it.
- Always check `patent_encoder_state` after a start. The failure is easy to miss, because the API
  still answers.

## Patent corpus lifecycle

A corpus moves through these states. Every step is a CLI command, writes a manifest to
`data/manifests/patents/<step>_<corpus>.json` and holds an advisory lock on the corpus. Activation
and retirement also hold a global serving lock.

| Step | Command | State after | Notes |
| --- | --- | --- | --- |
| ingest | `ingest-patents` | `loading` | Reads the source, samples (seeded), keeps a hold-out out of the corpus, resolves labels against the serving taxonomy (WIPO concordance on by default). Idempotent upsert. |
| embed | `embed-patents` | `embedded` | Resumable PaECTER embedding, or import of precomputed vectors (`--from-npy` with `--ids`; every imported document's text is checked). |
| index | `build-patent-index` | `embedded` | Builds the corpus's own partial HNSW index. Refused while the corpus is serving. |
| validate | `validate-patent-corpus` | `validated` | recall@k of the HNSW search against exact search (default 200 probes, k = `NEIGHBOUR_K`, minimum 0.95), and document count = embedding count. |
| activate | `activate-patent-corpus` | serving | Refused unless validated with a valid index. Any other corpus stops serving. |
| retire | `retire-patent-corpus` | `embedded` | Drops the index (`DROP INDEX CONCURRENTLY`). A serving corpus needs `--stop-serving`; the vote then falls back until another corpus is activated. |

`load-patents` runs ingest → embed → index → validate in order, stops at the first failure, and
resumes when the same command is run again. It skips an ingest that already completed with the same
parameters. Once a corpus is embedded or validated, it refuses to ingest into it with different
parameters, so a smoke load never passes for the real one (`--reingest` overrides; a new corpus id is
the clean way).
Activation is always a separate step.

### HUPD (what is serving)

The `hupd-2017` corpus was loaded with these commands on 2026-10-14 (from `apps/api`; vectors
imported from the offline experiment):

```bash
uv run ipc --target sakthi load-patents --source hupd --years 2017 --corpus hupd-2017 \
    --seed 20261013 --holdout 2000 --from-npy D:/ipc_knn/emb/paecter --ids D:/ipc_knn/corpus.jsonl \
    --m 32 --ef-construction 200 --maintenance-work-mem 512MB
uv run ipc --target sakthi activate-patent-corpus --corpus hupd-2017
uv run ipc --target sakthi patent-status
```

The same load, step by step:

```bash
uv run ipc --target sakthi ingest-patents --source hupd --years 2017 --corpus hupd-2017 --seed 20261013 --holdout 2000
uv run ipc --target sakthi embed-patents --corpus hupd-2017 --from-npy D:/ipc_knn/emb/paecter --ids D:/ipc_knn/corpus.jsonl
uv run ipc --target sakthi build-patent-index --corpus hupd-2017 --m 32 --ef-construction 200 --maintenance-work-mem 512MB
uv run ipc --target sakthi validate-patent-corpus --corpus hupd-2017
uv run ipc --target sakthi activate-patent-corpus --corpus hupd-2017
```

HUPD tarballs (`YYYY.tar.gz`) are read from `HUPD_DATA_DIR` (default `D:/ipc_knn/raw/data`) or
`--data-dir`. Without `--from-npy`, `embed-patents` encodes with PaECTER from `--model-path`, else
`PATENT_ENCODER_LOCAL_PATH`, else the pinned Hugging Face snapshot (the same loader as the API).

| hupd-2017 step | Result | Manifest |
| --- | --- | --- |
| ingest | 295,735 read, 2,000 held out, 293,711 selected, 26 with no resolvable label, **293,685 stored**; 148,239 labels resolved through the concordance; 401.4 s | `ingest_hupd-2017.json` |
| embed | 293,685 imported, 0 text mismatches; 810.0 s | `embed_hupd-2017.json` |
| index | m=32, ef_construction=200, maintenance_work_mem 512MB; **544.65 s, 749.6 MiB** | `index_hupd-2017.json` |
| validate | 200 probes, k=100, ef_search 200: **recall@100 0.9884** (lowest probe 0.9596); HNSW p50 129 ms | `validate_hupd-2017.json` |
| database size | 2.53 GB before → 3.68 GB after embed → **4.46 GB** after index | `db_size_hupd-2017.json` |

### Index build on sakthi: `--maintenance-work-mem 512MB`

A parallel HNSW build puts the graph in dynamic shared memory sized up to `maintenance_work_mem`,
and sakthi's PostgreSQL has a small `/dev/shm`. With `2GB` the build failed after 1.1 s
(`DiskFull: could not resize shared memory segment ... No space left on device`). With `512MB` (the
server default) the 293,685-row index built in 545 s. Use `512MB` on sakthi. A failed index step keeps
the ingest and embed work; re-running `load-patents` resumes at the index step.

### After activating

- **No API restart is needed.** The pipeline reads the serving corpus on every request.
- **Run `POST /v1/admin/reload-confidence`** (`Authorization: Bearer $ADMIN_TOKEN`). The confidence
  fingerprint includes the serving corpus's id and input hash, and it is only recomputed at startup
  and on that call, so it is stale until then.
- With the vote enabled, the fingerprint also includes the patent encoder and the vote settings. So
  enabling the vote, or activating another corpus, marks any fitted calibrator
  `incompatible_artifact`, including a reranker calibrator. No calibrator is fitted today; the open
  item is recorded in ADR-0005.

### USPTO grants (`--source uspto-odp`)

The reader for recent USPTO grants ([uspto_odp.py](../apps/api/app/services/patents/sources/uspto_odp.py))
needs a USPTO Open Data Portal API key. Put it in `.env` at the repository root, never in a command line
or a committed file:

```
USPTO_ODP_API_KEY=<your key>
```

The key is sent only in a request header and is never logged. Without it, the reader stops before
the first request with an error naming `USPTO_ODP_API_KEY`.

**Not yet usable for a real load:**
- The endpoint, query and field names (`odp_fields.py`) have never been checked against a live
  response, because no key existed when it was written. Every `uspto-odp` run prints a warning saying so.
- There is an open must-fix in the year paging (TASK-0004): a year ends once the records read reach
  the response's `total_hits`. If the API caps or misreports that count, a year is cut short without an
  error. Fix it before the first real USPTO load.

Once both are resolved, check a small load in a separate corpus first, then load the full range:

```bash
uv run ipc --target sakthi load-patents --source uspto-odp --years 2021-2025 --corpus uspto-smoke \
    --limit 2000 --m 32 --ef-construction 200 --maintenance-work-mem 512MB
uv run ipc --target sakthi load-patents --source uspto-odp --years 2021-2025 --corpus uspto-2021-2025 \
    --m 32 --ef-construction 200 --maintenance-work-mem 512MB
uv run ipc --target sakthi activate-patent-corpus --corpus uspto-2021-2025
```

m and ef_construction must be measured for a corpus of that size: validation fails below recall@100
0.95, and the remedy is `build-patent-index --replace` with larger values. The new corpus gets its own
partial index next to the serving one, so `hupd-2017` keeps serving until the activation.

## Confidence calibration

**The event being calibrated:** an expert accepts this code as a supported candidate at the
requested level, under the selected taxonomy, based on the supplied input fields and the documented
rule checks.

```bash
uv run ipc generate-pairs --eval-file data/evaluation/labelled.jsonl     # frozen pipeline scores
uv run ipc fit-calibrator data/calibration/pairs_labelled.jsonl --method platt \
    --label-provenance "examiners A/B, protocol v1"                       # fit split; dev metrics
uv run ipc evaluate-calibration data/calibration/cal-platt-XXXX.json data/calibration/pairs_labelled.jsonl  # test split
uv run ipc activate-calibrator data/calibration/cal-platt-XXXX.json --max-ece 0.05 \
    --min-test-pairs 200 --min-test-positives 30 --justification "…measured on test split…"
```

- **Feature and method:** the raw `reranker_score`. Platt scaling uses scikit-learn
  `LogisticRegression` and is the default. Isotonic regression is available, but choose it only on
  dev data with enough samples.
- **Splits:** `dev` for method selection, `fit` for fitting, `test` for the final held-out
  evaluation. Patent families must not span splits. The QKD fixture is refused.
- **Metrics:** a reliability table (10 equal-width bins with counts), Brier score, log loss and ECE,
  reported overall and per input mode and level.
- **Artifacts:** plain JSON manifests in `data/calibration/`, never pickles. Each holds the
  parameters and their SHA-256, the full pipeline fingerprint, input modes and levels, label
  provenance, split sizes and the pairs-file hash, and the metrics. `ACTIVE.json` points at the
  active manifest.
- **Invalidation:** the fingerprint ([fingerprint.py](../apps/api/app/services/confidence/fingerprint.py))
  covers the embedding policy and serving input hash, the reranker revision, the scope formatter
  version, the candidate-generation settings, the taxonomy and source hashes, and the field-mapping
  version. Changing any of them changes the hash. Confidence then becomes `incompatible_artifact`
  with `score=null`, while retrieval keeps working. An input mode or level absent from the
  calibration data gives `unsupported_input_mode`.
- **Current state:** no expert-labelled data was supplied, so **no calibrator is fitted or active**,
  neither for reranker scores nor for vote shares.
  - Vote codes (`classifier: neighbour_vote`) report the raw vote share at the code's own level:
    `status: vote_share_uncalibrated`, `score` = the share. The summary shows "vote share: 87.9%",
    and the UI shows the share followed by "(uncalibrated)". It is never called a probability or a
    confidence.
  - Definition-path codes report `score: null, status: not_calibrated`, and the UI shows
    "confidence: not calibrated".
  - The offline experiment found the vote share close to calibrated already
    ([benchmark-ipindia.md](benchmark-ipindia.md#nearest-neighbour-vote-over-labelled-patents-offline-experiment-2026-10-13)),
    but no calibrator has been fitted on the served vote.

## Label protocol

Experts judge each candidate code shown for a document as `accept` or `reject` at the requested
level, using only the supplied fields. They record whether their judgement list is `complete`.

- **What reviewers see.** Each review item lists the title, abstract and claims, plus up to 10 of the
  pipeline's candidates merged with the applicant-declared codes found in the catalogue. Every
  candidate shows its code, title and full WIPO path. The list is shuffled per document and carries
  no rank and no "declared" marker, so neither the model's order nor the applicant's choice anchors
  the judgement.
- **Judgements:** `accept`, `reject` or `unsure` per candidate. `unsure` and unjudged candidates
  produce no label. A reviewer may add codes the list missed; added codes count as accepts.
- **Completeness.** A reviewer sets `accepted_codes_are_exhaustive` only when the accepted codes
  are everything that applies across the whole catalogue. Only then are unlisted codes treated as
  negatives (`judgement_completeness: "complete"`). Otherwise the item is `partial`.
- **Reviewer** and **reviewed_at** are required on every reviewed item. Indexing codes (WIPO
  `entry_type` `I`) are never offered as candidates.
- **Claims are review evidence only.** They are shown to the reviewer but are not used for candidate
  retrieval, and the finalized labels use `input_mode: "abstract_only"`.

## Review, family splits and reranker training data

None of this needs a GPU, and none of it changes what the API serves. The file format is in
[data/evaluation/README.md](../data/evaluation/README.md) ("Producing expert labels").

```bash
# 1. Group applications into patent families and assign family-safe splits.
#    PATH=SPLIT pins that file's split; unpinned families are hashed into fit/dev/test.
uv run ipc families ../../data/evaluation/ipindia_2026-09.jsonl=dev \
    ../../data/evaluation/ipindia_test_pool_2026-10.jsonl=test \
    --raw-dir ../../data/evaluation/ipindia_raw/2026-09/details \
    --raw-dir ../../data/evaluation/ipindia_raw/test_pool_2026-10/details \
    --out ../../data/evaluation/families_ipindia.json          # exits 1 on any cross-split family

# 2. Build blinded review items from a benchmark run (keep the key file away from reviewers).
uv run ipc review-queue <docs.jsonl> --results-file <run>/results.jsonl \
    --families-file ../../data/evaluation/families_ipindia.json --out queue.jsonl --key-out queue_key.jsonl

# 3. Turn completed reviews into expert-labelled JSONL (validated by the calibration loader).
uv run ipc review-finalize queue.jsonl --source "<who reviewed, when>" --out labelled.jsonl

# 4. Mine reranker training groups (fit split only by default).
uv run ipc mine-negatives labelled.jsonl --results-file <run>/results.jsonl \
    --out groups.jsonl --flag-out flag_rows.jsonl
```

- **Families.** Two applications share a family if they share a priority claim, cite the same parent
  PCT application in their cross-reference wording, or have near-identical title and abstract
  (5-word-shingle Jaccard ≥ 0.8). The whole family goes to one split. On the current IP India data,
  255 applications form 253 families: one pair is inside dev and one inside test, so **no family
  crosses dev and test**. Priority claims are read from the saved detail pages because the older
  JSONL files predate that field.
- **Hard negatives,** in priority order, all excluding accepted codes and indexing codes:
  1. `explicit_reject`: codes the reviewer rejected;
  2. `ranked_confusable`: codes the pipeline ranked in its top 20;
  3. `sibling`: codes under the same dotted WIPO parent as an accepted code;
  4. `same_subclass`, then `same_class`: nearby parts of the scheme.

  Only explicit rejects are certain negatives. In a `partial` review the others are marked
  `assumed: true`, because the reviewer did not rule them out. Train on them with that in mind, or
  drop them. In a `complete` review every unaccepted code is a real negative.
- **Matching the serving input.** Candidate scope text is built by the same `format_scope` the
  reranker uses, with the current `RERANK_OFFICIAL_CONTEXT` setting. Every group records it as
  `scope_formatter` (for example `scope-v3:official=False`). Regenerate the groups whenever that
  value changes, or the model is trained on inputs it will never see.
- **Current state:** the tooling is tested end to end on real data, but **no reviewed labels exist**,
  so no reranker has been trained and no calibrator fitted. Benchmark agreement with declared IPC
  is not a substitute: declared codes are noisy and often broad.
