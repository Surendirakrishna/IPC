# Model operation

## Models (pinned)

| Role | Model | Revision | Loaded by |
| --- | --- | --- | --- |
| Dense embeddings (served) | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | `SentenceTransformer(..., revision=...)`; fp16 on CUDA, fp32 on CPU |
| Dense embeddings (comparison) | `embeddinggemma-2` via the self-hosted OpenAI-compatible endpoint | served model id + created stamp (the endpoint exposes no weight hash) | `GemmaApiProvider` (httpx) |
| Reranker | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `huggingface_hub.snapshot_download(revision=...)`, then `FlagReranker(local_path)`, because FlagReranker has no revision argument |
| Optional LLM | `gemma-4-26b` (self-hosted) | n/a | `OpenAICompatibleClient`; **off** unless `LLM_ENABLED=true` |

Both revisions were checked against the Hugging Face API on 2026-10-08.

- Models load **once per API process**, in a background thread at startup. `/ready` reports progress.
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
- **Current state:** no expert-labelled data was supplied, so **no calibrator is fitted or active**.
  Every candidate reports `score: null, status: not_calibrated`, and the UI shows "confidence: not
  calibrated".

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
