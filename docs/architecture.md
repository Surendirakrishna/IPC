# Architecture

```
 Browser ──► Next.js (apps/web)            server-side route handlers proxy to the API
               /, /taxonomy, /status         (IPC_API_BASE_URL never reaches the browser)
                     │
                     ▼
 FastAPI (apps/api) ── one process owns the models, bounded queue:
                       BGE-M3 encoder · BGE reranker · PaECTER patent encoder (only with NEIGHBOUR_VOTE_ENABLED)
   routes/        classify · ipc search/detail · health/ready/status · admin (token)
   services/
     ingestion/   validate → traverse tree → WIPO master files (wipo.py) → COPY-staged upsert → manifests
     embeddings/  providers (bge_m3 local | gemma_api) → resumable runs · paecter.py (patent encoder)
     patents/     sources (hupd | uspto-odp) → labels + WIPO concordance → ingest → embed → index → validate → activate
     retrieval/   query views → subclass routing → full-text + HNSW in routed subclasses → RRF → expansion
                  patent_search.py: k nearest corpus patents through the serving corpus's partial HNSW index
     classification/
       primary:   abstract (+title) → PaECTER → k=100 nearest patents → neighbour vote (neighbour_vote.py)
                  → gate: first code's subclass share ≥ NEIGHBOUR_MIN_VOTE_SHARE → vote codes, in vote order
                  → definition path as EXPLANATION: scope docs → reranker scores → evidence → WIPO rules → summary
       fallback:  definition path as CLASSIFIER (vote off, no serving corpus, encoder missing, vote weak/empty/failed):
                  scope docs → reranker → evidence → policy + WIPO rules → optional LLM → summary
     confidence/  fingerprint → JSON calibrator artifacts → fit/evaluate/activate
     fields/      versioned field-of-invention mapping
                     │ psycopg 3 (bounded API pool + statement timeout; jobs use a direct connection)
                     ▼
 sakthi: ShaktiDB/PostgreSQL 18.4 + pgvector 0.8.2 (serving)  ·  local docker PG17 (tests)
   taxonomy_releases · ipc_records (tsvector GIN) · ipc_source_occurrences · ipc_reference_edges
   ipc_official_scheme_entries · ipc_official_texts (rule_kind, rules, scope GIN) · ipc_official_catchwords
   embedding_runs · ipc_embeddings vector(1024) [HNSW m=64, ef_c=800] · ipc_embeddings_gemma vector(768)
   patent_corpora (state, is_serving: at most one) · patent_documents (title, abstract, raw + resolved IPC labels)
   patent_embeddings halfvec(1024) [one partial HNSW index per corpus, WHERE corpus_id = '<id>';
                                    hupd-2017: m=32, ef_c=200, 750 MiB]
   ingestion_jobs · confidence_calibration_runs · classification_audit (off by default)
```

`classifier` in every response says which path chose the codes (`neighbour_vote` or
`definition_fallback`), and `fallback_reason` says why the vote did not
([api.md](api.md#post-v1classify)). The selection rule is in
[classification-policy.md](classification-policy.md#neighbour-vote-primary-classifier).

## Labelled patent corpus (neighbour vote)

Migration `0004_patent_corpus.py` adds three tables. A corpus is one load of labelled patents, for
example `hupd-2017`.

| Table | Holds |
| --- | --- |
| `patent_corpora` | corpus id, source, state (`loading` → `embedded` → `validated`, or `failed`), `is_serving` (only a validated corpus; a unique partial index allows one), encoder model and revision, input hash, label edition |
| `patent_documents` | title, abstract, dates, the source's raw labels and the labels resolved against the serving taxonomy (main label first), per corpus |
| `patent_embeddings` | one PaECTER `halfvec(1024)` per document, stored inline (`STORAGE PLAIN`: 2,056 bytes, just over the TOAST threshold) |

- **One partial HNSW index per corpus** (`patent_hnsw_<slug>_<hash>`, `halfvec_cosine_ops`,
  `WHERE corpus_id = '<id>'`). A second corpus can be embedded, indexed and validated next to the
  serving one without changing its search. `patent_search.py` inlines the serving corpus id as a
  literal so the planner can use that index.
- **At most one corpus serves.** The pipeline reads the serving corpus on every request, so
  activating another one needs no API restart.
- **Labels** are resolved to the 2026.01 catalogue at ingest. Codes abolished since the source's
  edition go through the WIPO concordance lists in `data/source/` (for HUPD 2017, 148,239 labels,
  including all of H01L, which moved to H10). A label that resolves nowhere is dropped; a document
  with no resolvable label is not stored (26 in hupd-2017). Counts: `data/manifests/patents/ingest_hupd-2017.json`.
- **Serving today (2026-10-14):** `hupd-2017`, 293,685 USPTO applications filed in 2017, index
  749.6 MiB (m=32, ef_construction=200), recall@100 0.9884 against exact search
  (`data/manifests/patents/{index,validate}_hupd-2017.json`). The load grew sakthi from 2.53 GB to
  4.46 GB (`db_size_hupd-2017.json`).
- Operation (load, validate, activate, retire): [model-operation.md](model-operation.md#patent-corpus-lifecycle).

## WIPO master data (2026.01)

`data/source/wipo_ipc_{scheme,definitions,catchwordindex}_20260101.zip` each hold an EN and an FR XML;
the EN file is parsed ([wipo.py](../apps/api/app/services/ingestion/wipo.py)).

- **Hierarchy.** Each scheme entry's parent is the entry it is nested in, so a subgroup sits under its
  dotted parent (A01B 3/08 → 3/06 → 3/04 → 3/00), not directly under its main group as in the supplied
  index. `semantic_text`, which is embedded and searched, is the full path with each subgroup's dot level.
- **Entry types.** `K` entries are classification places; `I` entries (1,543, e.g. B29K, C12R, groups
  101/00 and up) are indexing codes. Both are stored; retrieval only returns `K`.
- **Notes and references.** Each scheme note paragraph, title reference and definition part is a row in
  `ipc_official_texts` with its owner symbol, the entry it sits in (`container_symbol`), its range
  (`scope_end_symbol`, resolved to `scope_symbols`), its note type and its edition(s).
- **Meaning.** The wording is read into `rule_kind` and structured `rules`: `precedence` (preferred and
  dominated places, with a `conditional` flag), `place_last`/`place_first` (scope), `limiting_reference`
  (cited places), and context-only kinds (`informative_reference`, `indexing`, `definition`, `scope`,
  `other`). Negated rules ("the last place priority rule is not applied") produce none. Counts for
  2026.01 are in `data/manifests/validation_20260101.json`.

## Boundaries

- **The DB layer's tables own the data.** All SQL is parameterized. The HNSW index is built by a
  separate command after loading, not in migrations.
- **The retrieval layer** returns ranked codes with their component scores, and `patent_search.py`
  returns the k nearest corpus patents with their labels. Neither decides a classification.
- **The patents layer** (`services/patents/`) owns the corpus lifecycle: reading a source, resolving
  labels, embedding, building and validating the per-corpus index, and activation. The request path
  never writes to the patent tables.
- **The classification layer** owns the decision policy, evidence and output validation. With the
  vote primary, the vote picks the codes and their order. Evidence and lexical checks only add review
  notes, and WIPO precedence rules may only demote one vote code in favour of another vote code. It
  hydrates every title and metadata field from PostgreSQL, so neither LLM text nor corpus text
  supplies them.
- **The confidence layer** owns the only path to a probability. It is keyed by the pipeline
  fingerprint.
- **The frontend** renders only validated fields from the shared summary contract, and handles
  UTF-16 offsets explicitly.

## Key decisions

They are recorded as ADRs in the vault at `10-Projects/ipc/04-Decisions/`:

- A PaECTER neighbour vote over labelled patents is the primary classifier; definition matching
  explains its codes and is the fallback. Patent vectors are `halfvec(1024)` with one partial HNSW
  index per corpus (ADR-0005, "A PaECTER neighbour vote is the primary classifier").
- BGE-M3 is the served index. EmbeddingGemma is a separate, comparison-only run in its own table.
- Calibration artifacts are JSON, never pickles. Active state is a checked pointer file, and the
  compatibility key is a fingerprint hash.
- Two-level RRF with equal per-source caps, so long specifications cannot outvote the abstract.
- Before calibration, a conservative relevance and evidence policy decides selection
  ([classification-policy.md](classification-policy.md)).
