# Architecture

```
 Browser ──► Next.js (apps/web)            server-side route handlers proxy to the API
               /, /taxonomy, /status         (IPC_API_BASE_URL never reaches the browser)
                     │
                     ▼
 FastAPI (apps/api) ── one process owns the models (BGE-M3 encoder, BGE reranker), bounded queue
   routes/        classify · ipc search/detail · health/ready/status · admin (token)
   services/
     ingestion/   validate → traverse tree → WIPO master files (wipo.py) → COPY-staged upsert → manifests
     embeddings/  providers (bge_m3 local | gemma_api) → resumable runs
     retrieval/   query views → subclass routing → full-text + HNSW in routed subclasses → RRF → expansion
     classification/ scope docs → reranker → evidence → policy + WIPO rules → optional LLM → summary
     confidence/  fingerprint → JSON calibrator artifacts → fit/evaluate/activate
     fields/      versioned field-of-invention mapping
                     │ psycopg 3 (bounded API pool + statement timeout; jobs use a direct connection)
                     ▼
 sakthi: ShaktiDB/PostgreSQL 18.4 + pgvector 0.8.2 (serving)  ·  local docker PG17 (tests)
   taxonomy_releases · ipc_records (tsvector GIN) · ipc_source_occurrences · ipc_reference_edges
   ipc_official_scheme_entries · ipc_official_texts (rule_kind, rules, scope GIN) · ipc_official_catchwords
   embedding_runs · ipc_embeddings vector(1024) [HNSW m=64, ef_c=800] · ipc_embeddings_gemma vector(768)
   ingestion_jobs · confidence_calibration_runs · classification_audit (off by default)
```

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
- **The retrieval layer** returns ranked codes with their component scores. It never decides a
  classification.
- **The classification layer** owns the decision policy, evidence and output validation. It
  hydrates every title and metadata field from PostgreSQL, so LLM text never supplies them.
- **The confidence layer** owns the only path to a probability. It is keyed by the pipeline
  fingerprint.
- **The frontend** renders only validated fields from the shared summary contract, and handles
  UTF-16 offsets explicitly.

## Key decisions

They are recorded as ADRs in the vault at `10-Projects/ipc/04-Decisions/`:

- BGE-M3 is the served index. EmbeddingGemma is a separate, comparison-only run in its own table.
- Calibration artifacts are JSON, never pickles. Active state is a checked pointer file, and the
  compatibility key is a fingerprint hash.
- Two-level RRF with equal per-source caps, so long specifications cannot outvote the abstract.
- Before calibration, a conservative relevance and evidence policy decides selection
  ([classification-policy.md](classification-policy.md)).
