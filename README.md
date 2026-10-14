# ipc-classifier

Paste a patent abstract and get the most likely **IPC codes**, each with a **vote share** and the
evidence behind it. The title, specification and claims are optional.

The primary classifier is a **neighbour vote**:
1. It finds the 100 already-classified patents most similar to your abstract.
2. Their examiner-assigned IPC codes vote.
3. Definition matching (search and reranking against the WIPO IPC 2026.01 scheme) explains the
   chosen codes, and takes over when the vote is weak or unavailable.

- **Backend:** FastAPI + PyTorch (`apps/api`).
- **Frontend:** Next.js (`apps/web`).
- **Data:** PostgreSQL with pgvector on the `sakthi` server.

---

## Contents

1. [Results at a glance](#results-at-a-glance)
2. [Example output](#example-output)
3. [How it works](#how-it-works)
4. [Repository layout](#repository-layout)
5. [Quick start](#quick-start)
6. [Configuration](#configuration)
7. [Using the API](#using-the-api)
8. [CLI reference](#cli-reference)
9. [Data](#data)
10. [Testing](#testing)
11. [Operations notes](#operations-notes)
12. [Storage (measured)](#storage-measured)
13. [Status and known gaps](#status-and-known-gaps)
14. [Documentation index](#documentation-index)
15. [Security](#security)

---

## Results at a glance

These figures are from the held-out test set: 123 published Indian patent applications (IP India,
journal issues September to October 2026), title and abstract only. The test set was scored once.
A prediction counts as correct when the **first** returned code matches any IPC code recorded on
the application. Figures in brackets are 95% Wilson intervals.

| Classifier | Subclass (e.g. `H04L`) | Main group (`H04L 9/00`) | Exact (`H04L 9/08`) | Field of invention | Median time |
| --- | --- | --- | --- | --- | --- |
| Definition matching only (before 2026-10-13) | 30.9% | 17.9% | 8.9% | 41.5% | 9.5 s |
| **Neighbour vote, live on sakthi (2026-10-14)** | **63.4%** (54.6–71.4) | **44.7%** (36.2–53.5) | **30.1%** (22.7–38.7) | **56.1%** | **1.6 s** |

- **What the numbers measure.** Agreement with the IPC recorded on publication, which the
  applicant usually declares. It is not examiner-verified accuracy.
- **How the vote was chosen and tuned.** Dev set (130 different applications): 71.5 / 59.2 / 32.3%.
  The offline experiment that led to the vote gave 65.9 / 48.0 / 31.7% on the same test set.
- **Answering and fallback.** The vote never abstains. 124 of 125 test responses came from the vote,
  and one weak vote fell back to definition matching.
- **Details.** [docs/benchmark-ipindia.md](docs/benchmark-ipindia.md#neighbour-vote-served-live-held-out-test-2026-10-14).

---

## Example output

The first output is always a two-row summary. This is the live result for the QKD abstract in
`data/evaluation/qkd_contract_request.json`: neighbour vote on, corpus `hupd-2017`, HTTP 200 in
4.0 s, 2026-10-14.

| Field Of Invention | COMMUNICATION |
| :--- | :--- |
| Classification (IPC) | H04L 9/08 — vote share: 87.9%; H04L 9/14 — vote share: 34.1%; H04L 9/32 — vote share: 22.1% |

- **H04L 9/08 "Key distribution".** Of the 100 nearest patents (weighted by similarity), 97.7%
  carry H04L, 92.5% carry H04L 9/00, and **87.9%** carry exactly H04L 9/08. The nearest patents are
  2017 US filings on quantum key distribution (similarity about 0.96).
- **Below the minimum share.** H04B 10/70 "Photonic quantum communication" has 16.1%, below the 0.2
  minimum, so it is not listed.
- **Similar patents.** The UI also shows the 10 nearest patents, with similarity and IPC codes, plus
  the supporting passages and WIPO notes for each code.
- **With the vote off** (the default), definition matching returns H04W 12/0433 "Key management
  protocols" (`data/manifests/acceptance/acceptance_qkd_sakthi_wipo2026.json`).

> **A vote share is not a calibrated probability.** The summary says "vote share" and the UI adds
> "(uncalibrated)". No calibrator has been fitted yet; see [Status and known gaps](#status-and-known-gaps).

---

## How it works

```
                 POST /v1/classify  {abstract, title?, claims?, specification?}
                                      │
             ┌────────────────────────┴─────────────────────────┐
             │ 1. Neighbour vote (primary, NEIGHBOUR_VOTE_ENABLED) │
             │   "title. abstract" ─► PaECTER (1024-dim)          │
             │   ─► 100 nearest patents in the serving corpus     │
             │      (patent_embeddings halfvec(1024), one         │
             │       partial HNSW index per corpus)               │
             │   ─► their IPC codes vote, weighted by similarity  │
             └────────────────────────┬─────────────────────────┘
                                      │ first code's subclass share ≥ 0.2?
                     yes ◄────────────┴────────────► no / vote unavailable
                      │                                  │
   2a. Explain the vote's codes:                2b. Definition matching (fallback):
       passages, WIPO notes,                        subclass routing ─► full-text + BGE-M3
       precedence rules (they only                   vector search over IPC 2026.01
       reorder; never add a code)                    ─► RRF fusion ─► BGE-reranker-v2-m3
                      │                              ─► evidence ─► selection policy
                      └──────────────┬───────────────────┘
                                     ▼
       Response: summary table, candidates (vote share or confidence status), alternatives,
       similar patents, field of invention, warnings, provenance (models, corpus, settings).
       `classifier` says which path chose the codes; `fallback_reason` says why the vote didn't.
```

Key points:

- **Why a vote.** Definition matching compares an abstract with short, abstract IPC headings, using
  general-purpose models. It plateaued at about 31% subclass accuracy. The vote instead learns from
  how examiners actually classified similar patents, which roughly doubled accuracy. The reasoning
  is recorded in the vault decision **ADR-0005** (neighbour vote as the primary classifier).
- **Vote maths.**
  - Each neighbour has weight `exp((s − s₁)/T)`, with T = 0.005.
  - A neighbour's main label counts 1 and its other labels count β = 1.0.
  - A code's share is the weighted share of neighbours that carry it.
  - The first code is chosen top-down: subclass, then main group, then subgroup.
- **The corpus.** `hupd-2017` is 293,685 US applications filed in 2017 (from the Harvard USPTO
  Patent Dataset), with examiner-assigned IPC codes.
  - Codes abolished by IPC 2026.01 are remapped with WIPO's official concordance lists; for
    example, H01L became H10.
  - A newer corpus (USPTO grants 2021–2025) can be loaded with one command once a USPTO API key is
    available.
- **Fallback reasons.** The vote can be skipped for these reasons:
  - `neighbour_vote_disabled`
  - `no_serving_corpus`
  - `patent_encoder_unavailable`
  - `vote_empty`
  - `vote_below_minimum`
  - `vote_failed`

  The full selection rule is in [docs/classification-policy.md](docs/classification-policy.md).

---

## Repository layout

```
IPC/
├── README.md                     this file
├── .env.example                  template for the git-ignored .env (secrets, DB target, feature flags)
├── .gitattributes / .gitignore   *.zip / *.json in data via Git LFS; *.jsonl, caches, worktrees ignored
│
├── apps/
│   ├── api/                      FastAPI service, ML pipeline, Alembic migrations, `ipc` CLI (Python 3.12, uv)
│   │   ├── app/
│   │   │   ├── main.py, runtime.py      app factory; background model loading (BGE-M3, reranker, PaECTER)
│   │   │   ├── config.py                all settings (pydantic-settings, read from .env)
│   │   │   ├── cli.py                   the `ipc` command line (see CLI reference)
│   │   │   ├── routes/                  /health, /ready, /v1/classify, /v1/ipc/*, /v1/taxonomy/status, /v1/admin/*
│   │   │   ├── schemas/classify.py      request/response contract (v2.0.0)
│   │   │   ├── db/                      SQLAlchemy/psycopg sessions, catalogue + patent-corpus repositories
│   │   │   └── services/
│   │   │       ├── classification/      pipeline.py (both paths), neighbour_vote.py, reranker, evidence, WIPO rules, LLM, summary
│   │   │       ├── retrieval/           keyword + vector search, fusion, expansion, patent_search.py, HNSW index tools
│   │   │       ├── embeddings/          BGE-M3 / EmbeddingGemma providers, PaECTER encoder, resumable runs
│   │   │       ├── patents/             corpus ingest, label resolution, WIPO concordance, embed, per-corpus index
│   │   │       │   └── sources/         readers: HUPD tarballs, USPTO PatentSearch API
│   │   │       ├── ingestion/           IPC catalogue import + validation (WIPO 2026.01 master files)
│   │   │       ├── confidence/          calibration harness, fingerprint, IPC symbol parsing
│   │   │       ├── evaluation/          benchmark scoring, IP India parser, patent families, review queue
│   │   │       ├── fields/              IPC → IP India "Field Of Invention" mapping
│   │   │       └── training/            hard-negative mining for reranker training
│   │   ├── alembic/versions/            0001 schema … 0004 patent corpus tables
│   │   ├── tests/                       pytest suite (unit; `integration` and `ml` markers)
│   │   ├── Dockerfile, pyproject.toml, uv.lock
│   │
│   └── web/                      Next.js 16 UI (TypeScript, Tailwind) + Playwright tests
│       ├── app/                         pages: / (classify), /taxonomy, /status; app/api/* proxies to the API
│       ├── components/                  SummaryTable, CandidateCard, SimilarPatents, FallbackNotice, Evidence
│       ├── lib/                         types.ts (contract mirror), backend.ts, text.ts
│       ├── scripts/                     ipindia-collect.mjs / ipindia-explore.mjs (benchmark collection)
│       └── tests/                       classify.spec.ts (mocked contract), live-qkd.spec.ts, fixtures.ts
│
├── data/
│   ├── source/                   supplied originals: IPC catalogue JSONL/tree, WIPO 2026.01 scheme/definitions/
│   │                             catchwords, WIPO concordance lists 2017–2026 (+ manifest with SHA-256)
│   ├── evaluation/               benchmark sets (IP India dev 2026-09, test pool 2026-10), QKD contract request,
│   │                             patent families; ipindia_raw/ = raw fetched pages (git-ignored)
│   ├── calibration/              calibrator artifacts (empty: none fitted yet)
│   └── manifests/                every measurement and run record (see "Data" below)
│
├── docs/                         architecture, API contract, policy, operations, benchmarks (see index below)
│   └── reference/                supplied reference material: IPC_Complete_Catalogue.html,
│                                 IPC_Reference_Guide.docx, extract_ipc_schema.js (original scheme extractor)
│
└── infra/
    ├── docker-compose.yml        local pgvector (PG17 + pgvector 0.8.0) for integration tests only
    ├── docker-compose.app.yml    api + web containers (local, not a public deployment)
    └── sakthi-db.json            non-secret description of the serving database
```

---

## Quick start

### Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- Network access to the `sakthi` database (office LAN only)
- Optional: an NVIDIA GPU (CPU works, but more slowly) and Docker Desktop, needed only for the
  local test database

### 1. Configure

```bash
cp .env.example .env          # fill in secrets locally; .env is git-ignored, never commit it
# required: SAKTHI_DB_HOST, SAKTHI_DB_USER, SAKTHI_DB_PASSWORD   (IPC_DB_TARGET=sakthi is the default)
```

### 2. Run the API

The catalogue, embeddings and patent corpus already live on sakthi, so there is nothing to import.

```bash
cd apps/api
uv sync --extra ml --group dev
uv run ipc db-info                                   # check the connection (prints a redacted URL)
uv run uvicorn app.main:app --port 8000              # OpenAPI docs at http://localhost:8000/docs
curl http://localhost:8000/ready                     # 200 once the models are loaded (~30 s)
```

### 3. Run the UI

```bash
cd apps/web
npm ci
npm run dev                                          # http://localhost:3000 (calls the API on :8000)
```

### 4. Turn on the neighbour vote (recommended)

It is off by default. Add these lines to `.env`, then restart the API:

```
NEIGHBOUR_VOTE_ENABLED=true
PATENT_ENCODER_LOCAL_PATH=D:/ipc_knn/models/paecter     # local PaECTER copy; the hub download stalls on this network
EMBEDDING_DEVICE=cuda
```

`GET /ready` should then report `checks.neighbour_vote.state: "serving"`. PaECTER loads in the
background a few seconds after the API reports ready; until then the definition fallback answers.
For a different PaECTER location or other details, see
[docs/model-operation.md](docs/model-operation.md#switching-it-on).

Leave `APPLY_PLACE_RULES` off while the vote is on. On the QKD example it would move H04L 9/32 (22.1%)
ahead of H04L 9/08 (87.9%).

### Building a database from scratch (local pgvector)

```bash
docker compose -f infra/docker-compose.yml up -d     # PG17 + pgvector 0.8.0 on :5433, shm_size 2gb
cd apps/api
uv run ipc --target local migrate
uv run ipc --target local validate-sources
uv run ipc --target local import --activate
uv run ipc --target local embed --provider bge_m3    # resumable; ~10 min on an RTX 2050
uv run ipc --target local build-index --table ipc_embeddings --maintenance-work-mem 1GB
uv run ipc --target local activate-embeddings <run_id>
# patent corpus: see "Patent corpus" under CLI reference
```

### Docker (both services)

```bash
docker compose -f infra/docker-compose.yml -f infra/docker-compose.app.yml up --build
```

---

## Configuration

All settings live in `apps/api/app/config.py`. They are read from `.env` in the repository root
(or `apps/api`) and from environment variables, case-insensitively. The most important:

| Setting | Default | Purpose |
| --- | --- | --- |
| `IPC_DB_TARGET` | `sakthi` | `sakthi` (serving) or `local` (docker, tests). Per command: `ipc --target …`. |
| `SAKTHI_DB_HOST` / `_PORT` / `_NAME` / `_USER` / `_PASSWORD` / `_SSLMODE` | — / 5432 / `ipc_db` / — / — / `disable` | Serving database. The server has `ssl=off` (LAN-only exception). |
| `LOCAL_DATABASE_URL` | — | Local docker pgvector. |
| `NEIGHBOUR_VOTE_ENABLED` | `false` | Makes the neighbour vote the primary classifier. |
| `PATENT_ENCODER_LOCAL_PATH` | — | Local PaECTER directory. Without it the encoder is fetched from the Hugging Face hub (bounded by `PATENT_ENCODER_DOWNLOAD_TIMEOUT_S`, 300 s). |
| `NEIGHBOUR_K` / `NEIGHBOUR_TEMPERATURE` / `NEIGHBOUR_BETA` | 100 / 0.005 / 1.0 | Vote parameters, chosen on the dev set. |
| `NEIGHBOUR_MIN_VOTE_SHARE` | 0.2 | Below this first-code subclass share, the definition fallback answers. |
| `PATENT_HNSW_EF_SEARCH` | 200 | HNSW search breadth for the patent index (must be ≥ k, ≤ 1000). |
| `USPTO_ODP_API_KEY` | — | Only needed to load USPTO grants. |
| `HUPD_DATA_DIR` | `D:/ipc_knn/raw/data` | HUPD year tarballs, for reloading the corpus. |
| `EMBEDDING_DEVICE` | `auto` | `cuda` is recommended (see Operations notes). |
| `APPLY_PRECEDENCE_RULES` / `APPLY_PLACE_RULES` | `true` / `false` | WIPO rule handling. |
| `CLAIMS_RETRIEVAL_MODE` | `off` | Claims are used as evidence only (`independent_weighted` was tested). |
| `LLM_ENABLED` | `false` | Optional self-hosted LLM step (definition path only). |
| `ADMIN_TOKEN` | — | Enables `/v1/admin/*`. |
| `STORE_INPUT_DOCUMENTS` | `false` | Submitted text is not stored unless this is set. |

Model revisions are pinned in `config.py`:

| Model | Revision |
| --- | --- |
| BGE-M3 | `5617a9f6…` |
| BGE-reranker-v2-m3 | `953dc6f6…` |
| PaECTER | `1f355d9e…` (weights SHA-256 checked on load) |

---

## Using the API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check (no model, no database). |
| `GET /ready` | 200 when the models, schema, active release and embedding run are ready. `checks.neighbour_vote` reports the vote separately and never makes the API unready. |
| `GET /v1/taxonomy/status` | Release, counts, hashes, embedding runs, indexes, model revisions, calibration, neighbour vote. |
| `GET /v1/ipc/search?q=…` | Look up IPC codes by code or by full-text search. |
| `GET /v1/ipc/{code}` | One record: wording, parents, children, references. |
| `POST /v1/classify` | Classify a document (below). |
| `GET /v1/admin/jobs`, `POST /v1/admin/reload-confidence` | Need `Authorization: Bearer $ADMIN_TOKEN`. |

```bash
curl -s -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"abstract": "A quantum key distribution protocol for multi-node smart grid encryption ...",
       "title": null, "target_level": "group", "max_results": 5}'
```

The response, abbreviated:

```jsonc
{
  "status": "needs_review",
  "classifier": "neighbour_vote",          // or "definition_fallback"
  "fallback_reason": null,                 // why the vote did not answer, when it didn't
  "summary": { "classification_cell_text": "H04L 9/08 — vote share: 87.9%; ...", "markdown": "..." },
  "field_of_invention": { "label": "COMMUNICATION", ... },
  "candidates": [ { "display_code": "H04L 9/08", "title": "Key distribution", "level": "subgroup",
                    "vote_share": {"subclass": 0.9765, "maingroup": 0.925, "exact": 0.8787},
                    "confidence": {"status": "vote_share_uncalibrated", "score": 0.8787},
                    "supporting_passages": [ ... ], "unresolved_ambiguity": [ ... ], "origin": ["neighbour_vote"] } ],
  "alternatives": [ ... ],
  "vote": { "first_code": "H04L 9/08", "first_code_share": {...}, "top_codes": {...},
            "neighbour_count": 100, "neighbours": [ /* 10 nearest: id, title, similarity, IPC codes */ ] },
  "provenance": { "patent_encoder": {...}, "patent_corpus": {"corpus_id": "hupd-2017", "document_count": 293685, ...},
                  "vote_settings": {"k": 100, "temperature": 0.005, "beta": 1.0, "min_vote_share": 0.2, "hnsw_ef_search": 200} }
}
```

The full contract (every field, nullability rules and error codes) is in [docs/api.md](docs/api.md).
Errors always have the shape `{"error", "message", "details"}`.

---

## CLI reference

Run commands from `apps/api` as `uv run ipc [--target sakthi|local] <command>`. Each command writes
a JSON manifest under `data/manifests/`. `uv run ipc <command> --help` lists every option.

| Area | Commands |
| --- | --- |
| Database | `db-info`, `migrate` |
| IPC catalogue | `validate-sources`, `import`, `activate-release`, `check-token-lengths` |
| Catalogue embeddings | `embed`, `build-index`, `activate-embeddings`, `explain-vector`, `estimate-storage`, `eval-retrieval`, `storage-probe`, `compare-precision` |
| **Patent corpus** | `load-patents` (ingest → embed → index → validate, resumable), or step by step `ingest-patents`, `embed-patents`, `build-patent-index`, `validate-patent-corpus`; then `activate-patent-corpus`, `retire-patent-corpus`, `patent-status` |
| Benchmarks | `ipindia-parse`, `benchmark`, `benchmark-tune`, `benchmark-compare` |
| Calibration | `generate-pairs`, `fit-calibrator`, `evaluate-calibration`, `activate-calibrator` |
| Review and training data | `families`, `review-queue`, `review-finalize`, `mine-negatives` |

### Patent corpus

`hupd-2017` was loaded with the command below. It imports the precomputed PaECTER vectors, so
nothing is re-embedded. It needs the files under `D:/ipc_knn` (see [Data](#data)).

```bash
uv run ipc --target sakthi load-patents --source hupd --years 2017 --corpus hupd-2017 \
    --seed 20261013 --holdout 2000 --from-npy D:/ipc_knn/emb/paecter --ids D:/ipc_knn/corpus.jsonl \
    --m 32 --ef-construction 200 --maintenance-work-mem 512MB
uv run ipc --target sakthi activate-patent-corpus --corpus hupd-2017     # one corpus serves at a time
uv run ipc --target sakthi patent-status
```

- **Locking.** Lifecycle commands take advisory locks, so a second concurrent command fails fast.
- **Safety checks.** A serving corpus's index is never rebuilt in place. A resume refuses to proceed
  when the ingest parameters differ from the last completed run.
- **After activating a new corpus.** Call `POST /v1/admin/reload-confidence`; no restart is needed.
- **USPTO grants (`--source uspto-odp`).** These need `USPTO_ODP_API_KEY`, and the reader has an
  open must-fix (see [Status](#status-and-known-gaps)).
- **Everything else** about the lifecycle is in
  [docs/model-operation.md](docs/model-operation.md#patent-corpus-lifecycle).

---

## Data

### In the repository

| Path | Contents |
| --- | --- |
| `data/source/` | Supplied originals (tracked via Git LFS): `ipc_classification_index.jsonl` (80,145 records), `ipc_schema_tree.json`, `ipc_maingroup_duplicates.json`, WIPO 2026.01 scheme / definitions / catchword index, WIPO concordance lists 2017.01–2026.01 with `wipo_ipc_concordance_manifest.json` (URL + SHA-256 per file). |
| `data/evaluation/` | `ipindia_2026-09.jsonl` (dev, 130), `ipindia_test_pool_2026-10.jsonl` (held-out test, 125), `ipindia_2026-09_with_claims.jsonl`, `families_ipindia.json`, `qkd_contract_request.json` (contract fixture, never calibration data). See `data/evaluation/README.md`. |
| `data/calibration/` | Calibrator artifacts plus an `ACTIVE.json` pointer. Empty for now. |
| `data/manifests/` | Run records and measurements; layout below. |

`data/manifests/` layout:

| Folder | What is in it |
| --- | --- |
| `expected_snapshot_20260101.json` | **Input**: the expected source hashes and counts that `validate-sources` checks against. Keep it here, because the importer reads it from this path. |
| `imports/` | `validation_20260101.json` and the latest catalogue import; older imports are in `imports/history/`. |
| `embeddings/` | Catalogue embedding runs, token-length checks, BGE-M3 vs EmbeddingGemma agreement. |
| `retrieval/` | HNSW recall vs exact search at ef_search 100/200/500; halfvec vs float32 precision. |
| `storage/` | Local storage probe (PG17 + pgvector 0.8.0). |
| `acceptance/` | QKD acceptance responses for each pipeline stage. |
| `patents/` | Patent corpus manifests: `ingest_`, `embed_`, `index_`, `validate_`, `load_`, `activate_`, `db_size_hupd-2017.json`. |
| `benchmarks/` | One folder per benchmark run (`report.json`; `results.jsonl` is git-ignored). `neighbour_v1__{dev,test}_abstract/` are the live vote runs (with `serving.json`); `knn_hupd/` holds the offline experiment's scripts and report. |

New manifests from catalogue commands (`validate-sources`, `import`, `embed`, `eval-retrieval`, …)
are written to the top of `data/manifests/`. File them into the folders above when you commit them.
Patent and benchmark commands write straight into their subfolders.

### Outside the repository (`D:/ipc_knn`, about 7 GB)

These files are needed only to reload or rebuild the HUPD corpus or to repeat the offline
experiment. Serving does not use them, apart from the PaECTER model.

| Path | Contents |
| --- | --- |
| `models/paecter/` | PaECTER at the pinned revision (SHA-verified). **The API loads it from here.** |
| `emb/paecter/` | 293,711 precomputed PaECTER vectors (fp16 `.npy` shards + `meta.json`), imported by `--from-npy`. |
| `emb/bge_m3/` | BGE-M3 vectors from the offline model comparison (not used by the system). |
| `raw/data/2017.tar.gz` | HUPD 2017 applications (4.7 GB, SHA-256 checked on download). |
| `corpus.jsonl`, `holdout.jsonl` | Offline corpus and hold-out ids/text (`--ids` for the vector import). |

---

## Testing

```bash
cd apps/api && uv run pytest -m "not integration and not ml"          # unit tests (~800, ~10 s)
cd apps/api && uv run pytest -m ml                                    # needs the local PaECTER copy
cd apps/api && IPC_DB_TARGET=local uv run pytest -m integration       # needs the local docker DB
cd apps/web && npm run typecheck && npm run lint
cd apps/web && PW_CHANNEL=msedge npx playwright test                  # browser tests, mocked contract
cd apps/web && IPC_E2E_LIVE=1 PW_CHANNEL=msedge npx playwright test tests/live-qkd.spec.ts   # live API
```

- **Contract drift.** `tests/test_contract_classify_v2.py` checks that the backend schema,
  `docs/api.md` and the web types and fixtures agree.
- **Integration tests on sakthi.** They can run there only with
  `IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1`. They then use throwaway `it-test-*`
  corpora and clean up after themselves.
- **When they refuse to run.** Whenever a real corpus is serving, loading or embedded. While
  `hupd-2017` serves, use the local docker DB instead.
- **Benchmark discipline.** Tune on the dev set only. The held-out test set is scored once per
  pipeline version.

---

## Operations notes

- **Memory.** The API with BGE-M3 and the reranker commits about 5–7.5 GB, and with PaECTER (vote
  on) about **8.3 GiB at peak**.
  - If Windows runs short of commit memory, CUDA fails to start and `EMBEDDING_DEVICE=auto` falls
    back to the CPU, which may then fail. Set `EMBEDDING_DEVICE=cuda`, stop unused GPU or Docker
    workloads, or enlarge the page file.
- **OS error 1455** ("The paging file is too small"). PaECTER can fail this way on a cold start.
  - The API keeps answering with definition matching, and `/ready` shows the error.
  - Restart once: Windows grows the page file after the failure.
- **sakthi** is on the office LAN at 192.168.200.29:11233 (ShaktiDB 18.4, pgvector 0.8.2), with
  `ssl=off`. Use it on the trusted network only.
  - Each round trip takes about 57 ms; the vote path makes about five.
  - Patent HNSW builds there need `--maintenance-work-mem 512MB`: 2GB fails on the container's
    `/dev/shm`.
- **Windows Application Control** blocks Playwright's downloaded browsers (`PW_CHANNEL=msedge` uses
  the installed Edge) and SQLAlchemy's compiled modules (`pyproject.toml` builds SQLAlchemy as pure
  Python).

---

## Storage (measured)

| What | Size |
| --- | --- |
| IPC catalogue: records, tree, references, indexes | 337 MB |
| Catalogue vectors, `vector(1024)` float32 + TOAST | 427 MiB |
| Catalogue HNSW (m=64, ef_construction=800) | 626 MiB |
| Patent corpus `hupd-2017`: 293,685 documents, labels, `halfvec(1024)` vectors | 1,149 MB |
| Patent HNSW (m=32, ef_construction=200, partial index for `hupd-2017`) | 786 MB (749.6 MiB) |
| **sakthi `ipc_db` total** | 2.53 GB before the patent corpus → **4.46 GB** after |

- **Sources.** `data/manifests/storage/` and `data/manifests/patents/db_size_hupd-2017.json`.
- **USPTO estimate.** A full USPTO 2021–2025 load (about 1.6M grants) would add roughly 10 GB, so
  check free space on sakthi first.

---

## Status and known gaps

| Area | State (2026-10-14) |
| --- | --- |
| Neighbour vote | ✅ Built, reviewed and benchmarked live. **Off by default.** `hupd-2017` is validated and serving on sakthi (recall@100 0.9884). |
| Definition matching | ✅ Serving. WIPO 2026.01 hierarchy, subclass-first routing, BGE-M3 run `emb-bge_m3-61c53c85a23d` (recall@100 0.961). |
| API / UI / CLI | ✅ About 800 backend unit tests; Playwright: 10 mocked-contract tests + 1 live. |
| Calibration | ⛔ The harness exists, but no calibrator is fitted (no expert-labelled data). Vote codes show the raw vote share. Before fitting one, decide how calibrators are fingerprinted (open item in ADR-0005). |
| Recent patent data | ⛔ Needs a USPTO Open Data Portal key. The reader's endpoint is unverified, and it has a must-fix: end a year only on an empty page (see vault TASK-0004). |
| Benchmark provenance | ⚠ `ipc benchmark` reports don't record which classifier answered. `serving.json` next to each vote report fills the gap. |
| Reranker training | ⛔ Tooling ready (`families`, `review-queue`, `mine-negatives`); no reviewed labels yet. |
| Optional LLM | Adapter ready, disabled by default (`LLM_ENABLED=false`). |
| Repository hygiene | ⚠ `.env` is untracked now, but it is still in git history (commit `992594a`). Remove it from history before pushing, and rotate the sakthi password if it was ever pushed. |

---

## Documentation index

| Document | What it covers |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Components, data flow, database tables, boundaries, key decisions |
| [docs/api.md](docs/api.md) | The full HTTP contract: request, response, `/ready`, errors, worked example |
| [docs/classification-policy.md](docs/classification-policy.md) | Selection rules for both paths, minimum share, fallback, WIPO rules, edge cases |
| [docs/model-operation.md](docs/model-operation.md) | Pinned models, embedding runs, switching the vote on, memory, the patent corpus lifecycle, USPTO, calibration, label protocol |
| [docs/benchmark-ipindia.md](docs/benchmark-ipindia.md) | IP India benchmark: data, every result (offline and live), how to read the numbers, how to reproduce |
| [docs/retrieval-evaluation.md](docs/retrieval-evaluation.md) | Retrieval recall, HNSW tuning, BGE-M3 vs EmbeddingGemma |
| [docs/data-limitations.md](docs/data-limitations.md) | Known issues in the supplied catalogue data |
| [data/evaluation/README.md](data/evaluation/README.md) | Evaluation data and the expert-label JSONL schema |
| [apps/web/README.md](apps/web/README.md) | Web app: pages, scripts, tests |
| `docs/reference/` | Supplied reference material (complete catalogue HTML, reference guide, original extractor script) |

The reasons behind each design (ADRs), the task history and review findings are kept in the
project's Obsidian vault (`10-Projects/ipc`), not in this repository.

---

## Security

- Secrets live only in the git-ignored `.env`, and connection URLs are always printed redacted.
- The sakthi connection is unencrypted (`ssl=off`); use it only on the trusted LAN.
- Submitted text is not stored unless `STORE_INPUT_DOCUMENTS=true`.
- The optional LLM receives document text only when the operator enables it, and treats it as
  untrusted data.
- The USPTO key is sent only in a request header; it never appears in logs or manifests.
- `/v1/admin/*` requires `ADMIN_TOKEN` and returns 404 when it is unset.
