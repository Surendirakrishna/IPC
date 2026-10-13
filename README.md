# ipc-classifier

ipc-classifier classifies a patent abstract into IPC codes, with evidence for each code. You can also
supply the specification, claims and title, but they are optional. For each request the system:

- routes to likely subclasses first, then runs PostgreSQL full-text search and BGE-M3 vector search
  (pgvector HNSW) for groups inside them, over the WIPO IPC 2026.01 scheme with its real dotted hierarchy;
- fuses the two result lists with reciprocal rank fusion and expands them with parents and siblings
  (a cross-reference never adds a candidate);
- reranks candidates with BGE-reranker-v2-m3, then applies WIPO precedence rules between supported
  candidates;
- quotes supporting passages with exact offsets and lists unresolved ambiguity;
- applies an optional LLM step that can only choose from the candidate codes;
- maps the result to a field of invention and attaches calibrated per-code confidence.

The first output is always the two-row summary:

| Field Of Invention | COMMUNICATION |
| :--- | :--- |
| Classification (IPC) | H04W 12/0433 — confidence: not calibrated |

The table above is the **live result for the QKD abstract-only request** in
`data/evaluation/qkd_contract_request.json` with the current pipeline (WIPO 2026.01 rebuild, 2026-10-11).
It is not a hard-coded value; the full response is in `data/manifests/acceptance_qkd_sakthi_wipo2026.json`.

The supported code carries a review note: H04W's "wireless" scope isn't evidenced by the abstract.
H04L 9/14, 9/06 and other H04L 9 groups are listed as alternatives; policy v1 returned H04L 9/08.
While WIPO notes were still in the reranker input, this request returned H02J 50/80 (wireless power).

No calibrator has been fitted, so confidence is always `null` / "not calibrated"; no invented
percentages are shown.

## Status (2026-10-11)

| Area | State |
| --- | --- |
| Source validation | ✅ The hashes match the expected snapshot and every count matches (80,145 records, 589 duplicate main groups, 15,035 reference edges, 0 blocking findings). Manifest: `data/manifests/validation_20260101.json` |
| Local corpus (docker pgvector, tests only) | ✅ Imported and activated. BGE-M3 run `emb-bge_m3-72c8b3da25a4` has 80,145/80,145 vectors, is validated, has its HNSW index (m=64, ef_construction=800; recall@100 0.955 against exact search) and is serving. |
| **sakthi DB (serving)** | ✅ The default target since 2026-10-10: ShaktiDB 18.4 (PostgreSQL 18.4), pgvector 0.8.2, `ipc_db` on 192.168.200.29:11233. The catalogue is imported and active. Migrated to Alembic `0003` and re-imported with the WIPO 2026.01 master files on 2026-10-11 (real dotted hierarchy, notes with owner/range/edition and rule meaning). BGE-M3 run `emb-bge_m3-61c53c85a23d` (dotted-path text) is float32 and validated, with 80,145/80,145 vectors. HNSW m=64/ef_construction=800 was built in 105 s (626 MiB) and gives recall@100 of 0.961 against exact search (ef_search 500). The pre-rebuild vectors, text and run are kept in `rollback_20261011_*` tables. The QKD request returns 200 in about 14 s (`data/manifests/acceptance_qkd_sakthi_wipo2026.json`). The VPN link to sakthi has dropped several times; benchmark runs pause until it is back. |
| API / UI / CLI | ✅ Implemented. 74 backend unit tests (2026-10-11), 9 database integration tests (local docker DB) and 5 Playwright browser tests (including the live QKD run against sakthi) pass. |
| Benchmark (IP India, 123 held-out applications) | Re-scored once after the rebuild (2026-10-12), abstract-only: answers 97.6%; first code matches a recorded subclass 30.9%, main group 17.9%, exact 8.9% (previous pipeline 28.5 / 17.1 / 8.1, within the CIs). A recorded symbol is in the top 100 candidates for 78.0 / 59.3 / **44.7%** (previously 71.5 / 48.0 / 26.8). With claims as evidence only: 29.3 / 16.3 / 8.1 (previously, with claims voting: 27.6 / 10.6 / 2.4). Median 9.5 s. See [docs/benchmark-ipindia.md](docs/benchmark-ipindia.md). |
| Benchmark, dev set after the WIPO 2026.01 rebuild (2026-10-11) | Dev only (130 applications, used for design decisions). With subclass-first retrieval and no WIPO notes in the reranker, the first code matches a recorded subclass 42.3%, main group 24.6%, exact 11.5% (old flat v2 on the same set: 37.7 / 22.3 / 10.8). The correct code is in the top 100 candidates for 83.1 / 66.2 / 50.0%. Median 10.1 s per document. See [Dev-set results](docs/benchmark-ipindia.md#dev-set-results-after-the-wipo-202601-rebuild-2026-10-11). |
| Review and reranker-training tooling | ✅ `ipc families` (patent-family grouping, family-safe splits; the current dev and test sets share no family), `ipc review-queue` / `review-finalize` (blinded expert review, then expert-labelled JSONL), `ipc mine-negatives` (hard negatives from nearby parts of the scheme). See [docs/model-operation.md](docs/model-operation.md#review-family-splits-and-reranker-training-data). ⛔ No reviewed labels yet, so no reranker has been trained. |
| Calibration | Harness ✅. Fitted calibrator ⛔: no expert-labelled data. The review tooling above produces the input it needs. |
| Optional LLM | Adapter for the self-hosted `gemma-4-26b` ✅. **Disabled by default** (`LLM_ENABLED=false`). |
| EmbeddingGemma comparison | Separate local table `ipc_embeddings_gemma`. See `docs/retrieval-evaluation.md`. |


## Layout

```
apps/api     FastAPI + ML services + Alembic + CLI (Python 3.12, uv)
apps/web     Next.js 16 App Router UI (TypeScript, Tailwind) + Playwright tests
data/source  supplied originals (git-ignored)      data/manifests  hashes, import/validation/measurement reports
data/evaluation  QKD contract request, labelled-data schema     data/calibration  calibrator manifests
infra        sakthi non-secret handoff (sakthi-db.json), docker compose (local pgvector, app services)
docs         architecture, API, classification policy, model operation, retrieval evaluation, IP India benchmark
```

## Setup

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Node 20+, Docker Desktop, and
optionally psql. An NVIDIA GPU is optional; CPU works, but more slowly.

```bash
cp .env.example .env                       # then fill in secrets locally; never commit .env
# set SAKTHI_DB_PASSWORD in .env (IPC_DB_TARGET=sakthi is the default)

# Backend
cd apps/api
uv sync --extra ml --group dev
docker compose -f ../../infra/docker-compose.yml up -d     # local pgvector (PG17, pgvector 0.8.0)
uv run ipc --target local migrate
uv run ipc --target local validate-sources
uv run ipc --target local import --activate
uv run ipc --target local check-token-lengths
uv run ipc --target local embed --provider bge_m3          # resumable; ~10 min on an RTX 2050
uv run ipc --target local build-index --table ipc_embeddings --maintenance-work-mem 1GB
uv run ipc --target local activate-embeddings <run_id>
IPC_DB_TARGET=local uv run uvicorn app.main:app --port 8000

# Frontend
cd ../web && npm ci && npm run dev         # http://localhost:3000  (IPC_API_BASE_URL defaults to :8000)
```

Serving on sakthi uses the same commands with `--target sakthi`, which is the default. The URL is
built from `SAKTHI_DB_*` in `.env`, and the API runs with `uv run uvicorn app.main:app --port 8000`.

The sakthi server has `ssl=off`, so connections use `SAKTHI_DB_SSLMODE=disable`. This is an explicit
LAN-only exception. Once TLS is enabled on the server, set it to `require`. For any other remote
host the app defaults to `sslmode=require`.

`--target local` points at the local Docker pgvector database from `infra/docker-compose.yml`. It is
used for integration tests and storage and recall measurements.

## Tests

```bash
cd apps/api && uv run pytest -m "not integration"                  # unit
cd apps/api && IPC_DB_TARGET=local uv run pytest -m integration     # needs the imported local docker DB
cd apps/web && PW_CHANNEL=msedge npx playwright test               # browser flow (mocked contract)
cd apps/web && IPC_E2E_LIVE=1 PW_CHANNEL=msedge npx playwright test tests/live-qkd.spec.ts   # live API
```

On this machine, Windows Application Control blocks Playwright's downloaded browsers (`PW_CHANNEL`
selects the installed Edge or Chrome instead) and SQLAlchemy's compiled modules. `pyproject.toml`
therefore builds SQLAlchemy as pure Python.

**Memory:** the API (CUDA torch plus both models) commits about 5–7.5 GB. If Windows runs short of
commit memory (RAM plus page file), CUDA fails to initialise and `EMBEDDING_DEVICE=auto` falls back
to the CPU in fp32. That fallback now logs a warning, but it can then fail with "bad allocation".
Run the API with `EMBEDDING_DEVICE=cuda`, and stop the local Docker DB when it isn't needed, or
enlarge the page file.

## Storage requirement (measured)

Measured locally on PostgreSQL 17 with pgvector 0.8.0, using the real corpus plus the
`ipc --target local storage-probe` results in `data/manifests/storage_probe_local_pg17_pgvector080.jsonl`:

| Component | Size |
| --- | --- |
| Records, tree occurrences, references and indexes (no vectors) | 337 MB |
| float32 `vector(1024)` table + TOAST | 427 MiB |
| HNSW index on float32 (m=32/200 or m=64/800; the size is the same) | 626–628 MiB |
| **Total, float32 as specified** | **~1,392 MiB** (sakthi measured 1,375 MB) |
| Alternative: `halfvec(1024)` table + HNSW | 427 MiB in total, giving ~764 MiB overall |

## Security notes

- Secrets live only in the git-ignored `.env`. Connection URLs are never printed unredacted.
- The sakthi connection is unencrypted (the server has `ssl=off`); use it only on the trusted LAN.
- No submitted document text is stored unless `STORE_INPUT_DOCUMENTS=true` (or the older
  `STORE_ABSTRACTS=true`) is set.
- The LLM receives document text only when the operator enables it, and treats that text as
  untrusted data.
- Administration endpoints require `ADMIN_TOKEN`.
