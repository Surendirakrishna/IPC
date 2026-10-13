# ipc-classifier

ipc-classifier classifies a patent abstract into IPC codes, with evidence for each code. You can also
supply the specification, claims and title, but they are optional. Two classifiers share one pipeline:

**Neighbour vote** (primary when `NEIGHBOUR_VOTE_ENABLED=true`; off by default):
- encodes "title. abstract" with PaECTER and finds the 100 most similar already-classified patents in
  a labelled corpus on PostgreSQL (`halfvec(1024)`, one pgvector HNSW index per corpus);
- lets their IPC codes vote, weighted by similarity; each code's vote share is the weighted share of
  those patents that carry it;
- reports the vote's codes in vote order, and falls back to definition matching when the vote is weak
  (first-code subclass share below 0.2) or cannot run.

**Definition matching** (explains the vote's codes; the classifier when the vote is off or falls back):
- routes to likely subclasses first, then runs PostgreSQL full-text search and BGE-M3 vector search
  (pgvector HNSW) for groups inside them, over the WIPO IPC 2026.01 scheme with its real dotted hierarchy;
- fuses the two result lists with reciprocal rank fusion and expands them with parents and siblings
  (a cross-reference never adds a candidate);
- reranks candidates with BGE-reranker-v2-m3, then applies WIPO precedence rules between supported
  candidates (with the vote, between the vote's codes only; a rule never adds a code);
- quotes supporting passages with exact offsets and lists unresolved ambiguity;
- applies an optional LLM step that can only choose from the candidate codes (not with the vote).

Both map the result to a field of invention. Each response says which classifier chose the codes
(`classifier`) and why the vote did not (`fallback_reason`). See
[docs/classification-policy.md](docs/classification-policy.md).

The first output is always the two-row summary:

| Field Of Invention | COMMUNICATION |
| :--- | :--- |
| Classification (IPC) | H04L 9/08 — vote share: 87.9%; H04L 9/14 — vote share: 34.1%; H04L 9/32 — vote share: 22.1% |

The table above is the **live result for the QKD abstract-only request** in
`data/evaluation/qkd_contract_request.json`, with the neighbour vote on over the `hupd-2017` corpus
(2026-10-14; `classifier: neighbour_vote`, HTTP 200 in 4.0 s). It is not a hard-coded value.

- **H04L 9/08** "Key distribution": 97.7% of the similarity-weighted 100 nearest patents carry H04L,
  92.5% H04L 9/00, and **87.9%** exactly H04L 9/08. The nearest are 2017 US filings on quantum key
  distribution and quantum communication (similarity about 0.96).
- **H04L 9/14** (34.1%) and **H04L 9/32** (22.1%) follow. H04B 10/70 "Photonic quantum communication"
  has 16.1%, below the 0.2 minimum share, so it is not listed.
- Every code carries review notes, for example the WIPO last-place rule for H04L 9/06–9/32, which is
  reported but not applied.

With the vote off (the default), definition matching returns H04W 12/0433 "Key management
protocols", with a review note that H04W's "wireless" scope isn't evidenced
(`data/manifests/acceptance_qkd_sakthi_wipo2026.json`, 2026-10-11).

A vote share is **not a calibrated probability**. The summary says "vote share", and the UI adds
"(uncalibrated)". No calibrator has been fitted, for vote shares or reranker scores; definition-path
codes show "confidence: not calibrated". No invented percentages are shown.

## Status (2026-10-14)

| Area | State |
| --- | --- |
| Source validation | ✅ The hashes match the expected snapshot and every count matches (80,145 records, 589 duplicate main groups, 15,035 reference edges, 0 blocking findings). Manifest: `data/manifests/validation_20260101.json` |
| Local corpus (docker pgvector, tests only) | ✅ Imported and activated. BGE-M3 run `emb-bge_m3-72c8b3da25a4` has 80,145/80,145 vectors, is validated, has its HNSW index (m=64, ef_construction=800; recall@100 0.955 against exact search) and is serving. |
| **sakthi DB (serving)** | ✅ The default target since 2026-10-10: ShaktiDB 18.4 (PostgreSQL 18.4), pgvector 0.8.2, `ipc_db` on 192.168.200.29:11233. The catalogue is imported and active. Migrated to Alembic `0003` and re-imported with the WIPO 2026.01 master files on 2026-10-11 (real dotted hierarchy, notes with owner/range/edition and rule meaning). BGE-M3 run `emb-bge_m3-61c53c85a23d` (dotted-path text) is float32 and validated, with 80,145/80,145 vectors. HNSW m=64/ef_construction=800 was built in 105 s (626 MiB) and gives recall@100 of 0.961 against exact search (ef_search 500). The pre-rebuild vectors, text and run are kept in `rollback_20261011_*` tables. The QKD request returns 200 in about 14 s (`data/manifests/acceptance_qkd_sakthi_wipo2026.json`). The VPN link to sakthi has dropped several times; benchmark runs pause until it is back. |
| **Neighbour vote** | ✅ Implemented, **off by default**. Corpus `hupd-2017` is loaded on sakthi, validated and serving: 293,685 USPTO applications filed in 2017, PaECTER `halfvec(1024)`, partial HNSW m=32/ef_construction=200 built in 545 s (749.6 MiB), recall@100 **0.9884** against exact search. sakthi grew from 2.53 GB to 4.46 GB. Manifests: `data/manifests/patents/*_hupd-2017.json`. To switch it on, see [docs/model-operation.md](docs/model-operation.md#switching-it-on). ⚠ The USPTO grants reader is unverified (no API key yet) and has an open paging must-fix. |
| API / UI / CLI | ✅ Implemented. 792 backend unit tests pass (2026-10-14; 11 skipped). The neighbour-vote integration test ran on sakthi against a synthetic `it-test-*` corpus (2026-10-14). Database integration tests otherwise need the local docker DB. Playwright: 10 mocked-contract browser tests in `classify.spec.ts` and the live QKD test in `live-qkd.spec.ts` (counts from the spec files; not re-run for this update). |
| **Benchmark, neighbour vote** (IP India, 123 held-out applications) | Scored once, live, 2026-10-14, abstract-only: first code matches a recorded subclass **63.4%** (54.6–71.4), main group **44.7%** (36.2–53.5), exact **30.1%** (22.7–38.7). Field label 56.1%. Never abstains; 124 of 125 responses from the vote, 1 fallback. Median **1.6 s**. Dev: 71.5 / 59.2 / 32.3%. The offline experiment gave 65.9 / 48.0 / 31.7% on the same test set. See [docs/benchmark-ipindia.md](docs/benchmark-ipindia.md#neighbour-vote-served-live-held-out-test-2026-10-14). |
| Benchmark, definition matching (IP India, 123 held-out applications) | Re-scored once after the rebuild (2026-10-12), abstract-only: answers 97.6%; first code matches a recorded subclass 30.9%, main group 17.9%, exact 8.9% (previous pipeline 28.5 / 17.1 / 8.1, within the CIs). A recorded symbol is in the top 100 candidates for 78.0 / 59.3 / **44.7%** (previously 71.5 / 48.0 / 26.8). With claims as evidence only: 29.3 / 16.3 / 8.1 (previously, with claims voting: 27.6 / 10.6 / 2.4). Median 9.5 s. These candidate-recall figures cover about 100 candidates and cannot be compared with the vote's, which counts only the ≤ 5 returned codes. See [docs/benchmark-ipindia.md](docs/benchmark-ipindia.md). |
| Benchmark, definition matching, dev set after the WIPO 2026.01 rebuild (2026-10-11) | Dev only (130 applications, used for design decisions). With subclass-first retrieval and no WIPO notes in the reranker, the first code matches a recorded subclass 42.3%, main group 24.6%, exact 11.5% (old flat v2 on the same set: 37.7 / 22.3 / 10.8). The correct code is in the top 100 candidates for 83.1 / 66.2 / 50.0%. Median 10.1 s per document. See [Dev-set results](docs/benchmark-ipindia.md#dev-set-results-after-the-wipo-202601-rebuild-2026-10-11). |
| Review and reranker-training tooling | ✅ `ipc families` (patent-family grouping, family-safe splits; the current dev and test sets share no family), `ipc review-queue` / `review-finalize` (blinded expert review, then expert-labelled JSONL), `ipc mine-negatives` (hard negatives from nearby parts of the scheme). See [docs/model-operation.md](docs/model-operation.md#review-family-splits-and-reranker-training-data). ⛔ No reviewed labels yet, so no reranker has been trained. |
| Calibration | Harness ✅. Fitted calibrator ⛔: no expert-labelled data, for reranker scores or vote shares. Vote codes show the raw vote share ("vote share: 87.9%", "(uncalibrated)" in the UI). The review tooling above produces the input a calibrator needs. |
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

**Neighbour vote (optional, off by default).** It needs a validated patent corpus serving on the
target database (`uv run ipc --target sakthi patent-status`; `hupd-2017` serves on sakthi). Add to
`.env`, then restart the API:

```
NEIGHBOUR_VOTE_ENABLED=true
PATENT_ENCODER_LOCAL_PATH=D:/ipc_knn/models/paecter
```

`/ready` must then show `checks.neighbour_vote.state: "serving"`. Loading a corpus (`load-patents`,
`activate-patent-corpus`) and the USPTO key (`USPTO_ODP_API_KEY` in `.env`) are described in
[docs/model-operation.md](docs/model-operation.md#patent-corpus-lifecycle).

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

**Memory:** the API (CUDA torch plus BGE-M3 and the reranker) commits about 5–7.5 GB. With the
neighbour vote on, PaECTER is a third model: about **8.3 GiB peak commit** (measured 2026-10-14). If
Windows runs short of commit memory (RAM plus page file), CUDA fails to initialise and
`EMBEDDING_DEVICE=auto` falls back to the CPU in fp32. That fallback now logs a warning, but it can
then fail with "bad allocation". Run the API with `EMBEDDING_DEVICE=cuda`, and stop the local Docker
DB when it isn't needed, or enlarge the page file.

On a cold start with the vote on, PaECTER can fail with **OS error 1455** ("The paging file is too
small"). The API still answers with definition matching, and `/ready` shows the error in
`checks.neighbour_vote.patent_encoder_error`. Restart the API once; the retry has worked, because
Windows grows a system-managed page file after the failure. Details:
[docs/model-operation.md](docs/model-operation.md#memory).

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

The labelled patent corpus for the neighbour vote, measured on sakthi
(`data/manifests/patents/db_size_hupd-2017.json`, `pg_database_size`):

| hupd-2017 (293,685 patents) | Size |
| --- | --- |
| Documents, labels and `halfvec(1024)` vectors (ingest + embed) | 1,149 MB |
| Partial HNSW index (m=32, ef_construction=200) | 786 MB (749.6 MiB) |
| **sakthi database, before → after** | **2.53 GB → 4.46 GB** |

## Security notes

- Secrets live only in the git-ignored `.env`. Connection URLs are never printed unredacted.
- The sakthi connection is unencrypted (the server has `ssl=off`); use it only on the trusted LAN.
- No submitted document text is stored unless `STORE_INPUT_DOCUMENTS=true` (or the older
  `STORE_ABSTRACTS=true`) is set.
- The LLM receives document text only when the operator enables it, and treats that text as
  untrusted data.
- Administration endpoints require `ADMIN_TOKEN`.
