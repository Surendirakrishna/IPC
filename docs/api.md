# API (v1)

The FastAPI service lives in `apps/api` and is started with `uv run uvicorn app.main:app --port 8000`.
OpenAPI is served at `/docs`. The web app reaches the API only through server-side route handlers
(`apps/web/app/api/*`), so the browser never sees the backend URL.

Every error has the same shape: `{"error": "<code>", "message": "<text>", "details": <any|null>}`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check. Loads no model and runs no database query. |
| `GET /ready` | 200 when models are loaded, the schema exists, a release is active and a validated embedding run is serving; otherwise 503 with `checks`. `checks.calibration` is reported separately, so an uncalibrated pipeline is still "ready". |
| `GET /v1/taxonomy/status` | Active release, counts, source hashes, source-quality warnings, embedding runs, HNSW indexes, model revisions, calibration state and keyword mode. |
| `GET /v1/ipc/search?q=&record_type=&section=&subclass=&limit=&offset=` | Code lookup (exact or prefix on internal or display code) or full-text search (`ts_rank_cd`). Paginated. |
| `GET /v1/ipc/{code}` | Accepts an internal or display code. Returns the record, original wording, parent path, children, duplicate contexts, every tree occurrence, outgoing references (with verbatim wording) and incoming references. |
| `POST /v1/classify` | See below. |
| `GET /v1/admin/jobs`, `POST /v1/admin/reload-confidence` | Need `Authorization: Bearer $ADMIN_TOKEN`; disabled (404) when it is unset. Import, embedding, index and calibration jobs are CLI-only. |

## POST /v1/classify

Request (`extra` fields are rejected):

| Field | Type | Notes |
| --- | --- | --- |
| `abstract` | string, **required** | Must be nonblank after trimming. Title + abstract must fit `ABSTRACT_MAX_TOKENS` (2,048); otherwise 422 `abstract_too_long`. |
| `title` | string \| null | Blank becomes absent. |
| `specification` | string \| null | Plain text; blank becomes absent. |
| `claims` | string \| null | Plain text with the original numbering; blank becomes absent. Arrays are rejected. |
| `taxonomy_version` | string \| null | Must be the active serving release. |
| `target_level` | `group` (default) \| `maingroup` \| `subgroup` \| `subclass` \| `class` | |
| `max_results` | int, default 5 | A ceiling (≤ `MAX_RESULTS_LIMIT`=20), not a quota. |
| `use_llm` | bool, default false | Takes effect only when the operator set `LLM_ENABLED=true`. |
| `section`, `subclass` | optional filters | Applied to both keyword and vector retrieval. |

Response (`ClassifyResponse`, [schemas/classify.py](../apps/api/app/schemas/classify.py)):

- `status`: `candidates_found` | `needs_review` | `insufficient_evidence`.
- `field_of_invention`: `{label, mapping_key, mapping_version, additional_labels, rationale}`. The
  label comes from the class of the primary candidate via
  [field_mapping.json](../apps/api/app/services/fields/field_mapping.json) (`H04 → COMMUNICATION`).
  Without a mapping it is the class title in uppercase; with no candidate it is `UNDETERMINED`.
- `summary`: the **single shared formatting contract** for the two-row table. It contains
  `classification_lines` (one per supported code) and a Markdown string with pipes and newlines
  escaped. The UI renders the lines as an HTML table and never injects the Markdown.
- `candidates[]`, the supported selections only. Each one has:
  - `code`, `display_code`, `title`, `original_title` and `level`;
  - `confidence {score|null, status, calibrator_id, interpretation, detail}`;
  - `supporting_passages[] {source, quote, start, end, claim_number, evidence_type, relevance}`;
  - `rationale` and `unresolved_ambiguity[]`;
  - `parent_path[]` and `references[]`;
  - `scores {cosine_similarity, keyword_rank, keyword_score, vector_rank, rrf_score, fused_rank, reranker_score, reranker_source}`;
  - `origin[]`.
- `alternatives[]`: tentative codes, each with a `reason`. They never appear in the summary cell.
- `input_coverage`: `{abstract, specification, claims}`, each one of `complete`, `partial`,
  `not_supplied` or `empty`, plus chunk counts.
- `extracted_terms`, `warnings`, `degraded_mode`, `provenance` (snapshot hashes, model revisions,
  embedding run, scope formatter, calibrator, mapping version, retrieval settings, encoding policy)
  and `timings_ms`.

Passage offsets are Unicode code-point indices. They are start-inclusive and end-exclusive, and are
relative to the original field named by `source`. The server checks
`field[start:end] == quote`. The frontend converts the offsets to UTF-16 before highlighting
([lib/text.ts](../apps/web/lib/text.ts)).

`confidence.score` is non-null only with an active, compatible, evaluated calibrator. The other
statuses are `not_calibrated`, `incompatible_artifact` and `unsupported_input_mode`. Retrieval
scores are relevance values, never probabilities.

Error codes:

| Code | HTTP status |
| --- | --- |
| `invalid_abstract` | 422 |
| `invalid_request` | 422 |
| `abstract_too_long` | 422 |
| `invalid_taxonomy_version` | 422 |
| `invalid_max_results` | 422 |
| `request_too_large` | 413 |
| `models_unavailable`, `corpus_unavailable`, `embeddings_unavailable`, `embeddings_incompatible`, `database_unavailable`, `busy` | 503 |
| `timeout` | 504 |

The API never serves a canned classification. The illustrative response in the brief exists only as
a test fixture (`apps/web/tests/fixtures.ts`, `origin: ["contract_fixture"]`).

## Example

```bash
curl -s localhost:8000/v1/classify -H 'content-type: application/json' \
  --data @data/evaluation/qkd_contract_request.json | jq .summary.markdown
```
