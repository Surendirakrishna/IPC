# API (v1)

The FastAPI service lives in `apps/api` and is started with `uv run uvicorn app.main:app --port 8000`.
OpenAPI is served at `/docs`. The web app reaches the API only through server-side route handlers
(`apps/web/app/api/*`), so the browser never sees the backend URL.

Every error has the same shape: `{"error": "<code>", "message": "<text>", "details": <any|null>}`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check. Loads no model and runs no database query. |
| `GET /ready` | 200 when models are loaded, the schema exists, a release is active and a validated embedding run is serving; otherwise 503 with `checks`. `checks.calibration` is reported separately, so an uncalibrated pipeline is still "ready". `checks.neighbour_vote` is also reported separately and never makes the API unready: without a serving patent corpus the definition fallback still serves (see below). |
| `GET /v1/taxonomy/status` | Active release, counts, source hashes, source-quality warnings, embedding runs, HNSW indexes, model revisions, calibration state, keyword mode and `neighbour_vote` (same shape as in `/ready`). |
| `GET /v1/ipc/search?q=&record_type=&section=&subclass=&limit=&offset=` | Code lookup (exact or prefix on internal or display code) or full-text search (`ts_rank_cd`). Paginated. |
| `GET /v1/ipc/{code}` | Accepts an internal or display code. Returns the record, original wording, parent path, children, duplicate contexts, every tree occurrence, outgoing references (with verbatim wording) and incoming references. |
| `POST /v1/classify` | See below. |
| `GET /v1/admin/jobs`, `POST /v1/admin/reload-confidence` | Need `Authorization: Bearer $ADMIN_TOKEN`; disabled (404) when it is unset. Import, embedding, index and calibration jobs are CLI-only. |

`checks.neighbour_vote` (in `/ready`) and `neighbour_vote` (in `/v1/taxonomy/status`):
`{enabled: bool, state, patent_encoder, patent_corpus, vote_settings}`. `state` is `serving`,
`disabled`, `no_serving_corpus` or `patent_encoder_unavailable` (not loaded, or its revision differs
from the corpus's). `patent_encoder`, `patent_corpus` and `vote_settings` have the shapes of the
`provenance` fields of the same names below, or null. `patent_corpus` additionally carries `state`
(`loading` | `embedded` | `validated` | `failed`) here.

## POST /v1/classify

Response contract **2.0.0** (2026-10-13). The endpoint path stays `/v1`; 2.0.0 adds fields and one
confidence status, removes nothing and renames nothing. Prose rationale, client guidance and the list
of breaking meaning changes: vault note `10-Projects/ipc/02-Contracts/classify-response-v2.md`.

Request (`extra` fields are rejected):

| Field | Type | Notes |
| --- | --- | --- |
| `abstract` | string, **required** | Must be nonblank after trimming. Title + abstract must fit `ABSTRACT_MAX_TOKENS` (2,048); otherwise 422 `abstract_too_long`. |
| `title` | string \| null, optional | Blank becomes absent. |
| `specification` | string \| null, optional | Plain text; blank becomes absent. |
| `claims` | string \| null, optional | Plain text with the original numbering; blank becomes absent. Arrays are rejected. |
| `taxonomy_version` | string \| null, optional | Must be the active serving release. |
| `target_level` | `group` (default) \| `maingroup` \| `subgroup` \| `subclass` \| `class`, optional | |
| `max_results` | int ≥ 1, optional, default 5 | A ceiling (≤ `MAX_RESULTS_LIMIT`=20), not a quota. |
| `use_llm` | bool, optional, default false | Takes effect only when the operator set `LLM_ENABLED=true`. |
| `include_ranked_debug` | bool, optional, default false | Evaluation diagnostics: fills `ranked_debug` and `retrieval_debug`. |
| `section`, `subclass` | string \| null, optional | Filters (`^[A-H]$`, `^[A-H]\d{2}[A-Z]$`) applied to keyword and vector retrieval. |

### Response (`ClassifyResponse`, [schemas/classify.py](../apps/api/app/schemas/classify.py))

Every response field is always present; "nullable" means the value can be `null`, never that the key
is missing.

- `status`: `candidates_found` | `needs_review` | `insufficient_evidence`. `candidates_found` needs
  every candidate calibrated (`confidence.status == "calibrated"`) and free of unresolved ambiguity,
  so a raw vote share gives `needs_review`. `insufficient_evidence` only occurs with the definition
  fallback.
- `classifier` (**2.0**, required): `neighbour_vote` | `definition_fallback`. Which component chose
  the codes in `candidates`.
- `fallback_reason` (**2.0**, nullable): null exactly when `classifier` is `neighbour_vote`. Otherwise
  `neighbour_vote_disabled` | `no_serving_corpus` | `patent_encoder_unavailable` | `vote_empty` |
  `vote_below_minimum` | `vote_failed`.
- `field_of_invention`: `{label, mapping_key, mapping_version, additional_labels, rationale}`. The
  label comes from the class of the primary candidate via
  [field_mapping.json](../apps/api/app/services/fields/field_mapping.json) (`H04 → COMMUNICATION`).
  Without a mapping it is the class title in uppercase; with no candidate it is `UNDETERMINED`.
- `summary`: the **single shared formatting contract** for the two-row table. It contains
  `classification_lines` (one per candidate) and a Markdown string with pipes and newlines
  escaped. The UI renders the lines as an HTML table and never injects the Markdown.
  `confidence_text` by `confidence.status`:

  | `confidence.status` | `confidence_text` |
  | --- | --- |
  | `calibrated` | `f"{round(100 * score, 1)}%"`, e.g. `89.3%` |
  | `vote_share_uncalibrated` | `f"vote share: {round(100 * score, 1)}%"`, e.g. `vote share: 79.1%` |
  | `not_calibrated` | `confidence: not calibrated` |
  | `incompatible_artifact` | `confidence: calibrator incompatible` |
  | `unsupported_input_mode` | `confidence: not calibrated for this input mode` |

- `candidates[]`, in selection order (vote order when `classifier` is `neighbour_vote`). With the
  neighbour vote the list is never empty, and `candidates[0].display_code == vote.first_code`. The one
  exception is when an unconditional precedence rule demoted the first code: it then appears in
  `alternatives` and the supported code it yields to leads. Each candidate has:
  - `code`, `display_code`, `title`, `original_title` (nullable) and `level`;
  - `confidence {score|null, status, calibrator_id|null, interpretation|null, detail|null}`;
  - `vote_share {subclass|null, maingroup|null, exact|null}` (**2.0**), see below;
  - `supporting_passages[] {source, quote, start, end, claim_number|null, section|null, evidence_type, relevance|null}`.
    With the neighbour vote this list may be empty: evidence explains a vote code and never removes
    it; a code without a passage carries a review note in `unresolved_ambiguity`;
  - `depends_on_optional_documents`, `rationale`, `unresolved_ambiguity[]`, `missing_defining_terms[]`;
  - `parent_path[]` and `references[]`;
  - `scores {cosine_similarity, keyword_rank, keyword_score, vector_rank, rrf_score, fused_rank, reranker_score, reranker_source}` (all nullable);
  - `origin[]`: contains `neighbour_vote` for vote codes.
- `alternatives[]`: tentative codes, each with a `reason`, `confidence` and (**2.0**) `vote_share`.
  They never appear in the summary cell.
- `vote` (**2.0**, nullable): the neighbour vote as computed. Non-null exactly when `classifier` is
  `neighbour_vote` or `fallback_reason` is `vote_below_minimum`.
  - `first_code` (display code) and `first_code_level` (`subclass` | `maingroup` | `subgroup`):
    chosen top-down, best subclass, then its best main group, then that main group's best code.
  - `first_code_share`: a `vote_share` object; its `subclass` value is the one compared with
    `provenance.vote_settings.min_vote_share`.
  - `top_codes {subclass[], maingroup[], exact[]}`: each a list of `{display_code, share}`, descending
    share, at most 10 (`MAX_TOP_CODES_PER_LEVEL`).
  - `neighbour_count`: neighbours that entered the vote (≤ `k`; fewer only if the corpus is smaller).
  - `neighbours[]`: the min(10, `neighbour_count`) nearest corpus patents (`MAX_NEIGHBOURS` = 10),
    descending similarity: `{document_id, title|null, similarity, vote_weight, ipc_codes[]}`.
    `similarity` is cosine similarity (PaECTER values are compressed, typically 0.95–0.99), not a
    probability. `vote_weight` is that neighbour's normalised weight in the vote (over all `k`
    neighbours the weights sum to 1). `ipc_codes` are resolved display codes, main label first.
- `input_coverage`: `{abstract, specification, claims}`, each one of `complete`, `partial`,
  `not_supplied` or `empty`, plus chunk counts in `details`.
- `extracted_terms[] {term, source, weight}`, `warnings[]`, `degraded_mode` (nullable), `timings_ms`
  (the vote path adds `patent_encode`, `patent_search` and `vote`).
- `provenance`: snapshot hashes, model revisions, embedding run, scope formatter, calibrator, mapping
  version, retrieval settings, encoding policy, and (**2.0**, each nullable):
  - `patent_encoder {model_id, model_revision}`: null when the vote is disabled or the encoder is not loaded;
  - `patent_corpus {corpus_id, source, document_count, labels_ipc_edition|null}`: null when no corpus
    is serving; `source` is `hupd` or `uspto_odp`;
  - `vote_settings {k, temperature, beta, min_vote_share, hnsw_ef_search|null}`: null when the vote is disabled.

  All three are non-null whenever `vote` is non-null.
- `ranked_debug`, `retrieval_debug` (nullable; only with `include_ranked_debug`). When a vote ran,
  `retrieval_debug` adds `vote_subclass`, `vote_maingroup`, `vote_exact` (display codes in vote order,
  at most 50 each) and `vote_neighbour_ids` (all neighbour ids, descending similarity).

Passage offsets are Unicode code-point indices. They are start-inclusive and end-exclusive, and are
relative to the original field named by `source`. The server checks
`field[start:end] == quote`. The frontend converts the offsets to UTF-16 before highlighting
([lib/text.ts](../apps/web/lib/text.ts)).

### Vote share and confidence

`vote_share` is the similarity-weighted share of the `k` nearest labelled corpus patents that carry
the code, in [0, 1]. A neighbour's weight is `exp((s − s₁) / T)` (s₁ is the nearest similarity); its
main label counts 1 and secondary labels count `beta`; each neighbour counts once per code per level.
Patents carry several codes, so shares of different codes do not sum to 1.

| Level | Counts neighbours carrying… | Null when |
| --- | --- | --- |
| `subclass` | any code in the candidate's subclass | no vote shares reported (`classifier` is `definition_fallback`) |
| `maingroup` | any code in the candidate's main group | as above, or the candidate is a subclass |
| `exact` | exactly this code (a main group counts neighbours labelled with that main group itself) | as above, or the candidate is a subclass |

For one code, `subclass ≥ maingroup ≥ exact`. `0.0` is a real share. Values are rounded to 4 decimals.

`confidence.score` is a **calibrated probability only when `confidence.status` is `calibrated`**
(`calibrator_id` names the calibrator, which is tied to the encoder and corpus through the
fingerprint). With `vote_share_uncalibrated`, `score` is the raw vote share at the candidate's own
level (`subclass` → `vote_share.subclass`, `maingroup` → `vote_share.maingroup`, `subgroup` →
`vote_share.exact`) and `calibrator_id` is null. With the neighbour vote, the status is always
`calibrated` or `vote_share_uncalibrated` and `score` is never null. With the definition fallback,
`score` is non-null only with an active, compatible, evaluated calibrator; the other statuses are
`not_calibrated`, `incompatible_artifact` and `unsupported_input_mode` (score null). Retrieval
scores are relevance values, never probabilities.

### Errors

| Code | HTTP status |
| --- | --- |
| `invalid_abstract` | 422 |
| `invalid_request` | 422 |
| `abstract_too_long` | 422 |
| `invalid_taxonomy_version` | 422 |
| `invalid_max_results` | 422 |
| `request_too_large` | 413 |
| `models_unavailable`, `corpus_unavailable`, `embeddings_unavailable`, `embeddings_incompatible`, `database_unavailable`, `busy` | 503 |
| `patent_corpus_unavailable` (**2.0**) | 503 |
| `timeout` | 504 |

`corpus_unavailable` still means the IPC taxonomy release, not the patent corpus.
`patent_corpus_unavailable` means the neighbour vote could not run **and** the definition fallback
could not serve the request either. Its `details` is
`{"reason": "no_serving_corpus" | "patent_encoder_unavailable" | "vote_empty" | "vote_failed", "fallback_error": "<the definition path's own error code>"}`.
When the vote is disabled, or it ran but was weak, and the definition path fails, the definition
path's own error code is returned, as in 1.x. A missing patent corpus alone is never an error: the
response is 200 with `classifier: "definition_fallback"`.

The API never serves a canned classification. The illustrative responses exist only as test fixtures
(`apps/web/tests/fixtures.ts`: `fixtureResponse()` / `fixtureFallbackResponse()`,
`fixtureVoteResponse()`, `fixtureWeakVoteResponse()`, `FIXTURE_PATENT_CORPUS_UNAVAILABLE`).

## Example

```bash
curl -s localhost:8000/v1/classify -H 'content-type: application/json' \
  --data @data/evaluation/qkd_contract_request.json | jq .summary.markdown
```

Worked example (neighbour vote primary). Validated with `ClassifyRequest.model_validate` and
`ClassifyResponse.model_validate`. Values are illustrative: the patent ids and titles are invented,
but the shares, weights and first code follow from the listed vote rule, so the numbers agree with
each other.

<!-- contract-example: classify-request -->
```json
{
  "abstract": "Quantum keys are distributed among smart grid nodes. The system comprises smart meters, substations, inverters, chargers, and control centers.",
  "title": "Quantum key distribution in a smart grid",
  "max_results": 2
}
```

<!-- contract-example: classify-response -->
```json
{
  "status": "needs_review",
  "taxonomy_version": "20260101",
  "classifier": "neighbour_vote",
  "fallback_reason": null,
  "field_of_invention": {
    "label": "COMMUNICATION",
    "mapping_key": "H04",
    "mapping_version": "2",
    "additional_labels": [],
    "rationale": "Primary candidate H04L 9/08 is in class H04 (COMMUNICATION)."
  },
  "summary": {
    "field_of_invention": "COMMUNICATION",
    "classification_lines": [
      {"display_code": "H04L 9/08", "confidence_text": "vote share: 79.1%"},
      {"display_code": "H04L 9/32", "confidence_text": "vote share: 47.9%"}
    ],
    "classification_cell_text": "H04L 9/08 — vote share: 79.1%; H04L 9/32 — vote share: 47.9%",
    "markdown": "| Field Of Invention | COMMUNICATION |\n| :--- | :--- |\n| Classification (IPC) | H04L 9/08 — vote share: 79.1%; H04L 9/32 — vote share: 47.9% |\n"
  },
  "candidates": [
    {
      "code": "H04L0009080000",
      "display_code": "H04L 9/08",
      "title": "Key distribution",
      "original_title": "Key distribution",
      "level": "subgroup",
      "confidence": {
        "score": 0.7907,
        "status": "vote_share_uncalibrated",
        "calibrator_id": null,
        "interpretation": "Raw neighbour vote share: the similarity-weighted fraction of the nearest labelled patents that carry this code. Not a calibrated probability.",
        "detail": null
      },
      "vote_share": {"subclass": 0.9551, "maingroup": 0.8865, "exact": 0.7907},
      "supporting_passages": [
        {
          "source": "abstract",
          "quote": "Quantum keys are distributed among smart grid nodes.",
          "start": 0,
          "end": 52,
          "claim_number": null,
          "section": null,
          "evidence_type": "retrieval_evidence_requires_review",
          "relevance": 1.2034
        }
      ],
      "depends_on_optional_documents": false,
      "rationale": "Neighbour vote: 95.5% of the similarity-weighted 100 nearest patents carry H04L, 88.6% H04L 9/00 and 79.1% exactly H04L 9/08. The abstract passage shares its defining terms (keys, distributed).",
      "unresolved_ambiguity": [
        "Abstract-only candidate review; applicable scope notes and priority rules still need verification against the official scheme."
      ],
      "missing_defining_terms": [],
      "parent_path": [
        {"code": "H", "display_code": "H", "title": "ELECTRICITY", "level": "section"},
        {"code": "H04", "display_code": "H04", "title": "ELECTRIC COMMUNICATION TECHNIQUE", "level": "class"},
        {
          "code": "H04L",
          "display_code": "H04L",
          "title": "TRANSMISSION OF DIGITAL INFORMATION, e.g. TELEGRAPHIC COMMUNICATION",
          "level": "subclass"
        },
        {
          "code": "H04L0009000000",
          "display_code": "H04L 9/00",
          "title": "Arrangements for secret or secure communications; Network security protocols",
          "level": "maingroup"
        }
      ],
      "references": [],
      "scores": {
        "cosine_similarity": null,
        "keyword_rank": null,
        "keyword_score": null,
        "vector_rank": null,
        "rrf_score": null,
        "fused_rank": null,
        "reranker_score": 1.2034,
        "reranker_source": "abstract"
      },
      "origin": ["neighbour_vote"]
    },
    {
      "code": "H04L0009320000",
      "display_code": "H04L 9/32",
      "title": "including means for verifying the identity or authority of a user of the system",
      "original_title": null,
      "level": "subgroup",
      "confidence": {
        "score": 0.4787,
        "status": "vote_share_uncalibrated",
        "calibrator_id": null,
        "interpretation": "Raw neighbour vote share: the similarity-weighted fraction of the nearest labelled patents that carry this code. Not a calibrated probability.",
        "detail": null
      },
      "vote_share": {"subclass": 0.9551, "maingroup": 0.8865, "exact": 0.4787},
      "supporting_passages": [],
      "depends_on_optional_documents": false,
      "rationale": "Neighbour vote: 88.6% of the similarity-weighted 100 nearest patents carry H04L 9/00 and 47.9% exactly H04L 9/32.",
      "unresolved_ambiguity": [
        "No supporting passage was found in the input for this code; it comes from the neighbour vote and needs review."
      ],
      "missing_defining_terms": [],
      "parent_path": [
        {"code": "H", "display_code": "H", "title": "ELECTRICITY", "level": "section"},
        {"code": "H04", "display_code": "H04", "title": "ELECTRIC COMMUNICATION TECHNIQUE", "level": "class"},
        {
          "code": "H04L",
          "display_code": "H04L",
          "title": "TRANSMISSION OF DIGITAL INFORMATION, e.g. TELEGRAPHIC COMMUNICATION",
          "level": "subclass"
        },
        {
          "code": "H04L0009000000",
          "display_code": "H04L 9/00",
          "title": "Arrangements for secret or secure communications; Network security protocols",
          "level": "maingroup"
        }
      ],
      "references": [],
      "scores": {
        "cosine_similarity": null,
        "keyword_rank": null,
        "keyword_score": null,
        "vector_rank": null,
        "rrf_score": null,
        "fused_rank": null,
        "reranker_score": null,
        "reranker_source": null
      },
      "origin": ["neighbour_vote"]
    }
  ],
  "alternatives": [
    {
      "code": "H02J0013000000",
      "display_code": "H02J 13/00",
      "title": "Circuit arrangements for providing remote indication of network conditions or remote control of switching means in a power distribution network",
      "level": "maingroup",
      "reason": "Beyond max_results ceiling.",
      "unresolved_ambiguity": [],
      "supporting_passages": [],
      "confidence": {
        "score": 0.1455,
        "status": "vote_share_uncalibrated",
        "calibrator_id": null,
        "interpretation": "Raw neighbour vote share: the similarity-weighted fraction of the nearest labelled patents that carry this code. Not a calibrated probability.",
        "detail": null
      },
      "vote_share": {"subclass": 0.1809, "maingroup": 0.1455, "exact": 0.1455},
      "scores": {
        "cosine_similarity": null,
        "keyword_rank": null,
        "keyword_score": null,
        "vector_rank": null,
        "rrf_score": null,
        "fused_rank": null,
        "reranker_score": null,
        "reranker_source": null
      }
    }
  ],
  "vote": {
    "first_code": "H04L 9/08",
    "first_code_level": "subgroup",
    "first_code_share": {"subclass": 0.9551, "maingroup": 0.8865, "exact": 0.7907},
    "top_codes": {
      "subclass": [
        {"display_code": "H04L", "share": 0.9551},
        {"display_code": "H02J", "share": 0.1809},
        {"display_code": "G06Q", "share": 0.1352},
        {"display_code": "H04W", "share": 0.0459},
        {"display_code": "H04B", "share": 0.0442}
      ],
      "maingroup": [
        {"display_code": "H04L 9/00", "share": 0.8865},
        {"display_code": "H02J 13/00", "share": 0.1455},
        {"display_code": "G06Q 50/00", "share": 0.1352},
        {"display_code": "H04L 67/00", "share": 0.1068},
        {"display_code": "H04W 12/00", "share": 0.0459}
      ],
      "exact": [
        {"display_code": "H04L 9/08", "share": 0.7907},
        {"display_code": "H04L 9/32", "share": 0.4787},
        {"display_code": "H02J 13/00", "share": 0.1455},
        {"display_code": "G06Q 50/06", "share": 0.1352},
        {"display_code": "H04L 67/12", "share": 0.1068}
      ]
    },
    "neighbour_count": 100,
    "neighbours": [
      {
        "document_id": "15300001",
        "title": "Quantum key distribution among nodes of a power distribution network",
        "similarity": 0.9812,
        "vote_weight": 0.3442,
        "ipc_codes": ["H04L 9/08", "H04L 9/32"]
      },
      {
        "document_id": "15300002",
        "title": "Secure key exchange between smart meters and a head-end system",
        "similarity": 0.9751,
        "vote_weight": 0.1016,
        "ipc_codes": ["H04L 9/08", "G06Q 50/06"]
      },
      {
        "document_id": "15300003",
        "title": "Encrypted communication between substations of a utility grid",
        "similarity": 0.9723,
        "vote_weight": 0.0581,
        "ipc_codes": ["H04L 9/08", "H02J 13/00"]
      },
      {
        "document_id": "15300004",
        "title": "Authentication of grid edge devices",
        "similarity": 0.9702,
        "vote_weight": 0.0381,
        "ipc_codes": ["H04L 9/32", "H04L 67/12"]
      },
      {
        "document_id": "15300005",
        "title": "Quantum random number generation for key management",
        "similarity": 0.9688,
        "vote_weight": 0.0288,
        "ipc_codes": ["H04L 9/08"]
      },
      {
        "document_id": "15300006",
        "title": "Charging station communication security",
        "similarity": 0.9671,
        "vote_weight": 0.0205,
        "ipc_codes": ["H04L 9/32", "H02J 7/00"]
      },
      {
        "document_id": "15300007",
        "title": "Remote control of substation switchgear over a secured link",
        "similarity": 0.9663,
        "vote_weight": 0.0175,
        "ipc_codes": ["H02J 13/00", "H04L 9/08"]
      },
      {
        "document_id": "15300008",
        "title": "Key rotation for distributed energy resources",
        "similarity": 0.9655,
        "vote_weight": 0.0149,
        "ipc_codes": ["H04L 9/08", "H02J 3/38"]
      },
      {
        "document_id": "15300009",
        "title": "Optical fibre quantum key distribution network",
        "similarity": 0.9649,
        "vote_weight": 0.0132,
        "ipc_codes": ["H04L 9/08", "H04B 10/70"]
      },
      {
        "document_id": "15300010",
        "title": null,
        "similarity": 0.9641,
        "vote_weight": 0.0113,
        "ipc_codes": ["H04W 12/04"]
      }
    ]
  },
  "input_coverage": {"abstract": "complete", "specification": "not_supplied", "claims": "not_supplied", "details": {}},
  "extracted_terms": [
    {"term": "quantum keys", "source": "abstract", "weight": 1.0},
    {"term": "smart grid", "source": "abstract", "weight": 0.8}
  ],
  "warnings": [
    "Abstract-only inference: not an official classification; scope and rule checks need review.",
    "Confidence: raw neighbour vote share, not a calibrated probability."
  ],
  "degraded_mode": null,
  "provenance": {
    "taxonomy_version": "20260101",
    "source_snapshot": {"ipc_scheme": "sha256:9b2e4c0d"},
    "embedding_provider": "bge_m3",
    "embedding_model": "BAAI/bge-m3",
    "model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "embedding_run_id": "bge_m3-20260101-01",
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "scope_formatter": "scope-v3",
    "calibrator_id": null,
    "field_mapping_version": "2",
    "llm_model": null,
    "retrieval_settings": {"rrf_k": 60, "min_reranker_score": null, "enforce_lexical_checks": false},
    "encoding_policy": {"dimension": 1024, "normalized": true, "dtype": "float16"},
    "patent_encoder": {"model_id": "mpi-inno-comp/paecter", "model_revision": "1f355d9e9a34dc4859e22d05ce27e970eeacfbd2"},
    "patent_corpus": {"corpus_id": "hupd-2017", "source": "hupd", "document_count": 293711, "labels_ipc_edition": null},
    "vote_settings": {"k": 100, "temperature": 0.005, "beta": 1.0, "min_vote_share": 0.3, "hnsw_ef_search": 200}
  },
  "timings_ms": {"patent_encode": 41.2, "patent_search": 18.7, "vote": 0.9, "total": 902.3},
  "ranked_debug": null,
  "retrieval_debug": null
}
```

With the definition fallback the same request would return `"classifier": "definition_fallback"`, a
non-null `fallback_reason` (for example `"no_serving_corpus"`), `"vote": null` (unless the reason is
`vote_below_minimum`), all-null `vote_share` objects and the 1.x confidence statuses; see
`fixtureResponse()` and `fixtureWeakVoteResponse()` in `apps/web/tests/fixtures.ts`.
