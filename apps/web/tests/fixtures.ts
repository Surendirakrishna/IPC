// TEST-ONLY synthetic contract fixtures (origin=contract_fixture or synthetic patent ids). Never served by the API.
// Mirrors apps/api/app/schemas/classify.py, response contract 2.0.0. Every fixture validates with
// ClassifyResponse.model_validate; fixtureVoteResponse() is the docs/api.md worked example with the emoji abstract.
// Patent ids 153xxxxx / 154xxxxx and their titles are invented; do not look them up.
import type { ApiErrorBody, ClassifyResponse } from "../lib/types";

export const ABSTRACT =
  "🔐 Quantum keys are distributed among smart grid nodes. The system comprises smart meters, substations, inverters, chargers, and control centers.";

/**
 * Definition fallback (fallback_reason "no_serving_corpus"). Same visible summary as the v1 fixture, so
 * existing assertions ("confidence: not calibrated", no "%") still hold.
 * Code-point offsets: the leading emoji is ONE code point but TWO UTF-16 units.
 */
export function fixtureResponse(): ClassifyResponse {
  return {
    "status": "needs_review",
    "taxonomy_version": "20260101",
    "classifier": "definition_fallback",
    "fallback_reason": "no_serving_corpus",
    "field_of_invention": {
      "label": "COMMUNICATION",
      "mapping_key": "H04",
      "mapping_version": "1",
      "additional_labels": [],
      "rationale": "fixture"
    },
    "summary": {
      "field_of_invention": "COMMUNICATION",
      "classification_lines": [{"display_code": "H04L 9/08", "confidence_text": "confidence: not calibrated"}],
      "classification_cell_text": "H04L 9/08 — confidence: not calibrated",
      "markdown": "| Field Of Invention | COMMUNICATION |\n| :--- | :--- |\n| Classification (IPC) | H04L 9/08 — confidence: not calibrated |\n"
    },
    "candidates": [
      {
        "code": "H04L0009080000",
        "display_code": "H04L 9/08",
        "title": "Key distribution",
        "original_title": "Key distribution",
        "level": "subgroup",
        "confidence": {
          "score": null,
          "status": "not_calibrated",
          "calibrator_id": null,
          "interpretation": null,
          "detail": "No validated calibrator artifact is active."
        },
        "vote_share": {"subclass": null, "maingroup": null, "exact": null},
        "supporting_passages": [
          {
            "source": "abstract",
            "quote": "Quantum keys are distributed among smart grid nodes.",
            "start": 2,
            "end": 54,
            "claim_number": null,
            "section": null,
            "evidence_type": "retrieval_evidence_requires_review",
            "relevance": 1.2
          }
        ],
        "depends_on_optional_documents": false,
        "rationale": "fixture rationale",
        "unresolved_ambiguity": ["fixture ambiguity"],
        "missing_defining_terms": [],
        "parent_path": [{"code": "H", "display_code": "H", "title": "ELECTRICITY", "level": "section"}],
        "references": [],
        "scores": {
          "cosine_similarity": 0.61,
          "keyword_rank": 3,
          "keyword_score": 0.1,
          "vector_rank": 1,
          "rrf_score": 0.03,
          "fused_rank": 1,
          "reranker_score": 1.2,
          "reranker_source": "abstract"
        },
        "origin": ["contract_fixture"]
      }
    ],
    "alternatives": [
      {
        "code": "H04L0009320000",
        "display_code": "H04L 9/32",
        "title": "including means for verifying identity",
        "level": "subgroup",
        "reason": "Same main group; rule check unresolved.",
        "unresolved_ambiguity": [],
        "supporting_passages": [],
        "confidence": {
          "score": null,
          "status": "not_calibrated",
          "calibrator_id": null,
          "interpretation": null,
          "detail": "No validated calibrator artifact is active."
        },
        "vote_share": {"subclass": null, "maingroup": null, "exact": null},
        "scores": {
          "cosine_similarity": null,
          "keyword_rank": null,
          "keyword_score": null,
          "vector_rank": null,
          "rrf_score": null,
          "fused_rank": null,
          "reranker_score": 0.4,
          "reranker_source": "abstract"
        }
      }
    ],
    "vote": null,
    "input_coverage": {"abstract": "complete", "specification": "not_supplied", "claims": "not_supplied", "details": {}},
    "extracted_terms": [],
    "warnings": [
      "Illustrative contract fixture.",
      "Neighbour vote not used (no serving patent corpus); definition matching chose these codes."
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
      "patent_corpus": null,
      "vote_settings": {"k": 100, "temperature": 0.005, "beta": 1.0, "min_vote_share": 0.3, "hnsw_ef_search": 200}
    },
    "timings_ms": {},
    "ranked_debug": null,
    "retrieval_debug": null
  };
}

/** Alias that names the case explicitly. */
export const fixtureFallbackResponse = fixtureResponse;

/** Neighbour vote primary: vote shares, uncalibrated, nearest patents present. */
export function fixtureVoteResponse(): ClassifyResponse {
  return {
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
            "start": 2,
            "end": 54,
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
      "Confidence: raw neighbour vote share, not a calibrated probability.",
      "Illustrative contract fixture."
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
  };
}

/** Definition fallback after a weak vote (fallback_reason "vote_below_minimum"): `vote` present, codes from definition matching. */
export function fixtureWeakVoteResponse(): ClassifyResponse {
  return {
    "status": "needs_review",
    "taxonomy_version": "20260101",
    "classifier": "definition_fallback",
    "fallback_reason": "vote_below_minimum",
    "field_of_invention": {
      "label": "COMMUNICATION",
      "mapping_key": "H04",
      "mapping_version": "1",
      "additional_labels": [],
      "rationale": "fixture"
    },
    "summary": {
      "field_of_invention": "COMMUNICATION",
      "classification_lines": [{"display_code": "H04L 9/08", "confidence_text": "confidence: not calibrated"}],
      "classification_cell_text": "H04L 9/08 — confidence: not calibrated",
      "markdown": "| Field Of Invention | COMMUNICATION |\n| :--- | :--- |\n| Classification (IPC) | H04L 9/08 — confidence: not calibrated |\n"
    },
    "candidates": [
      {
        "code": "H04L0009080000",
        "display_code": "H04L 9/08",
        "title": "Key distribution",
        "original_title": "Key distribution",
        "level": "subgroup",
        "confidence": {
          "score": null,
          "status": "not_calibrated",
          "calibrator_id": null,
          "interpretation": null,
          "detail": "No validated calibrator artifact is active."
        },
        "vote_share": {"subclass": null, "maingroup": null, "exact": null},
        "supporting_passages": [
          {
            "source": "abstract",
            "quote": "Quantum keys are distributed among smart grid nodes.",
            "start": 2,
            "end": 54,
            "claim_number": null,
            "section": null,
            "evidence_type": "retrieval_evidence_requires_review",
            "relevance": 1.2
          }
        ],
        "depends_on_optional_documents": false,
        "rationale": "fixture rationale",
        "unresolved_ambiguity": ["fixture ambiguity"],
        "missing_defining_terms": [],
        "parent_path": [{"code": "H", "display_code": "H", "title": "ELECTRICITY", "level": "section"}],
        "references": [],
        "scores": {
          "cosine_similarity": 0.61,
          "keyword_rank": 3,
          "keyword_score": 0.1,
          "vector_rank": 1,
          "rrf_score": 0.03,
          "fused_rank": 1,
          "reranker_score": 1.2,
          "reranker_source": "abstract"
        },
        "origin": ["contract_fixture"]
      }
    ],
    "alternatives": [
      {
        "code": "H04L0009320000",
        "display_code": "H04L 9/32",
        "title": "including means for verifying identity",
        "level": "subgroup",
        "reason": "Same main group; rule check unresolved.",
        "unresolved_ambiguity": [],
        "supporting_passages": [],
        "confidence": {
          "score": null,
          "status": "not_calibrated",
          "calibrator_id": null,
          "interpretation": null,
          "detail": "No validated calibrator artifact is active."
        },
        "vote_share": {"subclass": null, "maingroup": null, "exact": null},
        "scores": {
          "cosine_similarity": null,
          "keyword_rank": null,
          "keyword_score": null,
          "vector_rank": null,
          "rrf_score": null,
          "fused_rank": null,
          "reranker_score": 0.4,
          "reranker_source": "abstract"
        }
      }
    ],
    "vote": {
      "first_code": "G06Q 50/06",
      "first_code_level": "subgroup",
      "first_code_share": {"subclass": 0.2016, "maingroup": 0.2016, "exact": 0.2016},
      "top_codes": {
        "subclass": [
          {"display_code": "G06Q", "share": 0.2016},
          {"display_code": "H02J", "share": 0.2012},
          {"display_code": "H04L", "share": 0.2012},
          {"display_code": "H01M", "share": 0.101},
          {"display_code": "G06F", "share": 0.1006}
        ],
        "maingroup": [
          {"display_code": "G06Q 50/00", "share": 0.2016},
          {"display_code": "H02J 13/00", "share": 0.2012},
          {"display_code": "H04L 9/00", "share": 0.1014},
          {"display_code": "H01M 10/00", "share": 0.101},
          {"display_code": "G06F 21/00", "share": 0.1006}
        ],
        "exact": [
          {"display_code": "G06Q 50/06", "share": 0.2016},
          {"display_code": "H02J 13/00", "share": 0.2012},
          {"display_code": "H04L 9/08", "share": 0.1014},
          {"display_code": "H01M 10/44", "share": 0.101},
          {"display_code": "G06F 21/60", "share": 0.1006}
        ]
      },
      "neighbour_count": 100,
      "neighbours": [
        {
          "document_id": "15400001",
          "title": "Energy trading platform for prosumers",
          "similarity": 0.9612,
          "vote_weight": 0.0121,
          "ipc_codes": ["G06Q 50/06", "H02J 13/00"]
        },
        {
          "document_id": "15400002",
          "title": "Key management for meter data",
          "similarity": 0.96118,
          "vote_weight": 0.0121,
          "ipc_codes": ["H04L 9/08"]
        },
        {
          "document_id": "15400003",
          "title": "Battery charge balancing for grid storage",
          "similarity": 0.96116,
          "vote_weight": 0.012,
          "ipc_codes": ["H01M 10/44"]
        },
        {
          "document_id": "15400004",
          "title": "Access control for grid software",
          "similarity": 0.96114,
          "vote_weight": 0.012,
          "ipc_codes": ["G06F 21/60"]
        },
        {
          "document_id": "15400005",
          "title": "Device pairing for home energy networks",
          "similarity": 0.96112,
          "vote_weight": 0.0119,
          "ipc_codes": ["H04W 12/04"]
        },
        {
          "document_id": "15400006",
          "title": "Telemetry gateway for utilities",
          "similarity": 0.9611,
          "vote_weight": 0.0119,
          "ipc_codes": ["H04L 67/12", "G06Q 50/06"]
        },
        {
          "document_id": "15400007",
          "title": "Feeder automation controller",
          "similarity": 0.96108,
          "vote_weight": 0.0118,
          "ipc_codes": ["H02J 13/00"]
        },
        {
          "document_id": "15400008",
          "title": "Smart meter tamper detection",
          "similarity": 0.96106,
          "vote_weight": 0.0118,
          "ipc_codes": ["G01R 22/06"]
        },
        {
          "document_id": "15400009",
          "title": "Process control for a substation plant",
          "similarity": 0.96104,
          "vote_weight": 0.0117,
          "ipc_codes": ["G05B 19/418"]
        },
        {
          "document_id": "15400010",
          "title": "Free-space optical link for substations",
          "similarity": 0.96102,
          "vote_weight": 0.0117,
          "ipc_codes": ["H04B 10/70"]
        }
      ]
    },
    "input_coverage": {"abstract": "complete", "specification": "not_supplied", "claims": "not_supplied", "details": {}},
    "extracted_terms": [],
    "warnings": [
      "Illustrative contract fixture.",
      "Neighbour vote too weak (first code G06Q 50/06, subclass vote share 20.2% < 30.0%); definition matching chose these codes."
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
    "timings_ms": {},
    "ranked_debug": null,
    "retrieval_debug": null
  };
}

/** 503 when the vote cannot run and the definition fallback cannot serve either. */
export const FIXTURE_PATENT_CORPUS_UNAVAILABLE: { status: 503; body: ApiErrorBody } = {
  status: 503,
  body: {
    "error": "patent_corpus_unavailable",
    "message": "No serving patent corpus, and the definition fallback is unavailable.",
    "details": {"reason": "no_serving_corpus", "fallback_error": "embeddings_unavailable"}
  },
};
