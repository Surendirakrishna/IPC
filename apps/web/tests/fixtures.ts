// TEST-ONLY synthetic contract fixture (origin=contract_fixture). Never served by the API.
import type { ClassifyResponse } from "../lib/types";

export const ABSTRACT =
  "🔐 Quantum keys are distributed among smart grid nodes. The system comprises smart meters, substations, inverters, chargers, and control centers.";

export function fixtureResponse(): ClassifyResponse {
  return {
    status: "needs_review",
    taxonomy_version: "20260101",
    field_of_invention: { label: "COMMUNICATION", mapping_key: "H04", mapping_version: "1", additional_labels: [], rationale: "fixture" },
    summary: {
      field_of_invention: "COMMUNICATION",
      classification_lines: [{ display_code: "H04L 9/08", confidence_text: "confidence: not calibrated" }],
      classification_cell_text: "H04L 9/08 — confidence: not calibrated",
      markdown:
        "| Field Of Invention | COMMUNICATION |\n| :--- | :--- |\n| Classification (IPC) | H04L 9/08 — confidence: not calibrated |\n",
    },
    candidates: [
      {
        code: "H04L0009080000",
        display_code: "H04L 9/08",
        title: "Key distribution",
        original_title: "Key distribution",
        level: "subgroup",
        confidence: { score: null, status: "not_calibrated", calibrator_id: null },
        // Code-point offsets: the leading emoji is ONE code point but TWO UTF-16 units.
        supporting_passages: [
          {
            source: "abstract",
            quote: "Quantum keys are distributed among smart grid nodes.",
            start: 2,
            end: 54,
            claim_number: null,
            evidence_type: "retrieval_evidence_requires_review",
            relevance: 1.2,
          },
        ],
        depends_on_optional_documents: false,
        rationale: "fixture rationale",
        unresolved_ambiguity: ["fixture ambiguity"],
        missing_defining_terms: [],
        parent_path: [{ code: "H", display_code: "H", title: "ELECTRICITY", level: "section" }],
        references: [],
        scores: {
          cosine_similarity: 0.61, keyword_rank: 3, keyword_score: 0.1, vector_rank: 1, rrf_score: 0.03,
          fused_rank: 1, reranker_score: 1.2, reranker_source: "abstract",
        },
        origin: ["contract_fixture"],
      },
    ],
    alternatives: [
      {
        code: "H04L0009320000",
        display_code: "H04L 9/32",
        title: "including means for verifying identity",
        level: "subgroup",
        reason: "Same main group; rule check unresolved.",
        unresolved_ambiguity: [],
        supporting_passages: [],
        confidence: { score: null, status: "not_calibrated", calibrator_id: null },
        scores: {
          cosine_similarity: null, keyword_rank: null, keyword_score: null, vector_rank: null, rrf_score: null,
          fused_rank: null, reranker_score: 0.4, reranker_source: "abstract",
        },
      },
    ],
    input_coverage: { abstract: "complete", specification: "not_supplied", claims: "not_supplied", details: {} },
    extracted_terms: [],
    warnings: ["Illustrative contract fixture."],
    degraded_mode: null,
    provenance: {},
    timings_ms: {},
  };
}
