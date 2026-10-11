// Mirrors apps/api/app/schemas/classify.py (v1 contract). Keep in sync with docs/api.md.

export type Status = "candidates_found" | "needs_review" | "insufficient_evidence";
export type ConfidenceStatus = "calibrated" | "not_calibrated" | "incompatible_artifact" | "unsupported_input_mode";
export type SourceField = "abstract" | "title" | "specification" | "claims";

export interface ClassifyRequest {
  abstract: string;
  title?: string | null;
  specification?: string | null;
  claims?: string | null;
  taxonomy_version?: string | null;
  target_level?: "group" | "maingroup" | "subgroup" | "subclass" | "class";
  max_results?: number;
  use_llm?: boolean;
}

export interface SupportingPassage {
  source: SourceField;
  quote: string;
  /** Unicode code-point index into the original field (start inclusive). */
  start: number;
  /** Unicode code-point index into the original field (end exclusive). */
  end: number;
  claim_number: number | null;
  section?: string | null;
  evidence_type: "retrieval_evidence_requires_review" | "llm_cited_verified";
  relevance: number | null;
}

export interface Confidence {
  score: number | null;
  status: ConfidenceStatus;
  calibrator_id: string | null;
  interpretation?: string | null;
  detail?: string | null;
}

export interface PathEntry { code: string; display_code: string; title: string; level: string }

export interface ReferenceInfo {
  target_code: string;
  target_display_code: string | null;
  target_title: string | null;
  kind: "cross_reference" | "precedence_reference";
  original_wording: string | null;
  target_exists: boolean;
  scope_check: string;
}

export interface TechnicalScores {
  cosine_similarity: number | null;
  keyword_rank: number | null;
  keyword_score: number | null;
  vector_rank: number | null;
  rrf_score: number | null;
  fused_rank: number | null;
  reranker_score: number | null;
  reranker_source: string | null;
}

export interface Candidate {
  code: string;
  display_code: string;
  title: string;
  original_title: string | null;
  level: string;
  confidence: Confidence;
  supporting_passages: SupportingPassage[];
  depends_on_optional_documents: boolean;
  rationale: string;
  unresolved_ambiguity: string[];
  missing_defining_terms: string[];
  parent_path: PathEntry[];
  references: ReferenceInfo[];
  scores: TechnicalScores;
  origin: string[];
}

export interface Alternative {
  code: string;
  display_code: string;
  title: string;
  level: string;
  reason: string;
  unresolved_ambiguity: string[];
  supporting_passages: SupportingPassage[];
  confidence: Confidence;
  scores: TechnicalScores;
}

export interface SummaryLine { display_code: string; confidence_text: string }

export interface ClassifyResponse {
  status: Status;
  taxonomy_version: string;
  field_of_invention: {
    label: string;
    mapping_key: string | null;
    mapping_version: string;
    additional_labels: { label: string; mapping_key: string; from_code: string }[];
    rationale: string;
  };
  summary: {
    field_of_invention: string;
    classification_lines: SummaryLine[];
    classification_cell_text: string;
    markdown: string;
  };
  candidates: Candidate[];
  alternatives: Alternative[];
  input_coverage: { abstract: string; specification: string; claims: string; details: Record<string, Record<string, number>> };
  extracted_terms: { term: string; source: string; weight: number }[];
  warnings: string[];
  degraded_mode: string | null;
  provenance: Record<string, unknown>;
  timings_ms: Record<string, number>;
}

export interface ApiErrorBody { error: string; message: string; details?: unknown }
