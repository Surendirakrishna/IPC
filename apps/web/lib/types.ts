// Mirrors apps/api/app/schemas/classify.py field for field (endpoint v1, response contract 2.0.0).
// Keep in sync with docs/api.md and the vault contract 02-Contracts/classify-response-v2.md.
// Response fields are never optional (`?`): the server always emits every field, using null where the
// Pydantic type is nullable. Request fields with a server default are optional.

export const CLASSIFY_CONTRACT_VERSION = "2.0.0";
/** Hard cap on ClassifyResponse.vote.neighbours. */
export const MAX_NEIGHBOURS = 10;
/** Hard cap on each list in VoteResult.top_codes. */
export const MAX_TOP_CODES_PER_LEVEL = 10;

export type Status = "candidates_found" | "needs_review" | "insufficient_evidence";
export type ConfidenceStatus =
  | "calibrated"
  | "not_calibrated"
  | "incompatible_artifact"
  | "unsupported_input_mode"
  | "vote_share_uncalibrated";
export type Coverage = "complete" | "partial" | "not_supplied" | "empty";
export type TargetLevel = "group" | "maingroup" | "subgroup" | "subclass" | "class";
export type SourceField = "abstract" | "title" | "specification" | "claims";
/** Which component chose the codes in `candidates`. */
export type Classifier = "neighbour_vote" | "definition_fallback";
/** Why the definition fallback chose the codes. Null exactly when classifier === "neighbour_vote". */
export type FallbackReason =
  | "neighbour_vote_disabled"
  | "no_serving_corpus"
  | "patent_encoder_unavailable"
  | "vote_empty"
  | "vote_below_minimum"
  | "vote_failed";
export type VoteCodeLevel = "subclass" | "maingroup" | "subgroup";

export interface ClassifyRequest {
  abstract: string;
  title?: string | null;
  specification?: string | null;
  claims?: string | null;
  taxonomy_version?: string | null;
  target_level?: TargetLevel;
  max_results?: number;
  use_llm?: boolean;
  /** Diagnostics (evaluation only). */
  include_ranked_debug?: boolean;
  /** One letter A-H. */
  section?: string | null;
  /** Subclass symbol, e.g. "H04L". */
  subclass?: string | null;
}

export interface SupportingPassage {
  source: SourceField;
  quote: string;
  /** Unicode code-point index into the original field (start inclusive). */
  start: number;
  /** Unicode code-point index into the original field (end exclusive). */
  end: number;
  claim_number: number | null;
  section: string | null;
  evidence_type: "retrieval_evidence_requires_review" | "llm_cited_verified";
  /** Reranker relevance of this passage (not a probability). */
  relevance: number | null;
}

/**
 * `score` is a calibrated probability only when status === "calibrated".
 * status === "vote_share_uncalibrated": score is the raw vote share at the candidate's own level
 * (subclass -> vote_share.subclass, maingroup -> vote_share.maingroup, subgroup -> vote_share.exact);
 * never label it a probability. Every other status has score === null.
 */
export interface Confidence {
  score: number | null;
  status: ConfidenceStatus;
  calibrator_id: string | null;
  interpretation: string | null;
  detail: string | null;
}

/**
 * Similarity-weighted share of the k nearest corpus patents carrying the code, per level, in [0, 1].
 * Multi-label: shares of different codes do not sum to 1. null = not applicable (fallback, or the code is
 * coarser than the level); 0 is a real share.
 */
export interface VoteShare {
  subclass: number | null;
  maingroup: number | null;
  exact: number | null;
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

/** Retrieval relevance values. None of these is a probability. */
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
  /** All-null when classifier === "definition_fallback". */
  vote_share: VoteShare;
  /** May be empty when classifier === "neighbour_vote". */
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
  vote_share: VoteShare;
  scores: TechnicalScores;
}

export interface AdditionalField { label: string; mapping_key: string; from_code: string }

export interface FieldOfInvention {
  label: string;
  mapping_key: string | null;
  mapping_version: string;
  additional_labels: AdditionalField[];
  rationale: string;
}

/** confidence_text is server-rendered; show it verbatim. */
export interface SummaryLine { display_code: string; confidence_text: string }

export interface Summary {
  field_of_invention: string;
  classification_lines: SummaryLine[];
  classification_cell_text: string;
  markdown: string;
}

export interface InputCoverage {
  abstract: Coverage;
  specification: Coverage;
  claims: Coverage;
  details: Record<string, Record<string, number>>;
}

export interface CodeShare { display_code: string; share: number }

export interface VoteTopCodes {
  subclass: CodeShare[];
  maingroup: CodeShare[];
  exact: CodeShare[];
}

export interface NeighbourPatent {
  /** Source document id within provenance.patent_corpus. */
  document_id: string;
  title: string | null;
  /** Cosine similarity; not a probability. */
  similarity: number;
  /** Normalised weight of this neighbour in the vote (all k sum to 1). */
  vote_weight: number;
  /** Resolved IPC display codes, main label first. */
  ipc_codes: string[];
}

/** Present iff classifier === "neighbour_vote" or fallback_reason === "vote_below_minimum". */
export interface VoteResult {
  first_code: string;
  first_code_level: VoteCodeLevel;
  /** .subclass is compared with provenance.vote_settings.min_vote_share. */
  first_code_share: VoteShare;
  top_codes: VoteTopCodes;
  neighbour_count: number;
  /** Descending similarity, at most MAX_NEIGHBOURS. */
  neighbours: NeighbourPatent[];
}

export interface PatentEncoderInfo { model_id: string; model_revision: string }

export interface PatentCorpusInfo {
  corpus_id: string;
  /** Known values: "hupd", "uspto_odp". */
  source: string;
  document_count: number;
  labels_ipc_edition: string | null;
}

export interface VoteSettings {
  k: number;
  temperature: number;
  beta: number;
  min_vote_share: number;
  hnsw_ef_search: number | null;
}

export interface Provenance {
  taxonomy_version: string;
  source_snapshot: Record<string, string>;
  embedding_provider: string;
  embedding_model: string;
  model_revision: string;
  embedding_run_id: string | null;
  reranker_model: string;
  reranker_revision: string;
  scope_formatter: string;
  calibrator_id: string | null;
  field_mapping_version: string;
  llm_model: string | null;
  retrieval_settings: Record<string, number | string | boolean | null>;
  encoding_policy: Record<string, number | string | boolean | null>;
  /** Non-null whenever ClassifyResponse.vote is non-null. */
  patent_encoder: PatentEncoderInfo | null;
  patent_corpus: PatentCorpusInfo | null;
  vote_settings: VoteSettings | null;
}

/** Reranked candidate before the selection policy (relevance values, never probabilities). */
export interface RankedDebug {
  display_code: string;
  code: string;
  title: string;
  reranker_score: number;
  best_evidence_score: number | null;
  record_type: string;
  maingroup_code: string | null;
  application_context_only: boolean;
  subclass_scope_ok: boolean;
  defining_terms_ok: boolean;
  outcome: string;
}

export interface ClassifyResponse {
  status: Status;
  taxonomy_version: string;
  classifier: Classifier;
  fallback_reason: FallbackReason | null;
  field_of_invention: FieldOfInvention;
  summary: Summary;
  candidates: Candidate[];
  alternatives: Alternative[];
  vote: VoteResult | null;
  input_coverage: InputCoverage;
  /** Pydantic types this as list[dict[str, str | float]]; the server always emits exactly these keys. */
  extracted_terms: { term: string; source: string; weight: number }[];
  warnings: string[];
  degraded_mode: string | null;
  provenance: Provenance;
  timings_ms: Record<string, number>;
  ranked_debug: RankedDebug[] | null;
  retrieval_debug: Record<string, string[]> | null;
}

export interface ApiErrorBody { error: string; message: string; details?: unknown }

/** `details` of a 503 `patent_corpus_unavailable` error. */
export interface PatentCorpusUnavailableDetails {
  reason: "no_serving_corpus" | "patent_encoder_unavailable" | "vote_empty" | "vote_failed";
  /** The definition path's own error code, e.g. "embeddings_unavailable". */
  fallback_error: string;
}
