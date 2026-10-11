"""Request/response contracts for POST /v1/classify (v1).

abstract is required and must be nonempty after whitespace trimming. title/specification/claims are
nullable plain-text strings; blank values are normalized to absent while nonblank text is kept verbatim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["candidates_found", "needs_review", "insufficient_evidence"]
ConfidenceStatus = Literal["calibrated", "not_calibrated", "incompatible_artifact", "unsupported_input_mode"]
Coverage = Literal["complete", "partial", "not_supplied", "empty"]
TargetLevel = Literal["group", "maingroup", "subgroup", "subclass", "class"]
SourceField = Literal["abstract", "title", "specification", "claims"]


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    abstract: str
    title: str | None = None
    specification: str | None = None
    claims: str | None = None
    taxonomy_version: str | None = None
    target_level: TargetLevel = "group"
    max_results: int = Field(default=5, ge=1)
    use_llm: bool = False
    # Diagnostics (evaluation only): return the reranked candidate list regardless of the policy outcome.
    include_ranked_debug: bool = False
    section: str | None = Field(default=None, pattern=r"^[A-H]$")
    subclass: str | None = Field(default=None, pattern=r"^[A-H]\d{2}[A-Z]$")

    @field_validator("abstract")
    @classmethod
    def _abstract_nonblank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("abstract is required and must not be blank")
        return v

    @field_validator("title", "specification", "claims")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        return v


class SupportingPassage(BaseModel):
    source: SourceField
    quote: str
    start: int
    end: int
    claim_number: int | None = None
    section: str | None = None
    evidence_type: Literal["retrieval_evidence_requires_review", "llm_cited_verified"] = (
        "retrieval_evidence_requires_review"
    )
    relevance: float | None = Field(default=None, description="Reranker relevance of this passage (not a probability)")


class Confidence(BaseModel):
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ConfidenceStatus
    calibrator_id: str | None = None
    interpretation: str | None = None
    detail: str | None = None


class PathEntry(BaseModel):
    code: str
    display_code: str
    title: str
    level: str


class ReferenceInfo(BaseModel):
    target_code: str
    target_display_code: str | None = None
    target_title: str | None = None
    kind: Literal["cross_reference", "precedence_reference"]
    original_wording: str | None = None
    target_exists: bool
    scope_check: str = "unresolved: reference condition not evaluated automatically"


class TechnicalScores(BaseModel):
    """Retrieval relevance values. None of these is a probability."""

    cosine_similarity: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    fused_rank: int | None = None
    reranker_score: float | None = None
    reranker_source: str | None = None


class Candidate(BaseModel):
    code: str
    display_code: str
    title: str
    original_title: str | None = None
    level: str
    confidence: Confidence
    supporting_passages: list[SupportingPassage]
    depends_on_optional_documents: bool = False
    rationale: str
    unresolved_ambiguity: list[str]
    missing_defining_terms: list[str] = []
    parent_path: list[PathEntry]
    references: list[ReferenceInfo]
    scores: TechnicalScores
    origin: list[str]


class Alternative(BaseModel):
    code: str
    display_code: str
    title: str
    level: str
    reason: str
    unresolved_ambiguity: list[str] = []
    supporting_passages: list[SupportingPassage] = []
    confidence: Confidence
    scores: TechnicalScores


class AdditionalField(BaseModel):
    label: str
    mapping_key: str
    from_code: str


class FieldOfInvention(BaseModel):
    label: str
    mapping_key: str | None
    mapping_version: str
    additional_labels: list[AdditionalField] = []
    rationale: str


class SummaryLine(BaseModel):
    display_code: str
    confidence_text: str


class Summary(BaseModel):
    """Single shared formatting contract for the compact two-row table (UI + Markdown export)."""

    field_of_invention: str
    classification_lines: list[SummaryLine]
    classification_cell_text: str
    markdown: str


class InputCoverage(BaseModel):
    abstract: Coverage
    specification: Coverage
    claims: Coverage
    details: dict[str, dict[str, int]] = {}


class Provenance(BaseModel):
    taxonomy_version: str
    source_snapshot: dict[str, str]
    embedding_provider: str
    embedding_model: str
    model_revision: str
    embedding_run_id: str | None
    reranker_model: str
    reranker_revision: str
    scope_formatter: str
    calibrator_id: str | None
    field_mapping_version: str
    llm_model: str | None = None
    retrieval_settings: dict[str, int | float | str | bool | None]
    encoding_policy: dict[str, int | str | bool | None]


class RankedDebug(BaseModel):
    """Reranked candidate before the selection policy (relevance values, never probabilities)."""

    display_code: str
    code: str
    title: str
    reranker_score: float
    best_evidence_score: float | None
    record_type: str
    maingroup_code: str | None
    application_context_only: bool
    subclass_scope_ok: bool
    defining_terms_ok: bool
    outcome: str


class ClassifyResponse(BaseModel):
    status: Status
    taxonomy_version: str
    field_of_invention: FieldOfInvention
    summary: Summary
    candidates: list[Candidate]
    alternatives: list[Alternative]
    input_coverage: InputCoverage
    extracted_terms: list[dict[str, str | float]]
    warnings: list[str]
    degraded_mode: str | None = None
    provenance: Provenance
    timings_ms: dict[str, float] = {}
    ranked_debug: list[RankedDebug] | None = None
    # Evaluation diagnostics: route subclasses, pre-expansion candidates, and top reranked codes.
    retrieval_debug: dict[str, list[str]] | None = None


class ErrorBody(BaseModel):
    error: str
    message: str
    details: dict | list | None = None
