"""Request/response contracts for POST /v1/classify (endpoint v1, response contract 2.0.0).

abstract is required and must be nonempty after whitespace trimming. title/specification/claims are
nullable plain-text strings; blank values are normalized to absent while nonblank text is kept verbatim.

Contract 2.0.0 (neighbour vote) is additive over 1.x. A PaECTER neighbour vote over a labelled patent
corpus picks the codes and their scores (`classifier="neighbour_vote"`); the definition-matching
pipeline explains those codes, or produces the codes itself when the vote cannot run or is weak
(`classifier="definition_fallback"`, with `fallback_reason`). The prose half of the contract (why,
status semantics, client behaviour, breaking changes) is the vault note
`02-Contracts/classify-response-v2.md`; the wire summary is docs/api.md. Mirrors:
apps/web/lib/types.ts and apps/web/tests/fixtures.ts.

Every response field is always present on the wire. Defaults on response models exist only so the
v1 code path keeps validating; `json_schema_serialization_defaults_required` makes the OpenAPI
response schema mark them required.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CLASSIFY_CONTRACT_VERSION = "2.0.0"
# Hard caps, validated. The server fills `vote.neighbours` with the min(MAX_NEIGHBOURS, neighbour_count)
# nearest patents and each `vote.top_codes.<level>` with the min(MAX_TOP_CODES_PER_LEVEL, n) best codes.
MAX_NEIGHBOURS = 10
MAX_TOP_CODES_PER_LEVEL = 10

Status = Literal["candidates_found", "needs_review", "insufficient_evidence"]
ConfidenceStatus = Literal[
    "calibrated", "not_calibrated", "incompatible_artifact", "unsupported_input_mode", "vote_share_uncalibrated"
]
Coverage = Literal["complete", "partial", "not_supplied", "empty"]
TargetLevel = Literal["group", "maingroup", "subgroup", "subclass", "class"]
SourceField = Literal["abstract", "title", "specification", "claims"]
# Which component chose the codes in `candidates`.
Classifier = Literal["neighbour_vote", "definition_fallback"]
# Why the definition fallback chose the codes. Null exactly when classifier == "neighbour_vote".
FallbackReason = Literal[
    "neighbour_vote_disabled",  # operator setting off (also the v1 code path, which has no vote)
    "no_serving_corpus",  # no validated patent corpus is serving
    "patent_encoder_unavailable",  # patent encoder not loaded, or its revision differs from the corpus's
    "vote_empty",  # search returned no neighbour with a resolvable IPC label
    "vote_below_minimum",  # the vote ran; first code's subclass share < vote_settings.min_vote_share
    "vote_failed",  # patent search or vote raised; detail in warnings
]
VoteCodeLevel = Literal["subclass", "maingroup", "subgroup"]

_SCORE_STATUSES = {"calibrated", "vote_share_uncalibrated"}


class _Response(BaseModel):
    """Base for response models: fields with defaults are still always serialized, so mark them required."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


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


class SupportingPassage(_Response):
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


class Confidence(_Response):
    """`score` is a calibrated probability only when status == "calibrated".

    status == "vote_share_uncalibrated": score is the raw neighbour vote share of the candidate at its
    own level (subclass -> vote_share.subclass, maingroup -> vote_share.maingroup, subgroup ->
    vote_share.exact; class -> the class-level share, which VoteShare cannot hold), calibrator_id is null.
    Every other non-calibrated status has score null.
    """

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ConfidenceStatus
    calibrator_id: str | None = None
    interpretation: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _score_matches_status(self) -> Confidence:
        if self.status in _SCORE_STATUSES and self.score is None:
            raise ValueError(f"confidence.score is required when status is {self.status!r}")
        if self.status not in _SCORE_STATUSES and self.score is not None:
            raise ValueError(f"confidence.score must be null when status is {self.status!r}")
        if self.status == "vote_share_uncalibrated" and self.calibrator_id is not None:
            raise ValueError("confidence.calibrator_id must be null for an uncalibrated vote share")
        return self


class VoteShare(_Response):
    """Similarity-weighted share of the k nearest corpus patents carrying this code, per level, in [0, 1].

    subclass: neighbours carrying any code in the candidate's subclass. maingroup: any code in its main
    group. exact: exactly this code (for a main-group candidate, neighbours labelled with that main group
    itself). Multi-label, so shares of different codes do not sum to 1; for one code
    subclass >= maingroup >= exact. Null means "not applicable": no vote shares are reported (all three
    null whenever classifier == "definition_fallback"), or the code is coarser than the level (a
    subclass candidate has maingroup = exact = null; a class candidate, reported when target_level="class",
    has all three null and its class-level share only in confidence.score). 0.0 is a real share.
    """

    subclass: float | None = Field(default=None, ge=0.0, le=1.0)
    maingroup: float | None = Field(default=None, ge=0.0, le=1.0)
    exact: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _nested_levels(self) -> VoteShare:
        if self.exact is not None and self.maingroup is None:
            raise ValueError("vote_share.exact requires vote_share.maingroup")
        if self.maingroup is not None and self.subclass is None:
            raise ValueError("vote_share.maingroup requires vote_share.subclass")
        if self.maingroup is not None and self.subclass is not None and self.maingroup > self.subclass:
            raise ValueError("vote_share.maingroup cannot exceed vote_share.subclass")
        if self.exact is not None and self.maingroup is not None and self.exact > self.maingroup:
            raise ValueError("vote_share.exact cannot exceed vote_share.maingroup")
        return self

    def is_empty(self) -> bool:
        return self.subclass is None and self.maingroup is None and self.exact is None


class PathEntry(_Response):
    code: str
    display_code: str
    title: str
    level: str


class ReferenceInfo(_Response):
    target_code: str
    target_display_code: str | None = None
    target_title: str | None = None
    kind: Literal["cross_reference", "precedence_reference"]
    original_wording: str | None = None
    target_exists: bool
    scope_check: str = "unresolved: reference condition not evaluated automatically"


class TechnicalScores(_Response):
    """Retrieval relevance values. None of these is a probability."""

    cosine_similarity: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    fused_rank: int | None = None
    reranker_score: float | None = None
    reranker_source: str | None = None


class Candidate(_Response):
    code: str
    display_code: str
    title: str
    original_title: str | None = None
    level: str
    confidence: Confidence
    # v2: raw vote shares for this code. Always present; all-null when classifier == "definition_fallback".
    vote_share: VoteShare = Field(default_factory=VoteShare)
    # v2: may be empty when classifier == "neighbour_vote" (evidence explains a vote code, it does not veto it).
    supporting_passages: list[SupportingPassage]
    depends_on_optional_documents: bool = False
    rationale: str
    unresolved_ambiguity: list[str]
    missing_defining_terms: list[str] = []
    parent_path: list[PathEntry]
    references: list[ReferenceInfo]
    scores: TechnicalScores
    origin: list[str]


class Alternative(_Response):
    code: str
    display_code: str
    title: str
    level: str
    reason: str
    unresolved_ambiguity: list[str] = []
    supporting_passages: list[SupportingPassage] = []
    confidence: Confidence
    # v2: same semantics as Candidate.vote_share (a vote code demoted by a precedence rule keeps its shares).
    vote_share: VoteShare = Field(default_factory=VoteShare)
    scores: TechnicalScores


class AdditionalField(_Response):
    label: str
    mapping_key: str
    from_code: str


class FieldOfInvention(_Response):
    label: str
    mapping_key: str | None
    mapping_version: str
    additional_labels: list[AdditionalField] = []
    rationale: str


class SummaryLine(_Response):
    """confidence_text by confidence.status (server-rendered; clients show it verbatim):

    calibrated -> f"{round(100 * score, 1)}%"; vote_share_uncalibrated -> f"vote share: {round(100 * score, 1)}%";
    not_calibrated -> "confidence: not calibrated"; incompatible_artifact -> "confidence: calibrator incompatible";
    unsupported_input_mode -> "confidence: not calibrated for this input mode".
    """

    display_code: str
    confidence_text: str


class Summary(_Response):
    """Single shared formatting contract for the compact two-row table (UI + Markdown export)."""

    field_of_invention: str
    classification_lines: list[SummaryLine]
    classification_cell_text: str
    markdown: str


class InputCoverage(_Response):
    abstract: Coverage
    specification: Coverage
    claims: Coverage
    details: dict[str, dict[str, int]] = {}


# ---------------------------------------------------------------------- v2: neighbour vote


class CodeShare(_Response):
    """One code in a per-level vote ranking."""

    display_code: str
    share: float = Field(ge=0.0, le=1.0)


class VoteTopCodes(_Response):
    """Best codes per level, descending share, at most MAX_TOP_CODES_PER_LEVEL each.

    Keys at each level: subclass "H04L"; maingroup "H04L 9/00"; exact = the neighbours' resolved label
    codes (main group or subgroup display codes, e.g. "H04L 9/08").
    """

    subclass: list[CodeShare] = Field(max_length=MAX_TOP_CODES_PER_LEVEL)
    maingroup: list[CodeShare] = Field(max_length=MAX_TOP_CODES_PER_LEVEL)
    exact: list[CodeShare] = Field(max_length=MAX_TOP_CODES_PER_LEVEL)


class NeighbourPatent(_Response):
    """One of the nearest labelled corpus patents to the query."""

    document_id: str  # source document id within provenance.patent_corpus (e.g. HUPD application number)
    title: str | None  # null only when the source record has no title
    similarity: float = Field(ge=-1.0, le=1.0)  # cosine similarity (1 - cosine distance); not a probability
    vote_weight: float = Field(ge=0.0, le=1.0)  # this neighbour's normalised weight in the vote (all k sum to 1)
    ipc_codes: list[str]  # resolved display codes, main label first; empty if none resolved


class VoteResult(_Response):
    """The neighbour vote as computed. Present iff classifier == "neighbour_vote" or
    fallback_reason == "vote_below_minimum"."""

    first_code: str  # display code chosen hierarchically: subclass -> main group -> subgroup
    first_code_level: VoteCodeLevel
    first_code_share: VoteShare  # subclass is the value compared with vote_settings.min_vote_share
    top_codes: VoteTopCodes
    neighbour_count: int = Field(ge=1)  # neighbours that entered the vote (<= vote_settings.k)
    neighbours: list[NeighbourPatent] = Field(max_length=MAX_NEIGHBOURS)  # descending similarity

    @model_validator(mode="after")
    def _first_share_present(self) -> VoteResult:
        if self.first_code_share.subclass is None:
            raise ValueError("vote.first_code_share.subclass is required")
        if own_level_share(self.first_code_level, self.first_code_share) is None:
            raise ValueError(f"vote.first_code_share needs a share at first_code_level ({self.first_code_level})")
        if not _display_matches_level(self.first_code, self.first_code_level):
            raise ValueError(f"vote.first_code {self.first_code!r} is not a {self.first_code_level} display code")
        if len(self.neighbours) > self.neighbour_count:
            raise ValueError("vote.neighbours cannot exceed vote.neighbour_count")
        return self


class PatentEncoderInfo(_Response):
    model_id: str  # e.g. "mpi-inno-comp/paecter"
    model_revision: str  # pinned commit hash


class PatentCorpusInfo(_Response):
    corpus_id: str  # stable identifier of the serving corpus (stringified if stored as an integer)
    source: str  # known values: "hupd", "uspto_odp"
    document_count: int = Field(ge=0)
    labels_ipc_edition: str | None  # IPC edition of the corpus labels, e.g. "2017.01"; null if unknown


class VoteSettings(_Response):
    k: int = Field(ge=1)
    temperature: float = Field(gt=0.0)  # T in exp((s - s1) / T)
    beta: float = Field(ge=0.0, le=1.0)  # weight of secondary labels (main label weighs 1)
    min_vote_share: float = Field(ge=0.0, le=1.0)  # compared with vote.first_code_share.subclass
    hnsw_ef_search: int | None = Field(ge=1)  # null when the search is exact (no HNSW index)


class Provenance(_Response):
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
    # v2. Null when not applicable: patent_encoder when the vote is disabled or the encoder is not loaded;
    # patent_corpus when no corpus is serving (or the vote is disabled); vote_settings when the vote is
    # disabled. All three are non-null whenever ClassifyResponse.vote is non-null.
    patent_encoder: PatentEncoderInfo | None = None
    patent_corpus: PatentCorpusInfo | None = None
    vote_settings: VoteSettings | None = None


class RankedDebug(_Response):
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


class ClassifyResponse(_Response):
    status: Status
    taxonomy_version: str
    # v2. Required on the wire. The defaults describe the v1 code path (no vote implemented) so it keeps
    # validating; the v2 pipeline (TASK-0010) must set both explicitly on every path.
    classifier: Classifier = "definition_fallback"
    fallback_reason: FallbackReason | None = "neighbour_vote_disabled"
    field_of_invention: FieldOfInvention
    summary: Summary
    candidates: list[Candidate]
    alternatives: list[Alternative]
    # v2. See VoteResult for when it is present.
    vote: VoteResult | None = None
    input_coverage: InputCoverage
    extracted_terms: list[dict[str, str | float]]
    warnings: list[str]
    degraded_mode: str | None = None
    provenance: Provenance
    timings_ms: dict[str, float] = {}
    ranked_debug: list[RankedDebug] | None = None
    # Evaluation diagnostics: route subclasses, pre-expansion candidates, and top reranked codes.
    # v2 adds, when include_ranked_debug and a vote ran: vote_subclass, vote_maingroup, vote_exact (display
    # codes in vote order, at most 50 each) and vote_neighbour_ids (all neighbour ids, descending similarity).
    retrieval_debug: dict[str, list[str]] | None = None

    @model_validator(mode="after")
    def _classifier_consistency(self) -> ClassifyResponse:
        voted = self.classifier == "neighbour_vote"
        if voted and self.fallback_reason is not None:
            raise ValueError("fallback_reason must be null when classifier is 'neighbour_vote'")
        if not voted and self.fallback_reason is None:
            raise ValueError("fallback_reason is required when classifier is 'definition_fallback'")
        vote_expected = voted or self.fallback_reason == "vote_below_minimum"
        if vote_expected and self.vote is None:
            raise ValueError("vote is required when the neighbour vote ran")
        if not vote_expected and self.vote is not None:
            raise ValueError(f"vote must be null when fallback_reason is {self.fallback_reason!r}")
        p = self.provenance
        if self.vote is not None and (p.patent_encoder is None or p.patent_corpus is None or p.vote_settings is None):
            raise ValueError("provenance.patent_encoder, patent_corpus and vote_settings are required with a vote")
        if self.vote is not None and p.vote_settings is not None and self.vote.neighbour_count > p.vote_settings.k:
            raise ValueError("vote.neighbour_count cannot exceed provenance.vote_settings.k")
        if voted and self.vote is not None:
            # The vote always yields its first code (computed at the deepest level). It leads `candidates`,
            # reported either as itself or as its ancestor at a coarser requested target_level (main group,
            # subclass, class). Exception: an unconditional precedence rule demoted it (ADR-0004); then it
            # (or that ancestor) is in `alternatives` and a supported code leads.
            if not self.candidates:
                raise ValueError("candidates must not be empty when classifier is 'neighbour_vote'")
            first, first_level = self.vote.first_code, self.vote.first_code_level
            lead = self.candidates[0]
            if not is_first_code_or_ancestor(lead.display_code, lead.level, first, first_level) and not any(
                is_first_code_or_ancestor(a.display_code, a.level, first, first_level) for a in self.alternatives
            ):
                raise ValueError(
                    f"candidates[0] ({lead.display_code}, {lead.level}) must be vote.first_code {first} or its "
                    "ancestor at the candidate's level, or that code must be in alternatives when a precedence "
                    "rule demoted it"
                )
        for c in self.candidates:
            if voted:
                if c.level == "class":
                    # A class is coarser than every VoteShare level: all three are null (not applicable), and
                    # confidence.score carries the class-level share.
                    if not c.vote_share.is_empty():
                        raise ValueError(f"candidate {c.display_code}: a class candidate has an all-null vote_share")
                elif c.vote_share.subclass is None:
                    raise ValueError(f"candidate {c.display_code}: vote_share.subclass is required with the vote")
                if c.confidence.status not in _SCORE_STATUSES:
                    raise ValueError(
                        f"candidate {c.display_code}: confidence.status must be 'calibrated' or "
                        "'vote_share_uncalibrated' when classifier is 'neighbour_vote'"
                    )
            elif not c.vote_share.is_empty():
                raise ValueError(f"candidate {c.display_code}: vote_share must be all-null in the fallback")
            if c.confidence.status == "vote_share_uncalibrated":
                if not voted:
                    raise ValueError(
                        "confidence.status 'vote_share_uncalibrated' is only valid with the neighbour vote"
                    )
                if c.level == "class":
                    continue  # score is the class-level share; VoteShare has no class level to compare with
                own = own_level_share(c.level, c.vote_share)
                if own is None or abs(own - (c.confidence.score or 0.0)) > 1e-9:
                    raise ValueError(
                        f"candidate {c.display_code}: an uncalibrated confidence.score must equal its "
                        f"vote share at its own level ({c.level})"
                    )
        if not voted:
            for a in self.alternatives:
                if not a.vote_share.is_empty() or a.confidence.status == "vote_share_uncalibrated":
                    raise ValueError(f"alternative {a.display_code}: no vote shares in the fallback")
        return self


def own_level_share(level: str, share: VoteShare) -> float | None:
    """The vote share that is a candidate's raw score: subclass -> subclass, maingroup -> maingroup,
    subgroup -> exact."""
    return {"subclass": share.subclass, "maingroup": share.maingroup, "subgroup": share.exact}.get(level)


def _display_matches_level(display: str, level: str) -> bool:
    """Shape of catalogue display codes (ingestion.wipo.display_symbol): class "H04", subclass "H04L",
    main group "H04L 9/00", subgroup "H04L 9/08"."""
    if level == "class":
        return len(display) == 3
    if level == "subclass":
        return len(display) == 4
    if " " not in display or "/" not in display:
        return False
    return display.endswith("/00") if level == "maingroup" else level == "subgroup" and not display.endswith("/00")


def vote_code_ancestors(first_code: str, first_code_level: str) -> dict[str, str]:
    """Display codes of the vote's first code's strictly coarser ancestors, keyed by catalogue level."""
    out = {"class": first_code[:3]}
    if first_code_level in ("maingroup", "subgroup"):
        out["subclass"] = first_code[:4]
    if first_code_level == "subgroup":
        out["maingroup"] = first_code.split("/")[0] + "/00"
    return out


def is_first_code_or_ancestor(display: str, level: str, first_code: str, first_code_level: str) -> bool:
    """True when (display, level) is the vote's first code, or its ancestor at `level` (coarser target_level)."""
    if display == first_code:
        return True
    return vote_code_ancestors(first_code, first_code_level).get(level) == display


class ErrorBody(BaseModel):
    error: str
    message: str
    details: dict | list | None = None


# 503 `patent_corpus_unavailable`: the vote could not run and the definition fallback could not serve.
VoteUnavailableReason = Literal["no_serving_corpus", "patent_encoder_unavailable", "vote_empty", "vote_failed"]


class PatentCorpusUnavailableDetails(BaseModel):
    """`ErrorBody.details` for 503 patent_corpus_unavailable."""

    reason: VoteUnavailableReason
    fallback_error: str  # the definition path's own error code, e.g. "embeddings_unavailable"
