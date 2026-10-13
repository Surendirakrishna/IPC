"""End-to-end classification pipeline.

Primary classifier (ADR-0005, response contract 2.0.0): a PaECTER neighbour vote over the serving
labelled patent corpus picks the codes and their raw vote shares. The definition pipeline then only
explains those codes (evidence passages, WIPO notes, precedence rules; ADR-0004: a rule never adds a
code), and it is the fallback when the vote is disabled, cannot run, or is weak:

abstract (+ title) -> PaECTER -> k nearest corpus patents (partial HNSW index) -> vote
  -> gate: rounded first-code subclass share >= NEIGHBOUR_MIN_VOTE_SHARE
  -> vote codes as candidates (vote order) -> hydrate -> rerank (scores only) -> evidence -> rules

Definition pipeline (the fallback, unchanged from 1.x):

abstract (+ optional title/claims/specification)
  -> source-labelled query views (abstract view always; optional views bounded)
  -> per view: PostgreSQL full-text (ts_rank_cd) + pgvector HNSW cosine retrieval
  -> RRF within views, then across sources with equal per-source caps
  -> expansion (parents, bounded siblings, validated reference targets), capped pool
  -> BGE reranker over deterministic scope documents
  -> extractive evidence with verified offsets + defining-term heuristic + application-context check
  -> conservative relevance/evidence policy (pre-calibration) -> supported vs alternatives
  -> optional constrained LLM (allowed-code enum + backend validation)
  -> per-code confidence from a compatible calibrator (else null + status)
  -> field-of-invention mapping, summary table contract.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import patents as patents_db
from app.db import repository as repo
from app.db.repository import Record
from app.schemas.classify import (
    MAX_NEIGHBOURS,
    MAX_TOP_CODES_PER_LEVEL,
    Alternative,
    Candidate,
    ClassifyRequest,
    ClassifyResponse,
    CodeShare,
    Confidence,
    FieldOfInvention,
    InputCoverage,
    NeighbourPatent,
    PatentCorpusInfo,
    PatentEncoderInfo,
    PathEntry,
    Provenance,
    RankedDebug,
    ReferenceInfo,
    SupportingPassage,
    TechnicalScores,
    VoteResult,
    VoteSettings,
    VoteShare,
    VoteTopCodes,
    own_level_share,
    vote_code_ancestors,
)
from app.services.classification import llm as llm_mod
from app.services.classification.evidence import (
    Passage,
    defining_feature_check,
    split_passages,
    subclass_scope_check,
    verify,
)
from app.services.classification.neighbour_vote import LEVELS, NeighbourVote, vote
from app.services.classification.reranker import Reranker
from app.services.classification.rules import RuleCandidate, apply_scheme_rules
from app.services.classification.scope import SCOPE_FORMATTER_VERSION, format_scope, parent_path
from app.services.classification.summary import build_summary
from app.services.confidence.calibrator import ConfidenceService
from app.services.embeddings.paecter import patent_text
from app.services.embeddings.providers import EmbeddingProvider
from app.services.fields.mapping import FieldMapping, field_of_invention
from app.services.ingestion.wipo import display_symbol
from app.services.retrieval.expansion import expand
from app.services.retrieval.fusion import FusedCandidate, ViewHits, fuse, vote_subclasses
from app.services.retrieval.keyword import build_websearch_query, extract_terms, keyword_search
from app.services.retrieval.patent_search import NeighbourShortfall, PatentNeighbour, search_patents
from app.services.retrieval.query_views import QueryView, build_views, regex_token_offsets
from app.services.retrieval.vector import vector_search

log = logging.getLogger(__name__)

LEVEL_TYPES: dict[str, tuple[str, ...]] = {
    "group": ("maingroup", "subgroup"),
    "maingroup": ("maingroup",),
    "subgroup": ("subgroup",),
    "subclass": ("subclass",),
    "class": ("class",),
}
ENUM_CUE_RE = re.compile(r"\b(comprises|comprising|includes|including|such as|consisting of|e\.g\.)\b", re.I)


# Rule kinds listed on a candidate for the reviewer (the rules themselves are applied in rules.py).
RULE_REVIEW_KINDS = ("precedence", "place_last", "place_first")

# --- neighbour vote (contract 2.0.0) ---------------------------------------------------------------
SHARE_DECIMALS = 4  # every vote share on the wire, and the gate, use round(x, 4)
VOTE_DEBUG_CODES = 50  # retrieval_debug.vote_<level>: at most 50 display codes each
# fallback reasons for which the vote could not run at all: if the definition path then fails too, the
# answer is 503 patent_corpus_unavailable (vote disabled / below minimum keep the definition path's own error)
VOTE_UNAVAILABLE_REASONS = ("no_serving_corpus", "patent_encoder_unavailable", "vote_empty", "vote_failed")
VOTE_INTERPRETATION = ("Raw neighbour vote share: the similarity-weighted fraction of the nearest labelled patents "
                       "that carry this code. Not a calibrated probability.")
VOTE_CLASS_INTERPRETATION = ("Raw neighbour vote share: the similarity-weighted fraction of the nearest labelled "
                             "patents that carry a code in this class. Not a calibrated probability.")
VOTE_CONFIDENCE_WARNING = "Confidence: raw neighbour vote share, not a calibrated probability."
NO_PASSAGE_NOTE = ("No supporting passage was found in the input for this code; it comes from the neighbour vote "
                   "and needs review.")


class PipelineUnavailable(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class InputRejected(ValueError):
    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class _VoteUnusable(RuntimeError):
    """The vote ran, but its first code cannot be reported (e.g. absent from the serving taxonomy release)."""


@dataclass
class ServingState:
    release: repo.Release
    run_id: str | None
    storage_table: str
    encoding_policy: dict[str, Any]


@dataclass
class Scored:
    fc: FusedCandidate
    rec: Record
    reranker_score: float
    reranker_source: str
    view_scores: dict[str, float] = field(default_factory=dict)
    passages: list[tuple[Passage, float]] = field(default_factory=list)
    present_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    application_context_only: bool = False
    subclass_present: list[str] = field(default_factory=list)
    subclass_missing: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def select_retrieval_views(views: list[QueryView], s: Settings) -> tuple[list[QueryView], dict[str, float]]:
    """Views that vote in candidate retrieval, with per-view weights for claim views.

    Claims are parsed and retained for evidence review. They vote only in independent_weighted mode,
    with a per-claim weight; a weight of 0 keeps that claim out. Unnumbered claim chunks count as dependent.
    """
    weights: dict[str, float] = {}
    if s.claims_retrieval_mode == "independent_weighted":
        for v in views:
            if v.source == "claims":
                w = s.independent_claim_weight if v.is_independent_claim else s.dependent_claim_weight
                if w > 0:
                    weights[v.view_id] = w
    return [v for v in views if v.source != "claims" or v.view_id in weights], weights


def resolve_release(session: Session, version: str | None) -> repo.Release:
    """The active taxonomy release; a requested version must be that release."""
    release = repo.active_release(session)
    if release is None:
        raise PipelineUnavailable("corpus_unavailable", "No active, validated taxonomy release.")
    if version and version != release.taxonomy_version:
        other = repo.get_release(session, version)
        if other is None:
            raise InputRejected("invalid_taxonomy_version", f"Unknown taxonomy_version {version!r}.")
        raise InputRejected("invalid_taxonomy_version",
                            f"taxonomy_version {version!r} is not the active serving release "
                            f"({release.taxonomy_version}).")
    return release


def serving_run(session: Session, release: repo.Release, provider: EmbeddingProvider) -> ServingState:
    """The serving catalogue embedding run for ``release``, produced by ``provider``'s model and revision."""
    row = session.execute(
        text(
            "SELECT run_id, storage_table, encoding_policy, model_id, model_revision FROM embedding_runs"
            " WHERE taxonomy_version = :v AND storage_table = :t AND is_serving AND state = 'validated'"
        ),
        {"v": release.taxonomy_version, "t": provider.policy.storage_table},
    ).first()
    if row is None:
        raise PipelineUnavailable("embeddings_unavailable",
                                  "No complete, validated, serving embedding run for the active release.")
    if row[3] != provider.policy.model_id or row[4] != provider.policy.model_revision:
        raise PipelineUnavailable("embeddings_incompatible",
                                  "Serving embedding run was produced by a different model/revision than the "
                                  "loaded query encoder; regenerate embeddings.")
    return ServingState(release, row[0], row[1], row[2])


def serving_state(session: Session, provider: EmbeddingProvider, version: str | None) -> ServingState:
    return serving_run(session, resolve_release(session, version), provider)


def _is_enumeration(quote: str) -> bool:
    return quote.count(",") >= 4 and bool(ENUM_CUE_RE.search(quote))


# ---------------------------------------------------------------------- neighbour vote helpers


class PatentEncoderLike(Protocol):
    """What the pipeline needs from the patent encoder (``embeddings.paecter.PaecterEncoder`` fits)."""

    model_id: str
    revision: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class PatentVoteStore(Protocol):
    """Patent-corpus access for the vote. ``DbPatentVoteStore`` is the database implementation."""

    def serving_corpus(self, session: Session) -> patents_db.Corpus | None: ...

    def document_count(self, session: Session, corpus: patents_db.Corpus) -> int: ...

    def search(self, session: Session, corpus_id: str, query_vector: Any, k: int,
               ef_search: int) -> list[PatentNeighbour]: ...  # fmt: skip


class DbPatentVoteStore:
    """Reads the serving corpus from ``patent_corpora`` on every request (activation needs no restart) and
    searches its partial HNSW index with ``patent_search.search_patents``."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, Any], int] = {}

    def serving_corpus(self, session: Session) -> patents_db.Corpus | None:
        return patents_db.get_serving_corpus(session)

    def document_count(self, session: Session, corpus: patents_db.Corpus) -> int:
        # patent_corpora.document_count stays 0 until a job records it (set_corpus_state(document_count=...));
        # provenance must carry the real count, so count the rows once per corpus version instead.
        if corpus.document_count > 0:
            return corpus.document_count
        key = (corpus.corpus_id, corpus.updated_at)
        if key not in self._counts:
            n = session.execute(text("SELECT count(*) FROM patent_documents WHERE corpus_id = :c"),
                                {"c": corpus.corpus_id}).scalar_one()
            self._counts[key] = int(n)
        return self._counts[key]

    def search(self, session: Session, corpus_id: str, query_vector: Any, k: int, ef_search: int
               ) -> list[PatentNeighbour]:  # fmt: skip
        return search_patents(session, corpus_id, query_vector, k, ef_search)


@dataclass(frozen=True)
class VoteCode:
    """A vote code as reported: display code at a catalogue level, its rounded shares and its score."""

    display_code: str
    level: str  # class | subclass | maingroup | subgroup
    share: VoteShare
    score: float  # rounded share at its own level (class: the class-level share)


@dataclass
class VoteAttempt:
    """What happened when the neighbour vote was tried. ``reason`` None means the vote is primary."""

    reason: str | None
    warnings: list[str] = field(default_factory=list)
    vote: NeighbourVote | None = None
    result: VoteResult | None = None  # the contract mapping, rounded; kept only when the response carries it
    encoder_info: PatentEncoderInfo | None = None
    corpus_info: PatentCorpusInfo | None = None
    settings: VoteSettings | None = None

    @property
    def primary(self) -> bool:
        return self.reason is None

    def fail(self, reason: str, warning: str) -> VoteAttempt:
        self.reason = reason
        self.warnings.append(warning)
        if reason != "vote_below_minimum":
            self.result = None
        return self


def _r(x: float) -> float:
    return round(float(x), SHARE_DECIMALS)


def _pct(x: float) -> str:
    return f"{round(100 * x, 1)}%"


def internal_code(display: str) -> str:
    """Inverse of ``ingestion.wipo.display_symbol``: "H04L 9/08" -> "H04L0009080000". Sections, classes and
    subclasses are their own internal code."""
    if " " not in display:
        return display
    sub, group = display.split(" ", 1)
    mg, sg = group.split("/", 1)
    return f"{sub}{int(mg):04d}{sg.ljust(6, '0')}"


def code_level(display: str) -> str:
    """Catalogue level of a vote display code (vote codes are never sections)."""
    if "/" in display:
        return "maingroup" if display.endswith("/00") else "subgroup"
    return "class" if len(display) == 3 else "subclass"


def _strict_ancestors(display: str, level: str) -> set[str]:
    return {a for a in vote_code_ancestors(display, level).values() if a != display}


def _nested(a: VoteCode, b: VoteCode) -> bool:
    return (a.display_code == b.display_code or a.display_code in _strict_ancestors(b.display_code, b.level)
            or b.display_code in _strict_ancestors(a.display_code, a.level))


def vote_share_for(v: NeighbourVote, display: str, level: str) -> VoteShare:
    """Contract ``vote_share`` of one code, rounded. A main-group code's ``exact`` is the share labelled with
    that main group itself (``v.share("exact", mg)``, 0.0 when nobody is), never None."""
    if level == "class":
        return VoteShare()
    sub = _r(v.share("subclass", display[:4]))
    if level == "subclass":
        return VoteShare(subclass=sub)
    # A resolved label always nests (exact within its main group within its subclass), so the shares are
    # monotone; min() only guards against a catalogue that lacks a main group whose subgroup it has.
    mg = min(_r(v.share("maingroup", display.split("/")[0] + "/00")), sub)
    exact = min(_r(v.share("exact", display)), mg)
    return VoteShare(subclass=sub, maingroup=mg, exact=exact)


def class_shares(v: NeighbourVote) -> list[tuple[str, float]]:
    """Class-level shares (neighbours carrying any code in the class), descending, unrounded.

    Same rule as ``neighbour_vote.vote`` at the other levels: the main label counts 1, secondary labels
    ``beta``, a neighbour counts once per class with its best label weight; ties keep first-carrier order.
    """
    score: dict[str, float] = {}
    for n in v.neighbours:
        best: dict[str, float] = {}
        for j, label in enumerate(n.labels):
            lw = 1.0 if j == 0 else v.beta
            cls = label[0][:3] if label[0] else None
            if lw > 0 and cls and lw > best.get(cls, 0.0):
                best[cls] = lw
        for cls, lw in best.items():
            score[cls] = score.get(cls, 0.0) + n.vote_weight * lw
    return [(c, min(1.0, score[c])) for c in sorted(score, key=score.__getitem__, reverse=True)]


def vote_result(v: NeighbourVote) -> VoteResult:
    """``NeighbourVote`` -> contract ``VoteResult`` (shares rounded to 4 decimals)."""
    assert v.first_code is not None and v.first_code_level is not None
    return VoteResult(
        first_code=v.first_code,
        first_code_level=v.first_code_level,
        first_code_share=vote_share_for(v, v.first_code, v.first_code_level),
        top_codes=VoteTopCodes(**{
            lvl: [CodeShare(display_code=rc.code, share=_r(rc.share)) for rc in v.top_codes(lvl, MAX_TOP_CODES_PER_LEVEL)]
            for lvl in LEVELS
        }),  # fmt: skip
        neighbour_count=v.neighbour_count,
        neighbours=[
            NeighbourPatent(document_id=str(n.document_id), title=n.title,
                            similarity=max(-1.0, min(1.0, _r(n.similarity))), vote_weight=_r(n.vote_weight),
                            ipc_codes=n.ipc_codes)
            for n in v.nearest(MAX_NEIGHBOURS)
        ],  # fmt: skip
    )


def vote_debug(v: NeighbourVote) -> dict[str, list[str]]:
    """retrieval_debug keys for a vote that ran (benchmark scoring of candidate generation)."""
    out = {f"vote_{lvl}": [rc.code for rc in v.ranked[lvl][:VOTE_DEBUG_CODES]] for lvl in LEVELS}
    out["vote_neighbour_ids"] = [str(n.document_id) for n in v.neighbours]
    return out


def select_vote_codes(v: NeighbourVote, target_level: str, max_results: int,
                      min_share: float) -> tuple[list[VoteCode], list[VoteCode]]:  # fmt: skip
    """Codes the vote reports, in vote order: (candidates, overflow beyond max_results).

    The first is the vote's first code (always computed at the deepest level), reported as its ancestor
    when ``target_level`` is coarser. Further codes come from the vote ranking at the requested level
    (``group``: the neighbours' exact label codes) when their rounded own-level share is >= ``min_share``;
    a code nested with one already chosen (its ancestor or descendant) is skipped.
    """
    assert v.first_code is not None and v.first_code_level is not None
    first, first_level = v.first_code, v.first_code_level
    classes = class_shares(v) if target_level == "class" else []
    class_score = {c: _r(s) for c, s in classes}

    def make(display: str, level: str) -> VoteCode:
        if level == "class":
            return VoteCode(display, level, VoteShare(), class_score.get(display, 0.0))
        share = vote_share_for(v, display, level)
        score = own_level_share(level, share)
        assert score is not None
        return VoteCode(display, level, share, score)

    want = target_level if target_level in ("class", "subclass", "maingroup") else None
    ancestors = vote_code_ancestors(first, first_level)
    lead = make(ancestors[want], want) if want in ancestors else make(first, first_level)

    if target_level == "class":
        ranking = [(c, "class") for c, _ in classes]
    elif target_level in ("subclass", "maingroup"):
        ranking = [(rc.code, target_level) for rc in v.ranked[target_level]]
    else:  # group / subgroup: the exact label codes (subgroups, or main groups labelled as such)
        ranking = [(rc.code, code_level(rc.code)) for rc in v.ranked["exact"]]
        if target_level == "subgroup":
            ranking = [(c, lvl) for c, lvl in ranking if lvl == "subgroup"]

    chosen, overflow = [lead], []
    for display, level in ranking:
        if len(chosen) + len(overflow) >= 2 * max_results:
            break
        vc = make(display, level)
        if vc.score <= 0 or vc.score < min_share or any(_nested(vc, c) for c in (*chosen, *overflow)):
            continue
        (chosen if len(chosen) < max_results else overflow).append(vc)
    return chosen, overflow


def _savepoint(session: Any) -> contextlib.AbstractContextManager[Any]:
    """A SAVEPOINT around the vote's database work, so a failed search leaves the request transaction usable
    for the definition fallback."""
    begin_nested = getattr(session, "begin_nested", None)
    return begin_nested() if begin_nested is not None else contextlib.nullcontext()


class ClassificationPipeline:
    def __init__(
        self,
        settings: Settings,
        provider: EmbeddingProvider,
        reranker: Reranker,
        confidence: ConfidenceService,
        mapping: FieldMapping,
        llm_client: llm_mod.LLMClient | None = None,
        *,
        patent_encoder: PatentEncoderLike | None = None,
        patent_store: PatentVoteStore | None = None,
    ) -> None:
        """``patent_encoder`` (PaECTER) and ``patent_store`` are optional: without an encoder the vote reports
        ``patent_encoder_unavailable`` (when enabled); the store defaults to the database (serving corpus
        looked up per request)."""
        self.s = settings
        self.provider = provider
        self.reranker = reranker
        self.confidence = confidence
        self.mapping = mapping
        self.llm_client = llm_client
        self.patent_encoder = patent_encoder
        self.patent_store: PatentVoteStore = patent_store if patent_store is not None else DbPatentVoteStore()

    # ------------------------------------------------------------------ helpers
    def retrieval_settings(self) -> dict[str, int | float | str | bool | None]:
        s = self.s
        return {"vector_pool": s.vector_pool, "keyword_pool": s.keyword_pool, "rerank_pool": s.rerank_pool,
                "rrf_k": s.rrf_k, "hnsw_ef_search": s.hnsw_ef_search, "vector_search_mode": s.vector_search_mode, "keyword_max_terms": s.keyword_max_terms,
                "expansion_top_n": s.expansion_top_n, "expansion_max_siblings": s.expansion_max_siblings,
                "max_query_views": s.max_query_views, "per_source_candidate_cap": s.per_source_candidate_cap,
                "subclass_routing": "group_vote", "routing_group_pool": s.routing_group_pool,
                "hierarchical_subclass_limit": s.hierarchical_subclass_limit,
                "hierarchical_retrieval": True,
                "claims_retrieval": ("excluded" if s.claims_retrieval_mode == "off" else
                                     f"independent_weighted(independent={s.independent_claim_weight},"
                                     f"dependent={s.dependent_claim_weight},source={s.claim_source_weight})"),
                "keyword_mode": "postgresql_fulltext_ts_rank_cd", "fusion": "rrf_two_level_weighted_source_caps",
                "rerank_official_context": s.rerank_official_context,
                "min_reranker_score": s.min_reranker_score, "max_reranker_gap": s.max_reranker_gap,
                "min_evidence_score": s.min_evidence_score, "enforce_lexical_checks": s.enforce_lexical_checks,
                "apply_precedence_rules": s.apply_precedence_rules, "apply_place_rules": s.apply_place_rules}  # fmt: skip

    def _token_offsets(self):
        fn = getattr(self.provider, "token_offsets", None)
        return fn if fn is not None else regex_token_offsets

    def _check_limits(self, req: ClassifyRequest) -> None:
        s = self.s
        limits = {"abstract": s.max_abstract_chars, "title": s.max_title_chars,
                  "specification": s.max_specification_chars, "claims": s.max_claims_chars}
        over = {k: {"chars": len(v), "limit": limits[k]} for k in limits
                if (v := getattr(req, k)) is not None and len(v) > limits[k]}
        if over:
            raise InputRejected("request_too_large", "One or more fields exceed configured limits.", over)
        count = getattr(self.provider, "token_count", None)
        if count is not None:
            n = count(f"{req.title}\n{req.abstract}" if req.title else req.abstract)
            if n > s.abstract_max_tokens:
                raise InputRejected(
                    "abstract_too_long",
                    f"Title+abstract is {n} tokens; the configured limit is {s.abstract_max_tokens}. "
                    "Shorten the abstract or move detail into the specification field.",
                    {"tokens": n, "limit": s.abstract_max_tokens},
                )
        if req.max_results > s.max_results_limit:
            raise InputRejected("invalid_max_results", f"max_results must be <= {s.max_results_limit}.")

    def _inputs(self, req: ClassifyRequest):
        """Fields, input mode and source-labelled query views (shared by both classifiers)."""
        fields: dict[str, str] = {"abstract": req.abstract}
        for k in ("title", "specification", "claims"):
            if getattr(req, k):
                fields[k] = getattr(req, k)
        optional_supplied = bool(req.specification or req.claims)
        input_mode = "abstract_with_optional_documents" if optional_supplied else "abstract_only"
        plan = build_views(req.abstract, req.title, req.claims, req.specification,
                           max_views=self.s.max_query_views, chunk_tokens=self.provider.policy.query_chunk_tokens,
                           chunk_overlap=self.s.query_chunk_overlap_tokens, token_offsets=self._token_offsets())
        return fields, input_mode, plan

    def _terms(self, retrieval_views: list[QueryView], vectors: list[Any] | None = None):
        """Extracted keyword terms per distinct view text, and the (view, vector, websearch query) triples."""
        extracted_terms: list[dict[str, str | float]] = []
        prepared_views: list[tuple[Any, Any, str]] = []
        seen_texts: set[str] = set()
        vecs = vectors if vectors is not None else [None] * len(retrieval_views)
        for v, vec in zip(retrieval_views, vecs, strict=True):
            key = v.text.strip()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            terms = extract_terms(v.text, self.s.keyword_max_terms)
            if v.is_background:
                terms = terms[: max(3, self.s.keyword_max_terms // 4)]
            extracted_terms += [{"term": tm.text, "source": v.label(), "weight": round(tm.weight, 2)}
                                for tm in terms]
            prepared_views.append((v, vec, build_websearch_query(terms)))
        return extracted_terms, prepared_views

    def _provenance(self, version: str, release: repo.Release, run_id: str | None, calibrator_id: str | None,
                    llm_model: str | None, attempt: VoteAttempt) -> Provenance:  # fmt: skip
        pol = self.provider.policy
        return Provenance(
            taxonomy_version=version,
            source_snapshot=release.source_hashes,
            embedding_provider=pol.provider,
            embedding_model=pol.model_id,
            model_revision=pol.model_revision,
            embedding_run_id=run_id,
            reranker_model=self.reranker.model_id,
            reranker_revision=self.reranker.revision,
            scope_formatter=SCOPE_FORMATTER_VERSION,
            calibrator_id=calibrator_id,
            field_mapping_version=self.mapping.version,
            llm_model=llm_model,
            retrieval_settings=self.retrieval_settings(),
            encoding_policy={"dimension": pol.dimension, "normalized": pol.normalized, "dtype": pol.dtype,
                             "taxonomy_max_tokens": pol.taxonomy_max_tokens,
                             "query_chunk_tokens": pol.query_chunk_tokens,
                             "abstract_max_tokens": self.s.abstract_max_tokens,
                             "query_template": pol.query_template, "document_template": pol.document_template},
            patent_encoder=attempt.encoder_info,
            patent_corpus=attempt.corpus_info,
            vote_settings=attempt.settings,
        )

    # ------------------------------------------------------------------ main
    def classify(self, session: Session, req: ClassifyRequest) -> ClassifyResponse:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        self._check_limits(req)
        # An invalid taxonomy_version is rejected before any model work. A missing release is held until we
        # know whether the vote could run: if it could not, the error becomes patent_corpus_unavailable.
        release: repo.Release | None = None
        release_error: PipelineUnavailable | None = None
        try:
            release = resolve_release(session, req.taxonomy_version)
        except PipelineUnavailable as exc:
            release_error = exc
        attempt = self._attempt_vote(session, req, timings)
        if attempt.primary:
            if release_error is not None:
                raise release_error
            assert release is not None
            try:
                return self._classify_vote(session, req, release, attempt, t0, timings)
            except _VoteUnusable as exc:
                attempt.fail("vote_failed", f"Neighbour vote failed: {exc} Codes come from definition matching.")
        try:
            if release_error is not None:
                raise release_error
            assert release is not None
            state = serving_run(session, release, self.provider)
            return self._classify_definitions(session, req, state, attempt, t0, timings)
        except PipelineUnavailable as exc:
            if attempt.reason in VOTE_UNAVAILABLE_REASONS:
                raise PipelineUnavailable(
                    "patent_corpus_unavailable",
                    f"The neighbour vote could not run ({attempt.reason}) and the definition fallback could not "
                    f"serve ({exc.code}: {exc})",
                    {"reason": attempt.reason, "fallback_error": exc.code},
                ) from exc
            raise

    # ------------------------------------------------------------------ neighbour vote
    def _attempt_vote(self, session: Session, req: ClassifyRequest, timings: dict[str, float]) -> VoteAttempt:
        s = self.s
        if not s.neighbour_vote_enabled:
            return VoteAttempt("neighbour_vote_disabled", [
                "Neighbour vote is disabled by the operator (NEIGHBOUR_VOTE_ENABLED=false); codes come from "
                "definition matching."])
        enc = self.patent_encoder
        att = VoteAttempt(
            None,
            encoder_info=PatentEncoderInfo(model_id=enc.model_id, model_revision=enc.revision) if enc else None,
            settings=VoteSettings(k=s.neighbour_k, temperature=s.neighbour_temperature, beta=s.neighbour_beta,
                                  min_vote_share=s.neighbour_min_vote_share,
                                  hnsw_ef_search=s.patent_hnsw_ef_search),
        )
        k = s.neighbour_k
        try:
            with _savepoint(session):
                corpus = self.patent_store.serving_corpus(session)
                if corpus is None:
                    return att.fail("no_serving_corpus", "No validated patent corpus is serving; codes come from "
                                                         "definition matching.")
                att.corpus_info = PatentCorpusInfo(
                    corpus_id=str(corpus.corpus_id), source=corpus.source,
                    document_count=self.patent_store.document_count(session, corpus),
                    labels_ipc_edition=corpus.label_ipc_edition,  # DB/Corpus name -> contract name
                )
                if enc is None:
                    return att.fail("patent_encoder_unavailable", "The patent encoder (PaECTER) is not loaded; codes "
                                                                  "come from definition matching.")
                if (enc.model_id, enc.revision) != (corpus.encoder_model_id, corpus.encoder_model_revision):
                    return att.fail(
                        "patent_encoder_unavailable",
                        f"Patent corpus {corpus.corpus_id!r} was embedded with {corpus.encoder_model_id}@"
                        f"{corpus.encoder_model_revision}, but the loaded encoder is {enc.model_id}@{enc.revision}; "
                        "codes come from definition matching.")
                t = time.perf_counter()
                query_vector = enc.encode([patent_text(req.title, req.abstract)])[0]
                timings["patent_encode"] = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                neighbours = self.patent_store.search(session, str(corpus.corpus_id), query_vector, k,
                                                      s.patent_hnsw_ef_search)
                timings["patent_search"] = (time.perf_counter() - t) * 1000
                if len(neighbours) < k:
                    raise NeighbourShortfall(str(corpus.corpus_id), k, list(neighbours))
                t = time.perf_counter()
                v = vote(neighbours, k, s.neighbour_temperature, s.neighbour_beta)
                timings["vote"] = (time.perf_counter() - t) * 1000
        except NeighbourShortfall as exc:
            return att.fail("vote_failed", f"Neighbour vote failed: {exc}. Codes come from definition matching.")
        except Exception as exc:  # noqa: BLE001 - any search/encode failure degrades to the fallback, with the reason
            log.exception("neighbour vote failed")
            return att.fail("vote_failed", f"Neighbour vote failed ({type(exc).__name__}: {str(exc)[:300]}); codes "
                                           "come from definition matching.")
        att.vote = v
        if v.is_empty:
            return att.fail("vote_empty", f"None of the {k} nearest corpus patents has a resolvable IPC label; codes "
                                          "come from definition matching.")
        att.result = vote_result(v)
        share = att.result.first_code_share.subclass
        assert share is not None
        # Gate on the rounded (wire) value, so a client re-checking the response reaches the same verdict.
        if share < s.neighbour_min_vote_share:
            return att.fail(
                "vote_below_minimum",
                f"Neighbour vote too weak ({_pct(share)} for {att.result.first_code[:4]}, minimum "
                f"{_pct(s.neighbour_min_vote_share)}); codes come from definition matching.")
        return att

    def _classify_vote(self, session: Session, req: ClassifyRequest, release: repo.Release, attempt: VoteAttempt,
                       t0: float, timings: dict[str, float]) -> ClassifyResponse:  # fmt: skip
        v, result = attempt.vote, attempt.result
        assert v is not None and result is not None
        version = release.taxonomy_version
        warnings: list[str] = list(attempt.warnings)
        fields, input_mode, plan = self._inputs(req)
        warnings += plan.warnings
        views = plan.views
        retrieval_views, _ = select_retrieval_views(views, self.s)
        extracted_terms, _ = self._terms(retrieval_views)
        if req.section or req.subclass:
            warnings.append("The section/subclass filters apply to definition matching only; the neighbour vote "
                            "chose these codes without them.")

        chosen, overflow = select_vote_codes(v, req.target_level, req.max_results, self.s.neighbour_min_vote_share)

        # --- hydrate --------------------------------------------------------
        t = time.perf_counter()
        wanted = {internal_code(vc.display_code): vc for vc in (*chosen, *overflow)}
        records = self._hydrate(session, version, list(wanted))
        lead = chosen[0]
        lead_code = internal_code(lead.display_code)
        if lead_code not in records:
            raise _VoteUnusable(f"its first code {lead.display_code} is not in taxonomy release {version}.")
        missing = [vc.display_code for code, vc in wanted.items() if code not in records]
        if missing:
            warnings.append(f"Vote code(s) not in taxonomy release {version}, not reported: {', '.join(missing)}.")
        by_code = {code: vc for code, vc in wanted.items() if code in records}
        chosen_codes = [c for c in (internal_code(vc.display_code) for vc in chosen) if c in by_code]
        overflow_codes = [c for c in (internal_code(vc.display_code) for vc in overflow) if c in by_code]
        timings["hydrate"] = (time.perf_counter() - t) * 1000

        # --- rerank (relevance scores and evidence orientation only; never reorders the vote) -------
        t = time.perf_counter()
        spec_views = {vw.view_id: {"rrf": 0.0} for vw in views[1:]
                      if vw.source == "specification" and not vw.is_background}
        pool = [FusedCandidate(code=c, rrf_score=0.0, views=dict(spec_views), origin={"neighbour_vote"})
                for c in (*chosen_codes, *overflow_codes)]
        scored_list, scopes, view_by_id = self._rerank(pool, records, views, warnings)
        scored = {s_.fc.code: s_ for s_ in scored_list}
        timings["rerank"] = (time.perf_counter() - t) * 1000

        # --- evidence (explains the vote codes; never removes one) ---------------------------------
        t = time.perf_counter()
        top = [scored[c] for c in chosen_codes]
        self._evidence(top, req, views, fields, scopes, view_by_id)
        timings["evidence"] = (time.perf_counter() - t) * 1000

        supported, alternatives = self._apply_vote_policy(top, records)
        alternatives += [(scored[c], "Beyond max_results ceiling.") for c in overflow_codes]

        if req.use_llm:
            warnings.append("use_llm is not applied when the neighbour vote chose the codes; the constrained LLM "
                            "step only selects among definition-matching candidates.")

        # --- assemble -------------------------------------------------------
        k = v.k
        cands = [self._candidate(s_, records, input_mode, req, vote=by_code[s_.fc.code], k=k) for s_ in supported]
        alts = [self._alternative(s_, reason, input_mode, req, vote=by_code[s_.fc.code])
                for s_, reason in alternatives]
        foi = field_of_invention(self.mapping, [records[c.code] for c in cands], "No candidate was selected.")
        status = ("candidates_found" if all(c.confidence.status == "calibrated" for c in cands)
                  and not any(c.unresolved_ambiguity for c in cands) else "needs_review")
        if input_mode == "abstract_only":
            warnings.append("Abstract-only inference: not an official classification; scope and rule checks need review.")
        warnings.append(VOTE_CONFIDENCE_WARNING)

        run_id: str | None = None
        try:
            run_id = serving_run(session, release, self.provider).run_id
        except PipelineUnavailable as exc:
            # The vote does not use catalogue embeddings; only the fallback would have needed them.
            warnings.append(f"Catalogue embeddings are unavailable ({exc.code}); the definition fallback could not "
                            "have served this request.")

        ranked_debug = retrieval_debug = None
        if req.include_ranked_debug:
            outcome = {c.fc.code: "supported" for c in supported}
            outcome |= {a.fc.code: f"alternative: {reason}" for a, reason in alternatives if a.fc.code not in outcome}
            ranked_debug = [self._ranked_debug(scored[c], outcome.get(c, "dropped")) for c in (*chosen_codes,
                                                                                                *overflow_codes)]
            retrieval_debug = vote_debug(v)

        summary = build_summary(foi["label"], cands)
        timings["total"] = (time.perf_counter() - t0) * 1000
        return ClassifyResponse(
            status=status,
            taxonomy_version=version,
            classifier="neighbour_vote",
            fallback_reason=None,
            field_of_invention=FieldOfInvention(**foi),
            summary=summary,
            candidates=cands,
            alternatives=alts,
            vote=result,
            input_coverage=InputCoverage(abstract="complete", specification=plan.coverage["specification"],
                                         claims=plan.coverage["claims"], details=plan.counts),
            extracted_terms=extracted_terms[:200],
            warnings=list(dict.fromkeys(warnings)),
            degraded_mode=None,
            provenance=self._provenance(version, release, run_id, None, None, attempt),
            timings_ms={k_: round(v_, 1) for k_, v_ in timings.items()},
            ranked_debug=ranked_debug,
            retrieval_debug=retrieval_debug,
        )

    def _apply_vote_policy(self, top: list[Scored], records: dict[str, Record]
                           ) -> tuple[list[Scored], list[tuple[Scored, str]]]:  # fmt: skip
        """With the vote primary, evidence explains and never vetoes: weak or missing passages and lexical
        mismatches become review notes. WIPO rules still arbitrate between the vote codes (ADR-0004):
        an unconditional precedence demotion moves a code to the alternatives."""
        s = self.s
        ev_min = s.min_evidence_score
        for c in top:
            c.passages = [ps for ps in c.passages if ev_min is None or ps[1] >= ev_min]
            if not c.passages:
                c.issues.append(NO_PASSAGE_NOTE)
            lexical: list[str] = []
            if c.application_context_only:
                lexical.append("Only supported by a list of application devices/contexts; no substantive "
                               "disclosure concerning this subject was found.")
            if c.subclass_missing and not c.subclass_present:
                lexical.append(f"Subclass scope ({c.rec.subclass_code}: {c.rec.subclass_title}) not evidenced "
                               "lexically; none of its distinctive terms appear in the reviewed text: "
                               + ", ".join(c.subclass_missing[:6]) + ".")
            if c.rec.record_type == "subgroup" and c.missing_terms and len(c.missing_terms) >= max(1, len(c.present_terms)):
                lexical.append("Defining feature(s) of this subgroup not found verbatim in the reviewed text: "
                               + ", ".join(c.missing_terms[:6]) + ".")
            c.issues += [f"Scope check (lexical heuristic, needs review): {m}" for m in lexical]

        outcome = apply_scheme_rules([self._rule_view(c) for c in top], [],
                                     precedence=s.apply_precedence_rules, place=s.apply_place_rules)
        kept: list[Scored] = []
        alternatives: list[tuple[Scored, str]] = []
        for c in top:
            c.issues += outcome.notes.get(c.fc.code, [])
            if c.fc.code in outcome.demoted:
                alternatives.append((c, outcome.demoted[c.fc.code]))
                continue
            kept.append(c)
        if not kept:
            # Only a cycle of non-mutual rules can demote every code; the vote's first code then stays.
            lead, reason = alternatives.pop(0)
            lead.issues.append(f"WIPO priority rules demote every vote code in a cycle; kept for review: {reason}")
            kept = [lead]
        elif top[0] not in kept:
            # The first code yielded to a supported code: that code leads (the demotion reason names it).
            reason = next(r for c, r in alternatives if c is top[0])
            winner = next((c for c in kept if f"({c.rec.display_code} takes precedence" in reason), None)
            if winner is not None:
                kept.remove(winner)
                kept.insert(0, winner)
        return self._same_maingroup_notes(kept, records), alternatives

    def _ranked_debug(self, c: Scored, outcome: str) -> RankedDebug:
        return RankedDebug(display_code=c.rec.display_code, code=c.fc.code, title=c.rec.title,
                           reranker_score=round(c.reranker_score, 4),
                           best_evidence_score=round(max(sc for _, sc in c.passages), 4) if c.passages else None,
                           record_type=c.rec.record_type, maingroup_code=c.rec.maingroup_code,
                           application_context_only=c.application_context_only,
                           subclass_scope_ok=not (c.subclass_missing and not c.subclass_present),
                           defining_terms_ok=not (c.rec.record_type == "subgroup" and c.missing_terms
                                                  and len(c.missing_terms) >= max(1, len(c.present_terms))),
                           outcome=outcome)

    # ------------------------------------------------------------------ shared stages
    def _hydrate(self, session: Session, version: str, codes: list[str], extra_codes: Sequence[str] = ()
                 ) -> dict[str, Record]:  # fmt: skip
        """Records for ``codes`` (+ ``extra_codes``) and their ancestors, with references for ``codes`` and the
        version-matched WIPO texts of each record and its ancestors (nearest owner first)."""
        records = repo.get_records(session, version, list(dict.fromkeys([*codes, *extra_codes])))
        ancestor_codes = [a for r in records.values() for a in r.ancestor_codes if a not in records]
        records.update(repo.get_records(session, version, ancestor_codes))
        refs = repo.get_references(session, version, codes)
        for code, rec in records.items():
            rec.references = refs.get(code, [])
        context_symbols = list(dict.fromkeys(
            [*records, *(ancestor for rec in records.values() for ancestor in rec.ancestor_codes)]
        ))
        official_context = repo.get_official_texts(session, version, context_symbols)
        for rec in records.values():
            owners = [rec.code, *reversed(rec.ancestor_codes)]
            seen_texts: set[tuple] = set()
            rec.official_context = []
            for owner in owners:  # nearest owner first; a range note can match the code and an ancestor
                for item in official_context.get(owner, []):
                    key = (item["owner_symbol"], item["source_kind"], item["part_type"], item["ordinal"])
                    if key not in seen_texts:
                        seen_texts.add(key)
                        rec.official_context.append(item)
        return records

    def _rerank(self, pool: list[FusedCandidate], records: dict[str, Record], views: list[QueryView],
                warnings: list[str]) -> tuple[list[Scored], dict[str, str], dict[str, QueryView]]:  # fmt: skip
        view_by_id = {v.view_id: v for v in views}
        scopes = {c.code: format_scope(records[c.code], records, official=self.s.rerank_official_context)
                  for c in pool}
        pairs: list[tuple[str, str]] = []
        pair_keys: list[tuple[str, str]] = []
        # Orientation: the shorter text is the query. The candidate scope document (short) is the query and
        # the abstract/claim/specification view (long) is the passage, matching the reranker's training format.
        for c in pool:
            pairs.append((scopes[c.code], views[0].text))
            pair_keys.append((c.code, "abstract"))
            # Specification can contribute to candidate relevance; claims are evidence-only.
            opt = sorted(((vid, e["rrf"]) for vid, e in c.views.items()
                          if vid != "abstract" and vid in view_by_id
                          and view_by_id[vid].source != "claims" and not view_by_id[vid].is_background),
                         key=lambda kv: -float(kv[1] or 0))[:2]
            for vid, _ in opt:
                pairs.append((scopes[c.code], view_by_id[vid].text))
                pair_keys.append((c.code, vid))
        scores = self.reranker.score(pairs)
        if any(sc.query_truncated or sc.passage_truncated for sc in scores):
            warnings.append("Some reranker inputs exceeded the reranker token budget and were truncated "
                            "(candidate wording is placed first so optional context is cut first).")
        per_code: dict[str, dict[str, float]] = {}
        for (code, vid), sc in zip(pair_keys, scores, strict=True):
            per_code.setdefault(code, {})[vid] = sc.score
        scored: list[Scored] = []
        for c in pool:
            vs = per_code.get(c.code, {})
            best_vid = max(vs, key=lambda k: vs[k])
            scored.append(Scored(c, records[c.code], vs[best_vid],
                                 "abstract" if best_vid == "abstract" else view_by_id[best_vid].source, vs))
        scored.sort(key=lambda s_: (-s_.reranker_score, s_.fc.code))
        return scored, scopes, view_by_id

    def _evidence(self, top: list[Scored], req: ClassifyRequest, views: list[QueryView], fields: dict[str, str],
                  scopes: dict[str, str], view_by_id: dict[str, QueryView]) -> None:  # fmt: skip
        abstract_passages = split_passages(req.abstract, source="abstract")
        if req.title:
            abstract_passages += split_passages(req.title, source="title")
        opt_passages: dict[str, list[Passage]] = {}
        for v in views[1:]:
            if v.is_background:
                continue
            meta = {"claim_number": v.claim_number, "section": v.section}
            opt_passages[v.view_id] = split_passages(v.text, base=v.start, source=v.source, **meta)
        ev_pairs, ev_keys = [], []
        claim_review_limit = min(len(top), max(3, req.max_results)) if req.claims else 0
        for i, s_ in enumerate(top):
            scope = scopes[s_.fc.code]
            cand_passages = list(abstract_passages)
            for vid in s_.view_scores:
                if view_by_id.get(vid) and view_by_id[vid].source == "specification":
                    cand_passages += opt_passages.get(vid, [])
            # Claim passages are review evidence for only the highest-ranked candidates. They do
            # not enter the retrieval fusion or candidate reranker score.
            if i < claim_review_limit:
                cand_passages += [p for vid, passages in opt_passages.items()
                                  if view_by_id[vid].source == "claims" for p in passages]
            # Evidence orientation: a sentence/clause (short) is the query, the scope document the passage.
            for p in cand_passages:
                ev_pairs.append((p.quote, scope))
                ev_keys.append((s_.fc.code, p))
        ev_scores = self.reranker.score(ev_pairs)
        by_code: dict[str, list[tuple[Passage, float]]] = {}
        for (code, p), sc in zip(ev_keys, ev_scores, strict=True):
            by_code.setdefault(code, []).append((p, sc.score))
        reviewed_text = "\n".join([req.title or "", req.abstract] + [v.text for v in views[1:] if not v.is_background])
        for s_ in top:
            ranked = sorted(by_code.get(s_.fc.code, []), key=lambda ps: -ps[1])
            chosen: list[tuple[Passage, float]] = []
            best_abs = next((ps for ps in ranked if ps[0].source in ("abstract", "title")), None)
            if best_abs:
                chosen.append(best_abs)
            best_opt = next((ps for ps in ranked if ps[0].source in ("claims", "specification")), None)
            if best_opt and (not best_abs or best_opt[1] > best_abs[1]):
                chosen.append(best_opt)
            for p, _ in chosen:
                verify(p, fields)
            s_.passages = chosen
            s_.present_terms, s_.missing_terms = defining_feature_check(s_.rec.title, reviewed_text)
            s_.subclass_present, s_.subclass_missing = subclass_scope_check(s_.rec.subclass_title or "", reviewed_text)
            ev_min = self.s.min_evidence_score
            strong = [ps for ps in chosen if ev_min is None or ps[1] >= ev_min]
            s_.application_context_only = bool(strong) and all(_is_enumeration(p.quote) for p, _ in strong)

    # ------------------------------------------------------------------ definition matching
    def _classify_definitions(self, session: Session, req: ClassifyRequest, state: ServingState,
                              attempt: VoteAttempt, t0: float, timings: dict[str, float]) -> ClassifyResponse:  # fmt: skip
        """The 1.x definition pipeline: the fallback, or the only classifier while the vote is disabled."""
        version = state.release.taxonomy_version
        types = LEVEL_TYPES[req.target_level]
        warnings: list[str] = list(attempt.warnings)

        fields, input_mode, plan = self._inputs(req)
        warnings += plan.warnings
        views = plan.views

        retrieval_views, view_weights = select_retrieval_views(views, self.s)
        source_weights = {"claims": self.s.claim_source_weight}
        t = time.perf_counter()
        abstract_vec = self.provider.encode_queries(
            [retrieval_views[0].text], max_tokens=self.s.abstract_max_tokens
        )[0]
        other_vectors = (self.provider.encode_queries([v.text for v in retrieval_views[1:]])
                         if len(retrieval_views) > 1 else np.zeros((0, abstract_vec.shape[0]), dtype=np.float32))
        retrieval_vectors = [abstract_vec, *list(other_vectors)]
        timings["encode"] = (time.perf_counter() - t) * 1000

        # --- hierarchical retrieval ----------------------------------------
        t = time.perf_counter()
        extracted_terms, prepared_views = self._terms(retrieval_views, retrieval_vectors)

        def retrieve_stage(
            record_types: tuple[str, ...],
            *,
            pool_size: int,
            subclass: str | None = None,
            subclasses: list[str] | None = None,
        ) -> list[ViewHits]:
            hits: list[ViewHits] = []
            for view, vec, query in prepared_views:
                kw = keyword_search(session, version=version, query=query, limit=pool_size,
                                    record_types=record_types, section=req.section, subclass=subclass,
                                    subclasses=subclasses)
                vh = vector_search(session, table=state.storage_table, version=version, query_vector=vec,
                                   limit=pool_size, ef_search=min(self.s.hnsw_ef_search, self.s.hnsw_ef_search_max),
                                   record_types=record_types, section=req.section, subclass=subclass,
                                   subclasses=subclasses, exact=self.s.vector_search_mode == "exact")
                if len(vh) < pool_size:
                    warnings.append(f"Vector retrieval for {view.label()} returned {len(vh)} of {pool_size} "
                                    "requested in this stage.")
                hits.append(ViewHits(view.view_id, view.source,
                                     [(h.code, h.rank, h.score) for h in kw],
                                     [(h.code, h.rank, h.cosine_similarity) for h in vh],
                                     weight=view_weights.get(view.view_id, 1.0)))
            return hits

        routed = req.target_level in ("group", "maingroup", "subgroup")
        stage1_subclasses: list[str] = []
        stage2_candidates: list[str] = []
        if routed:
            if req.subclass:
                stage1_subclasses = [req.subclass]
            else:
                # Subclasses are voted from a deep group search (each scores the sum of its groups' fused
                # RRF). Matching subclass headings alone routed poorly: dev recall@20 0.32 vs 0.61.
                deep_hits = retrieve_stage(("maingroup", "subgroup"), pool_size=self.s.routing_group_pool)
                deep = fuse(deep_hits, k=self.s.rrf_k, per_source_cap=self.s.routing_group_pool,
                            source_weights=source_weights)
                stage1_subclasses = vote_subclasses(deep, self.s.hierarchical_subclass_limit)
            if not stage1_subclasses:
                warnings.append("Subclass routing returned no route; group retrieval fell back to the "
                                "whole selected section/catalogue.")
            group_hits = retrieve_stage(types, pool_size=max(self.s.keyword_pool, self.s.vector_pool),
                                        subclasses=stage1_subclasses or None)
            fused = fuse(group_hits, k=self.s.rrf_k,
                         per_source_cap=self.s.per_source_candidate_cap,
                         source_weights=source_weights)
            stage2_candidates = [c.code for c in fused[:100]]
        else:
            direct_hits = retrieve_stage(types, pool_size=max(self.s.keyword_pool, self.s.vector_pool),
                                         subclass=req.subclass)
            fused = fuse(direct_hits, k=self.s.rrf_k, per_source_cap=self.s.per_source_candidate_cap,
                         source_weights=source_weights)

        pool = expand(session, version=version, table=state.storage_table, fused=fused, abstract_vector=abstract_vec,
                      top_n=self.s.expansion_top_n, max_siblings=self.s.expansion_max_siblings,
                      pool_cap=self.s.rerank_pool, allowed_types=set(types))
        timings["retrieve"] = (time.perf_counter() - t) * 1000

        # --- hydrate --------------------------------------------------------
        codes = [c.code for c in pool]
        records = self._hydrate(session, version, codes, stage2_candidates[:100])
        pool = [c for c in pool if c.code in records and records[c.code].record_type in types]

        # --- rerank ---------------------------------------------------------
        t = time.perf_counter()
        scored, scopes, view_by_id = self._rerank(pool, records, views, warnings)
        timings["rerank"] = (time.perf_counter() - t) * 1000

        # --- evidence -------------------------------------------------------
        t = time.perf_counter()
        n_eval = min(len(scored), max(3 * req.max_results + 5, 15), 30)
        top = scored[:n_eval]
        self._evidence(top, req, views, fields, scopes, view_by_id)
        timings["evidence"] = (time.perf_counter() - t) * 1000

        # --- policy ---------------------------------------------------------
        supported, alternatives = self._apply_policy(top, req, records, input_mode)
        ranked_debug = None
        if req.include_ranked_debug:
            outcome = {c.fc.code: "supported" for c in supported}
            outcome |= {a.fc.code: f"alternative: {reason}" for a, reason in alternatives if a.fc.code not in outcome}
            ranked_debug = [self._ranked_debug(c, outcome.get(c.fc.code, "dropped: below MIN_RERANKER_SCORE"))
                            for c in top]
        retrieval_debug = None
        if req.include_ranked_debug:
            retrieval_debug = {
                "stage1_subclasses": stage1_subclasses,
                "stage2_group_candidates": [records[code].display_code for code in stage2_candidates
                                            if code in records],
                "reranked_candidates": [s_.rec.display_code for s_ in scored[:100]],
            }
            if attempt.vote is not None:  # a weak or empty vote: still scoreable by the benchmark
                retrieval_debug |= vote_debug(attempt.vote)

        # --- optional LLM ---------------------------------------------------
        degraded: str | None = None
        llm_model: str | None = None
        if req.use_llm:
            if not self.s.llm_enabled or self.llm_client is None:
                warnings.append("use_llm was requested but the LLM step is disabled by the operator; "
                                "reranker-only result returned.")
            else:
                supported, alternatives, degraded, llm_model = self._llm_stage(
                    top, supported, alternatives, fields, views, records)

        # --- assemble -------------------------------------------------------
        cands = [self._candidate(s_, records, input_mode, req) for s_ in supported[: req.max_results]]
        overflow = supported[req.max_results :]
        alts = [self._alternative(s_, reason, input_mode, req) for s_, reason in alternatives]
        alts += [self._alternative(s_, "Beyond max_results ceiling.", input_mode, req) for s_ in overflow]

        no_support_reason = (
            "No candidate had a defensible supporting passage under the evidence policy; the input may lack "
            "the technical detail needed to classify it."
        )
        foi = field_of_invention(self.mapping, [records[c.code] for c in cands], no_support_reason)
        if not cands:
            status = "insufficient_evidence"
            warnings.append("Insufficient evidence: " + no_support_reason)
        elif all(c.confidence.status == "calibrated" for c in cands) and not any(c.unresolved_ambiguity for c in cands):
            status = "candidates_found"
        else:
            status = "needs_review"
        if input_mode == "abstract_only":
            warnings.append("Abstract-only inference: not an official classification; scope and rule checks need review.")
        conf_status = self.confidence.status()
        if conf_status["state"] != "calibrated":
            warnings.append("Confidence: not calibrated — no validated calibrator is active for this pipeline; "
                            "numeric confidence is null.")

        summary = build_summary(foi["label"], cands)
        timings["total"] = (time.perf_counter() - t0) * 1000
        return ClassifyResponse(
            status=status,
            taxonomy_version=version,
            classifier="definition_fallback",
            fallback_reason=attempt.reason,
            field_of_invention=FieldOfInvention(**foi),
            summary=summary,
            candidates=cands,
            alternatives=alts,
            # The contract keeps a weak vote visible (why the fallback ran); every other fallback has none.
            vote=attempt.result if attempt.reason == "vote_below_minimum" else None,
            input_coverage=InputCoverage(abstract="complete", specification=plan.coverage["specification"],
                                         claims=plan.coverage["claims"], details=plan.counts),
            extracted_terms=extracted_terms[:200],
            warnings=list(dict.fromkeys(warnings)),
            degraded_mode=degraded,
            provenance=self._provenance(version, state.release, state.run_id, conf_status.get("calibrator_id"),
                                        llm_model, attempt),
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            ranked_debug=ranked_debug,
            retrieval_debug=retrieval_debug,
        )

    # ------------------------------------------------------------------ policy
    def _rule_view(self, c: Scored) -> RuleCandidate:
        return RuleCandidate(c.fc.code, c.rec.display_code, c.rec.ancestor_codes, c.rec.official_context)

    def _same_maingroup_notes(self, supported: list[Scored], records: dict[str, Record]) -> list[Scored]:
        # Do not discard same-main-group candidates without evaluating the WIPO rule text.
        final: list[Scored] = []
        by_mg: dict[str, Scored] = {}
        for c in supported:
            mg = c.rec.maingroup_code or c.fc.code
            if mg in by_mg:
                keep = by_mg[mg]
                mg_rec = records.get(mg)
                note = (f"Same-main-group candidates {keep.rec.display_code} and {c.rec.display_code} need review "
                        f"against WIPO scope and precedence notes for {mg_rec.display_code if mg_rec else mg}; "
                        "both are retained because a reference or shared parent alone does not decide assignment.")
                keep.issues.append(note)
                c.issues.append(note)
            else:
                by_mg[mg] = c
            final.append(c)
        return final

    def _apply_policy(self, top: list[Scored], req: ClassifyRequest, records: dict[str, Record],
                      input_mode: str) -> tuple[list[Scored], list[tuple[Scored, str]]]:
        """Conservative pre-calibration relevance/evidence policy (documented in docs/classification-policy.md)."""
        s = self.s
        alternatives: list[tuple[Scored, str]] = []
        if not top:
            return [], []
        # Pass 1: content checks. Absolute floors are optional (None = off). The lexical heuristics
        # (application-context, subclass-scope, defining-terms) compare catalogue title words with the
        # input's wording; on the IP India benchmark they rejected most correct candidates, so by default
        # they are reported as unresolved ambiguity instead of rejecting (ENFORCE_LEXICAL_CHECKS=true restores).
        passed: list[Scored] = []
        for c in top:
            if s.min_reranker_score is not None and c.reranker_score < s.min_reranker_score:
                continue  # below relevance floor: not even an alternative
            if s.min_evidence_score is not None and not any(ps[1] >= s.min_evidence_score for ps in c.passages):
                alternatives.append((c, "No supporting passage met the evidence policy."))
                continue
            if not c.passages:
                alternatives.append((c, "No supporting passage could be extracted."))
                continue
            lexical: list[str] = []
            if c.application_context_only:
                lexical.append("Only supported by a list of application devices/contexts; no substantive "
                               "disclosure concerning this subject was found.")
            if c.subclass_missing and not c.subclass_present:
                lexical.append(f"Subclass scope ({c.rec.subclass_code}: {c.rec.subclass_title}) not evidenced "
                               "lexically; none of its distinctive terms appear in the reviewed text: "
                               + ", ".join(c.subclass_missing[:6]) + ".")
            if c.rec.record_type == "subgroup" and c.missing_terms and len(c.missing_terms) >= max(1, len(c.present_terms)):
                lexical.append("Defining feature(s) of this subgroup not found verbatim in the reviewed text: "
                               + ", ".join(c.missing_terms[:6]) + ".")
            if lexical and s.enforce_lexical_checks:
                alternatives.append((c, lexical[0]))
                continue
            c.issues += [f"Scope check (lexical heuristic, needs review): {m}" for m in lexical]
            passed.append(c)
        eligible: list[Scored] = []
        if passed:
            best = max(c.reranker_score for c in passed)
            for c in passed:
                if best - c.reranker_score > s.max_reranker_gap:
                    alternatives.append((c, "Relevance well below the best supported candidate; listed for review only."))
                    continue
                eligible.append(c)

        # Hierarchy: prefer a supported subgroup over its own main group; keep main group otherwise.
        # A main group's record carries its own code as maingroup_code, so only subgroups may count.
        supported: list[Scored] = []
        for c in eligible:
            if c.rec.record_type == "maingroup" and any(
                o.rec.record_type == "subgroup" and o.rec.maingroup_code == c.fc.code for o in eligible
            ):
                alternatives.append((c, "Broader main group of a selected subgroup."))
                continue
            supported.append(c)

        # WIPO precedence/place rules between supported candidates; never adds a code.
        supported_codes = {c.fc.code for c in supported}
        outcome = apply_scheme_rules([self._rule_view(c) for c in supported],
                                     [self._rule_view(c) for c in top if c.fc.code not in supported_codes],
                                     precedence=s.apply_precedence_rules, place=s.apply_place_rules)
        kept: list[Scored] = []
        for c in supported:
            c.issues += outcome.notes.get(c.fc.code, [])
            if c.fc.code in outcome.demoted:
                alternatives.append((c, outcome.demoted[c.fc.code]))
                continue
            kept.append(c)
        return self._same_maingroup_notes(kept, records), alternatives

    # ------------------------------------------------------------------ LLM
    def _llm_stage(self, top, supported, alternatives, fields, views, records):
        cand_payload = []
        for s_ in top[: self.s.llm_max_candidates]:
            r = s_.rec
            cand_payload.append({
                "code": r.code, "display_code": r.display_code, "title": r.title,
                "original_wording": r.original_title, "level": r.record_type,
                "official_scope_and_rules": format_scope(r, records),
                "parent_path": [f"{p['display_code']} {p['title']}" for p in parent_path(r, records)],
                "context_heading": r.context_heading,
                "references": [{"kind": x["kind"], "target": x.get("target_display_code") or x["target_code"],
                                "target_title": x.get("target_title"), "wording": x.get("original_wording")}
                               for x in r.references[:6]],
            })
        passages = [{"source": v.source, "claim_number": v.claim_number, "section": v.section, "text": v.text}
                    for v in views[1:] if not v.is_background][: self.s.max_query_views]
        result, degraded = llm_mod.run_constrained(self.llm_client, fields, passages, cand_payload,
                                                   self.s.llm_model or "llm")
        if result is None:
            return supported, alternatives, degraded, self.s.llm_model
        by_code = {s_.fc.code: s_ for s_ in top}
        new_supported: list[Scored] = []
        for sel in result.selections:
            s_ = by_code[sel.code]
            passages_v: list[tuple[Passage, float]] = []
            for q in sel.quotes:
                src, quote = q["source"], q["quote"]
                idx = fields[src].find(quote)
                claim_no = q.get("claim_number")
                if claim_no is not None and src == "claims":
                    span = next((v for v in views if v.source == "claims" and v.claim_number == claim_no), None)
                    if span is None or not (span.start <= idx and idx + len(quote) <= span.end):
                        claim_no = None
                else:
                    claim_no = None
                p = Passage(source=src, quote=quote, start=idx, end=idx + len(quote), claim_number=claim_no)
                verify(p, fields)
                passages_v.append((p, float("nan")))
            s_.passages = passages_v
            s_.issues += sel.unresolved
            s_.issues.append(f"LLM rationale: {sel.rationale}")
            new_supported.append(s_)
        sel_codes = {s_.fc.code for s_ in new_supported}
        new_alts = [(s_, reason) for s_, reason in alternatives if s_.fc.code not in sel_codes]
        new_alts += [(s_, "Policy-supported by reranker but not selected by the constrained LLM comparison.")
                     for s_ in supported if s_.fc.code not in sel_codes]
        for r in result.rejected_alternatives:
            if r["code"] in by_code and r["code"] not in sel_codes and all(a[0].fc.code != r["code"] for a in new_alts):
                new_alts.append((by_code[r["code"]], f"LLM comparison: {r['reason'][:300]}"))
        return new_supported, new_alts, None, self.s.llm_model

    # ------------------------------------------------------------------ assembly
    def _conf(self, s_: Scored, input_mode: str, req: ClassifyRequest, vote: VoteCode | None = None) -> Confidence:
        if vote is not None:
            # No vote-share calibrator exists yet: the raw share at the code's own level, labelled as such.
            return Confidence(score=vote.score, status="vote_share_uncalibrated", calibrator_id=None,
                              interpretation=VOTE_CLASS_INTERPRETATION if vote.level == "class" else VOTE_INTERPRETATION)
        return Confidence(**self.confidence.confidence(s_.reranker_score, input_mode=input_mode,
                                                        level=req.target_level).as_dict())

    def _scores(self, s_: Scored) -> TechnicalScores:
        fc = s_.fc
        return TechnicalScores(cosine_similarity=fc.cosine_similarity, keyword_rank=fc.keyword_rank,
                               keyword_score=fc.keyword_score, vector_rank=fc.vector_rank,
                               rrf_score=fc.rrf_score or None, fused_rank=fc.rank or None,
                               reranker_score=s_.reranker_score, reranker_source=s_.reranker_source)

    def _passages(self, s_: Scored, llm: bool = False) -> list[SupportingPassage]:
        return [SupportingPassage(source=p.source, quote=p.quote, start=p.start, end=p.end,
                                  claim_number=p.claim_number, section=p.section,
                                  evidence_type="llm_cited_verified" if llm else "retrieval_evidence_requires_review",
                                  relevance=None if score != score else round(score, 4))
                for p, score in s_.passages]

    @staticmethod
    def _vote_rationale(vc: VoteCode, k: int) -> str:
        head = f"Neighbour vote: {{}} of the similarity-weighted {k} nearest patents carry "
        d, sh = vc.display_code, vc.share
        if vc.level == "class":
            return head.format(_pct(vc.score)) + f"a code in class {d}."
        assert sh.subclass is not None
        text_ = head.format(_pct(sh.subclass)) + d[:4]
        if vc.level == "subclass" or sh.maingroup is None or sh.exact is None:
            return text_ + "."
        mg = d.split("/")[0] + "/00"
        if vc.level == "maingroup":
            return text_ + f" and {_pct(sh.maingroup)} {mg} ({_pct(sh.exact)} labelled with {mg} itself)."
        return text_ + f", {_pct(sh.maingroup)} {mg} and {_pct(sh.exact)} exactly {d}."

    def _candidate(self, s_: Scored, records: dict[str, Record], input_mode: str, req: ClassifyRequest,
                   vote: VoteCode | None = None, k: int = 0) -> Candidate:  # fmt: skip
        r = s_.rec
        llm_used = any(score != score for _, score in s_.passages)
        passages = self._passages(s_, llm=llm_used)
        depends_optional = bool(passages) and all(p.source in ("claims", "specification") for p in passages)
        ambiguity = list(s_.issues)
        if not llm_used and (vote is None or passages):
            ambiguity.insert(0, "Retrieval evidence requiring review: sentence similarity does not prove that all "
                                "classification conditions are met.")
        if input_mode == "abstract_only":
            ambiguity.append("Abstract-only candidate review; applicable scope notes and priority rules still need "
                             "verification against the official scheme.")
        if r.official_context:
            used = "were included in the ranking input" if self.s.rerank_official_context else "apply to this code"
            ambiguity.append(f"Version-matched WIPO notes, definitions and reference wording {used}; confirm any "
                             "conditional or precedence rule applies to this disclosure.")
            rules = [item for item in r.official_context if item.get("rule_kind") in RULE_REVIEW_KINDS]
            for item in rules[:3]:
                ambiguity.append(f"WIPO {item['rule_kind'].replace('_', ' ')} rule for "
                                 f"{display_symbol(item['owner_symbol'])}: {item['text'][:300]}")
        if s_.missing_terms:
            ambiguity.append("Title terms not found in the reviewed text: " + ", ".join(s_.missing_terms[:8]) + ".")
        if depends_optional:
            ambiguity.append("Support depends on an optional-document passage, not on the abstract alone.")
        refs = [ReferenceInfo(**{k_: x[k_] for k_ in ("target_code", "target_display_code", "target_title", "kind",
                                                     "original_wording", "target_exists")}) for x in r.references]
        if refs:
            ambiguity.append(f"{len(refs)} reference(s) from this place require a scope check; reference conditions "
                             "are not evaluated automatically and referenced codes are not auto-assigned.")
        if r.is_duplicate_symbol:
            ambiguity.append(f"Duplicate main-group symbol in source; context heading: {r.context_heading!r}.")
        best = s_.passages[0][0] if s_.passages else None
        if llm_used:
            rationale = next((i[len("LLM rationale: "):] for i in s_.issues if i.startswith("LLM rationale: ")), "")
            ambiguity = [i for i in ambiguity if not i.startswith("LLM rationale: ")]
        elif vote is not None:
            terms = ", ".join(s_.present_terms[:6]) or "related technical wording"
            rationale = self._vote_rationale(vote, k) + (
                f" The {best.source} passage shares its defining terms ({terms})." if best else
                " No input passage supports it; it needs review.")
        else:
            terms = ", ".join(s_.present_terms[:6]) or "related technical wording"
            within = f" within {r.maingroup_title}" if r.record_type == "subgroup" and r.maingroup_title else ""
            rationale = (f"The {best.source if best else 'input'} passage shares the defining terms ({terms}) of "
                         f"{r.display_code} \"{r.title}\"{within}; ranked by the cross-encoder against the "
                         "candidate's scope wording.")
        return Candidate(
            code=r.code, display_code=r.display_code, title=r.title, original_title=r.original_title,
            level=r.record_type, confidence=self._conf(s_, input_mode, req, vote),
            vote_share=vote.share if vote is not None else VoteShare(), supporting_passages=passages,
            depends_on_optional_documents=depends_optional, rationale=rationale,
            unresolved_ambiguity=list(dict.fromkeys(ambiguity)), missing_defining_terms=s_.missing_terms,
            parent_path=[PathEntry(**p) for p in parent_path(r, records)], references=refs,
            scores=self._scores(s_), origin=sorted(s_.fc.origin),
        )

    def _alternative(self, s_: Scored, reason: str, input_mode: str, req: ClassifyRequest,
                     vote: VoteCode | None = None) -> Alternative:  # fmt: skip
        r = s_.rec
        return Alternative(code=r.code, display_code=r.display_code, title=r.title, level=r.record_type,
                           reason=reason, unresolved_ambiguity=list(s_.issues),
                           supporting_passages=self._passages(s_), confidence=self._conf(s_, input_mode, req, vote),
                           vote_share=vote.share if vote is not None else VoteShare(), scores=self._scores(s_))
