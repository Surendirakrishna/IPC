"""End-to-end classification pipeline.

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

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import repository as repo
from app.db.repository import Record
from app.schemas.classify import (
    Alternative,
    Candidate,
    ClassifyRequest,
    ClassifyResponse,
    Confidence,
    FieldOfInvention,
    InputCoverage,
    PathEntry,
    Provenance,
    RankedDebug,
    ReferenceInfo,
    SupportingPassage,
    TechnicalScores,
)
from app.services.classification import llm as llm_mod
from app.services.classification.evidence import (
    Passage,
    defining_feature_check,
    split_passages,
    subclass_scope_check,
    verify,
)
from app.services.classification.reranker import Reranker
from app.services.classification.rules import RuleCandidate, apply_scheme_rules
from app.services.classification.scope import SCOPE_FORMATTER_VERSION, format_scope, parent_path
from app.services.classification.summary import build_summary
from app.services.confidence.calibrator import ConfidenceService
from app.services.embeddings.providers import EmbeddingProvider
from app.services.fields.mapping import FieldMapping, field_of_invention
from app.services.retrieval.expansion import expand
from app.services.retrieval.fusion import FusedCandidate, ViewHits, fuse, vote_subclasses
from app.services.retrieval.keyword import build_websearch_query, extract_terms, keyword_search
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


class PipelineUnavailable(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class InputRejected(ValueError):
    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass
class ServingState:
    release: repo.Release
    run_id: str
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


def serving_state(session: Session, provider: EmbeddingProvider, version: str | None) -> ServingState:
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


def _is_enumeration(quote: str) -> bool:
    return quote.count(",") >= 4 and bool(ENUM_CUE_RE.search(quote))


class ClassificationPipeline:
    def __init__(
        self,
        settings: Settings,
        provider: EmbeddingProvider,
        reranker: Reranker,
        confidence: ConfidenceService,
        mapping: FieldMapping,
        llm_client: llm_mod.LLMClient | None = None,
    ) -> None:
        self.s = settings
        self.provider = provider
        self.reranker = reranker
        self.confidence = confidence
        self.mapping = mapping
        self.llm_client = llm_client

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

    # ------------------------------------------------------------------ main
    def classify(self, session: Session, req: ClassifyRequest) -> ClassifyResponse:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        self._check_limits(req)
        state = serving_state(session, self.provider, req.taxonomy_version)
        version = state.release.taxonomy_version
        types = LEVEL_TYPES[req.target_level]
        warnings: list[str] = []

        fields: dict[str, str] = {"abstract": req.abstract}
        for k in ("title", "specification", "claims"):
            if getattr(req, k):
                fields[k] = getattr(req, k)
        optional_supplied = bool(req.specification or req.claims)
        input_mode = "abstract_with_optional_documents" if optional_supplied else "abstract_only"

        plan = build_views(req.abstract, req.title, req.claims, req.specification,
                           max_views=self.s.max_query_views, chunk_tokens=self.provider.policy.query_chunk_tokens,
                           chunk_overlap=self.s.query_chunk_overlap_tokens, token_offsets=self._token_offsets())
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
        extracted_terms: list[dict[str, str | float]] = []
        prepared_views: list[tuple[Any, np.ndarray, str]] = []
        seen_texts: set[str] = set()
        for v, vec in zip(retrieval_views, retrieval_vectors, strict=True):
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
        stage2_codes = stage2_candidates[:100]
        records = repo.get_records(session, version, list(dict.fromkeys([*codes, *stage2_codes])))
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
        pool = [c for c in pool if c.code in records and records[c.code].record_type in types]

        # --- rerank ---------------------------------------------------------
        t = time.perf_counter()
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
        timings["rerank"] = (time.perf_counter() - t) * 1000

        # --- evidence -------------------------------------------------------
        t = time.perf_counter()
        n_eval = min(len(scored), max(3 * req.max_results + 5, 15), 30)
        top = scored[:n_eval]
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
        timings["evidence"] = (time.perf_counter() - t) * 1000

        # --- policy ---------------------------------------------------------
        supported, alternatives = self._apply_policy(top, req, records, input_mode)
        ranked_debug = None
        if req.include_ranked_debug:
            outcome = {c.fc.code: "supported" for c in supported}
            outcome |= {a.fc.code: f"alternative: {reason}" for a, reason in alternatives if a.fc.code not in outcome}
            ranked_debug = [
                RankedDebug(display_code=c.rec.display_code, code=c.fc.code, title=c.rec.title,
                            reranker_score=round(c.reranker_score, 4),
                            best_evidence_score=round(max(sc for _, sc in c.passages), 4) if c.passages else None,
                            record_type=c.rec.record_type, maingroup_code=c.rec.maingroup_code,
                            application_context_only=c.application_context_only,
                            subclass_scope_ok=not (c.subclass_missing and not c.subclass_present),
                            defining_terms_ok=not (c.rec.record_type == "subgroup" and c.missing_terms
                                                   and len(c.missing_terms) >= max(1, len(c.present_terms))),
                            outcome=outcome.get(c.fc.code, "dropped: below MIN_RERANKER_SCORE"))
                for c in top
            ]
        retrieval_debug = None
        if req.include_ranked_debug:
            retrieval_debug = {
                "stage1_subclasses": stage1_subclasses,
                "stage2_group_candidates": [records[code].display_code for code in stage2_candidates
                                            if code in records],
                "reranked_candidates": [s_.rec.display_code for s_ in scored[:100]],
            }

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
        pol = self.provider.policy
        return ClassifyResponse(
            status=status,
            taxonomy_version=version,
            field_of_invention=FieldOfInvention(**foi),
            summary=summary,
            candidates=cands,
            alternatives=alts,
            input_coverage=InputCoverage(abstract="complete", specification=plan.coverage["specification"],
                                         claims=plan.coverage["claims"], details=plan.counts),
            extracted_terms=extracted_terms[:200],
            warnings=list(dict.fromkeys(warnings)),
            degraded_mode=degraded,
            provenance=Provenance(
                taxonomy_version=version,
                source_snapshot=state.release.source_hashes,
                embedding_provider=pol.provider,
                embedding_model=pol.model_id,
                model_revision=pol.model_revision,
                embedding_run_id=state.run_id,
                reranker_model=self.reranker.model_id,
                reranker_revision=self.reranker.revision,
                scope_formatter=SCOPE_FORMATTER_VERSION,
                calibrator_id=conf_status.get("calibrator_id"),
                field_mapping_version=self.mapping.version,
                llm_model=llm_model,
                retrieval_settings=self.retrieval_settings(),
                encoding_policy={"dimension": pol.dimension, "normalized": pol.normalized, "dtype": pol.dtype,
                                 "taxonomy_max_tokens": pol.taxonomy_max_tokens,
                                 "query_chunk_tokens": pol.query_chunk_tokens,
                                 "abstract_max_tokens": self.s.abstract_max_tokens,
                                 "query_template": pol.query_template, "document_template": pol.document_template},
            ),
            timings_ms={k: round(v, 1) for k, v in timings.items()},
            ranked_debug=ranked_debug,
            retrieval_debug=retrieval_debug,
        )

    # ------------------------------------------------------------------ policy
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
        def rule_view(c: Scored) -> RuleCandidate:
            return RuleCandidate(c.fc.code, c.rec.display_code, c.rec.ancestor_codes, c.rec.official_context)

        supported_codes = {c.fc.code for c in supported}
        outcome = apply_scheme_rules([rule_view(c) for c in supported],
                                     [rule_view(c) for c in top if c.fc.code not in supported_codes],
                                     precedence=s.apply_precedence_rules, place=s.apply_place_rules)
        kept: list[Scored] = []
        for c in supported:
            c.issues += outcome.notes.get(c.fc.code, [])
            if c.fc.code in outcome.demoted:
                alternatives.append((c, outcome.demoted[c.fc.code]))
                continue
            kept.append(c)
        supported = kept

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
        return final, alternatives

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
    def _conf(self, s_: Scored, input_mode: str, req: ClassifyRequest) -> Confidence:
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

    def _candidate(self, s_: Scored, records: dict[str, Record], input_mode: str, req: ClassifyRequest) -> Candidate:
        r = s_.rec
        llm_used = any(score != score for _, score in s_.passages)
        passages = self._passages(s_, llm=llm_used)
        depends_optional = bool(passages) and all(p.source in ("claims", "specification") for p in passages)
        ambiguity = list(s_.issues)
        if not llm_used:
            ambiguity.insert(0, "Retrieval evidence requiring review: sentence similarity does not prove that all "
                                "classification conditions are met.")
        if input_mode == "abstract_only":
            ambiguity.append("Abstract-only candidate review; applicable scope notes and priority rules still need "
                             "verification against the official scheme.")
        if r.official_context:
            ambiguity.append("Version-matched WIPO notes, definitions and reference wording were included in the "
                             "candidate scope; confirm any conditional or precedence rule applies to this disclosure.")
        if s_.missing_terms:
            ambiguity.append("Title terms not found in the reviewed text: " + ", ".join(s_.missing_terms[:8]) + ".")
        if depends_optional:
            ambiguity.append("Support depends on an optional-document passage, not on the abstract alone.")
        refs = [ReferenceInfo(**{k: x[k] for k in ("target_code", "target_display_code", "target_title", "kind",
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
        else:
            terms = ", ".join(s_.present_terms[:6]) or "related technical wording"
            within = f" within {r.maingroup_title}" if r.record_type == "subgroup" and r.maingroup_title else ""
            rationale = (f"The {best.source if best else 'input'} passage shares the defining terms ({terms}) of "
                         f"{r.display_code} \"{r.title}\"{within}; ranked by the cross-encoder against the "
                         "candidate's scope wording.")
        return Candidate(
            code=r.code, display_code=r.display_code, title=r.title, original_title=r.original_title,
            level=r.record_type, confidence=self._conf(s_, input_mode, req), supporting_passages=passages,
            depends_on_optional_documents=depends_optional, rationale=rationale,
            unresolved_ambiguity=list(dict.fromkeys(ambiguity)), missing_defining_terms=s_.missing_terms,
            parent_path=[PathEntry(**p) for p in parent_path(r, records)], references=refs,
            scores=self._scores(s_), origin=sorted(s_.fc.origin),
        )

    def _alternative(self, s_: Scored, reason: str, input_mode: str, req: ClassifyRequest) -> Alternative:
        r = s_.rec
        return Alternative(code=r.code, display_code=r.display_code, title=r.title, level=r.record_type,
                           reason=reason, unresolved_ambiguity=list(s_.issues),
                           supporting_passages=self._passages(s_), confidence=self._conf(s_, input_mode, req),
                           scores=self._scores(s_))
