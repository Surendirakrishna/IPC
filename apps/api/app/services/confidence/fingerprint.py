"""Pipeline fingerprint used as the confidence-artifact compatibility key.

Any change to the embedding model/revision/serving run policy, reranker revision, scope formatter,
candidate-generation settings, taxonomy/source hashes or field mapping changes the hash and
invalidates calibrated confidence (score=null, status=incompatible_artifact).

With the neighbour vote enabled (ADR-0005), the patent encoder (model, revision, encoding policy), the
serving patent corpus (id, input hash) and the vote parameters are part of it too, so a calibrator fitted
on one corpus or encoder is never applied to another. With the vote disabled the component is absent, so
fingerprints of the 1.x pipeline are unchanged.

The serving corpus is read when the fingerprint is computed (startup, ``POST /v1/admin/reload-confidence``).
Activating another corpus while the API runs leaves the fingerprint stale until the next reload.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.services.classification.scope import SCOPE_FORMATTER_VERSION
from app.services.confidence.calibrator import pipeline_fingerprint

CANDIDATE_GENERATION_KEYS = (
    "vector_pool", "keyword_pool", "rerank_pool", "rrf_k", "hnsw_ef_search", "vector_search_mode", "keyword_max_terms",
    "expansion_top_n", "expansion_max_siblings", "max_query_views", "per_source_candidate_cap",
    "query_chunk_tokens", "query_chunk_overlap_tokens", "abstract_max_tokens", "taxonomy_max_tokens",
    "reranker_max_length", "min_reranker_score", "max_reranker_gap", "min_evidence_score",
    "enforce_lexical_checks",
)  # fmt: skip
VOTE_SETTING_KEYS = (
    "neighbour_k", "neighbour_temperature", "neighbour_beta", "neighbour_min_vote_share", "patent_hnsw_ef_search",
)  # fmt: skip
NO_PATENT_CORPUS: dict[str, Any] = {"corpus_id": None, "input_hash": None}


def serving_patent_corpus_identity(session: Any) -> dict[str, Any]:
    """``{corpus_id, input_hash}`` of the serving patent corpus (both None when none serves or the patent tables
    do not exist yet)."""
    from sqlalchemy import text

    has_table = session.execute(text("SELECT to_regclass('public.patent_corpora') IS NOT NULL")).scalar_one()
    if not has_table:
        return dict(NO_PATENT_CORPUS)
    row = session.execute(text("SELECT corpus_id, input_hash FROM patent_corpora WHERE is_serving")).first()
    return {"corpus_id": str(row[0]), "input_hash": row[1]} if row else dict(NO_PATENT_CORPUS)


def neighbour_vote_component(settings: Settings, corpus: dict[str, Any]) -> dict[str, Any]:
    """The vote's part of the fingerprint. The encoder identity is the configured one, which is what the runtime
    loads (``runtime.load_patent_encoder``); a corpus embedded with another revision is never voted on."""
    from app.services.embeddings.paecter import PatentEncodingPolicy

    policy = PatentEncodingPolicy(model_id=settings.patent_encoder_model_id,
                                  model_revision=settings.patent_encoder_model_revision)
    return {
        "patent_encoder": policy.identity(),
        "patent_corpus": {"corpus_id": corpus.get("corpus_id"), "input_hash": corpus.get("input_hash")},
        "vote_settings": {k: getattr(settings, k) for k in VOTE_SETTING_KEYS},
    }


def current_fingerprint(settings: Settings, provider: Any, reranker: Any) -> tuple[dict[str, Any], str]:
    release: dict[str, Any] = {"taxonomy_version": None, "source_hashes": None}
    run: dict[str, Any] = {"run_id": None, "input_hash": None}
    patent_corpus: dict[str, Any] = dict(NO_PATENT_CORPUS)
    try:
        from sqlalchemy import text

        from app.db import repository as repo
        from app.db.session import api_session

        with api_session() as session:
            rel = repo.active_release(session)
            if rel is not None:
                release = {"taxonomy_version": rel.taxonomy_version, "source_hashes": rel.source_hashes}
                row = session.execute(
                    text("SELECT input_hash FROM embedding_runs WHERE taxonomy_version = :v AND storage_table = :t"
                         " AND is_serving"),
                    {"v": rel.taxonomy_version, "t": provider.policy.storage_table},
                ).first()
                if row:
                    run = {"input_hash": row[0]}
            if settings.neighbour_vote_enabled:
                patent_corpus = serving_patent_corpus_identity(session)
    except Exception:  # noqa: BLE001 - DB unavailable: fingerprint without corpus identity
        pass
    from app.services.fields.mapping import load_mapping

    components = {
        "embedding": {k: v for k, v in provider.policy.as_dict().items() if k != "extra"},
        "embedding_run": run,
        "reranker": {"model_id": reranker.model_id, "revision": reranker.revision},
        "scope_formatter": SCOPE_FORMATTER_VERSION,
        "candidate_generation": {k: getattr(settings, k) for k in CANDIDATE_GENERATION_KEYS},
        "taxonomy": release,
        "field_mapping_version": load_mapping(settings.field_mapping_path).version,
    }
    if settings.neighbour_vote_enabled:
        components["neighbour_vote"] = neighbour_vote_component(settings, patent_corpus)
    return pipeline_fingerprint(components)
