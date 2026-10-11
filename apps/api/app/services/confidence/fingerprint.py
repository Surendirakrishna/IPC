"""Pipeline fingerprint used as the confidence-artifact compatibility key.

Any change to the embedding model/revision/serving run policy, reranker revision, scope formatter,
candidate-generation settings, taxonomy/source hashes or field mapping changes the hash and
invalidates calibrated confidence (score=null, status=incompatible_artifact).
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


def current_fingerprint(settings: Settings, provider: Any, reranker: Any) -> tuple[dict[str, Any], str]:
    release: dict[str, Any] = {"taxonomy_version": None, "source_hashes": None}
    run: dict[str, Any] = {"run_id": None, "input_hash": None}
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
    return pipeline_fingerprint(components)
