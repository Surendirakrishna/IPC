"""GET /health (liveness, no model/DB work), GET /ready, GET /v1/taxonomy/status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.db import repository as repo
from app.db.session import api_session, database_configured
from app.runtime import Runtime, get_runtime

router = APIRouter(tags=["system"])

# checks.neighbour_vote.state (docs/api.md): serving | disabled | no_serving_corpus | patent_encoder_unavailable,
# plus corpus_too_small (document_count < NEIGHBOUR_K: every vote would raise NeighbourShortfall) and unknown (the
# patent corpus could not be read). Only "serving" and "disabled" are not degraded. None of them makes the API
# unready: the definition fallback serves (docs/api.md, /ready).
_fallback_store: Any = None


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _corpus_state() -> dict[str, Any]:
    s = get_settings()
    if not database_configured():
        return {"database": "not_configured"}
    out: dict[str, Any] = {"database": "ok", "target": s.ipc_db_target}
    with api_session() as session:
        has_schema = session.execute(text("SELECT to_regclass('public.taxonomy_releases') IS NOT NULL")).scalar_one()
        if not has_schema:
            return {**out, "schema": "missing"}
        out["schema"] = "ok"
        rel = repo.active_release(session)
        if rel is None:
            return {**out, "active_release": None}
        out["active_release"] = rel.taxonomy_version
        out["source_hashes"] = rel.source_hashes
        counts = session.execute(
            text("SELECT record_type, count(*) FROM ipc_records WHERE taxonomy_version = :v GROUP BY record_type"),
            {"v": rel.taxonomy_version},
        ).all()
        out["record_counts"] = {k: v for k, v in counts}
        out["records_total"] = sum(v for _, v in counts)
        runs = session.execute(
            text("SELECT run_id, provider, model_id, model_revision, storage_table, state, is_serving,"
                 " expected_count, completed_count FROM embedding_runs WHERE taxonomy_version = :v"
                 " AND state <> 'superseded' ORDER BY created_at DESC"),
            {"v": rel.taxonomy_version},
        ).mappings().all()
        out["embedding_runs"] = [dict(r) for r in runs]
        indexes = session.execute(
            text("SELECT tablename, indexname FROM pg_indexes WHERE tablename LIKE 'ipc_embeddings%%'"
                 " AND indexdef ILIKE '%%USING hnsw%%'")
        ).all()
        out["hnsw_indexes"] = [{"table": t, "index": i} for t, i in indexes]
        serving = next((r for r in runs if r["is_serving"] and r["storage_table"] == _serving_table()), None)
        out["serving_embedding_run"] = serving["run_id"] if serving else None
        out["embedding_coverage"] = (
            {"completed": serving["completed_count"], "expected": serving["expected_count"]} if serving else None
        )
        out["validation"] = {
            "ok": (rel.validation_report or {}).get("ok"),
            "source_quality": (rel.validation_report or {}).get("source_quality", []),
        }
    return out


def _serving_table() -> str:
    return "ipc_embeddings" if get_settings().embedding_provider == "bge_m3" else "ipc_embeddings_gemma"


def _vote_store(rt: Runtime) -> Any:
    """The pipeline's patent store (its document-count cache is shared), else a module-level database store."""
    global _fallback_store
    store = getattr(rt.pipeline, "patent_store", None)
    if store is not None:
        return store
    if _fallback_store is None:
        from app.services.classification.pipeline import DbPatentVoteStore

        _fallback_store = DbPatentVoteStore()
    return _fallback_store


def _serving_patent_corpus(rt: Runtime) -> tuple[bool, Any, int | None]:
    """(patent tables exist, serving ``db.patents.Corpus`` or None, its document count or None)."""
    from app.db import patents as patents_db

    with api_session() as session:
        if not session.execute(text("SELECT to_regclass('public.patent_corpora') IS NOT NULL")).scalar_one():
            return False, None, None
        corpus = patents_db.get_serving_corpus(session)
        if corpus is None:
            return True, None, None
        return True, corpus, int(_vote_store(rt).document_count(session, corpus))


def neighbour_vote_state(rt: Runtime) -> dict[str, Any]:
    """``checks.neighbour_vote`` in /ready and ``neighbour_vote`` in /v1/taxonomy/status (docs/api.md).

    The checks run in the pipeline's order (corpus, encoder loaded, encoder revision), so ``state`` names the
    same cause that responses report as ``fallback_reason``. A corpus smaller than k is ``corpus_too_small``:
    the pipeline would answer every request with ``vote_failed`` (NeighbourShortfall).
    """
    s = rt.settings
    enc = rt.patent_encoder
    out: dict[str, Any] = {
        "enabled": rt.neighbour_vote_enabled,
        "state": "disabled",
        "degraded": False,
        "reason": None,
        "patent_encoder": {"model_id": enc.model_id, "model_revision": enc.revision} if enc is not None else None,
        "patent_encoder_error": rt.patent_encoder_error,
        "patent_corpus": None,
        "vote_settings": None,
    }
    if not rt.neighbour_vote_enabled:
        out["reason"] = "Neighbour vote is disabled (NEIGHBOUR_VOTE_ENABLED=false); codes come from definition matching."
        return out
    out["vote_settings"] = {"k": s.neighbour_k, "temperature": s.neighbour_temperature, "beta": s.neighbour_beta,
                            "min_vote_share": s.neighbour_min_vote_share, "hnsw_ef_search": s.patent_hnsw_ef_search}

    def degraded(state: str, reason: str) -> dict[str, Any]:
        out.update(state=state, degraded=True, reason=reason)
        return out

    if not database_configured():
        return degraded("unknown", "Database not configured; the serving patent corpus cannot be read.")
    try:
        tables, corpus, count = _serving_patent_corpus(rt)
    except Exception as exc:  # noqa: BLE001 - reported, never raised: the vote never makes the API unready
        return degraded("unknown", f"Could not read the serving patent corpus ({type(exc).__name__}).")
    if not tables:
        return degraded("no_serving_corpus", "The patent corpus tables do not exist (migration 0004 not applied); "
                                             "codes come from definition matching.")
    if corpus is None:
        return degraded("no_serving_corpus", "No validated patent corpus is serving; codes come from definition "
                                             "matching.")
    out["patent_corpus"] = {"corpus_id": str(corpus.corpus_id), "source": corpus.source, "document_count": count,
                            "labels_ipc_edition": corpus.label_ipc_edition, "state": corpus.state}
    if enc is None:
        why = rt.patent_encoder_error or f"models are {rt.state}"
        return degraded("patent_encoder_unavailable", f"The patent encoder (PaECTER) is not loaded ({why}); codes "
                                                      "come from definition matching.")
    if (enc.model_id, enc.revision) != (corpus.encoder_model_id, corpus.encoder_model_revision):
        return degraded(
            "patent_encoder_unavailable",
            f"Patent corpus {corpus.corpus_id!r} was embedded with {corpus.encoder_model_id}@"
            f"{corpus.encoder_model_revision}, but the loaded encoder is {enc.model_id}@{enc.revision}; the vote is "
            "disabled until they match and codes come from definition matching.")
    if count is not None and count < s.neighbour_k:
        return degraded(
            "corpus_too_small",
            f"Patent corpus {corpus.corpus_id!r} has {count} documents, fewer than NEIGHBOUR_K={s.neighbour_k}: every "
            "vote would return fewer than k neighbours (vote_failed); codes come from definition matching.")
    out.update(state="serving", degraded=False, reason=None)
    return out


@router.get("/ready")
def ready() -> JSONResponse:
    rt = get_runtime()
    checks: dict[str, Any] = {"models": rt.state, "models_error": rt.error}
    try:
        corpus = _corpus_state()
    except Exception as exc:  # noqa: BLE001
        corpus = {"database": "unreachable", "error": type(exc).__name__}
    checks["corpus"] = corpus
    ok = (
        rt.state == "ready"
        and corpus.get("schema") == "ok"
        and corpus.get("active_release") is not None
        and corpus.get("serving_embedding_run") is not None
    )
    # Calibration readiness is reported separately: an uncalibrated pipeline can still serve honestly.
    checks["calibration"] = rt.confidence.status() if rt.confidence else {"state": "not_calibrated"}
    checks["llm"] = {"enabled": get_settings().llm_enabled}
    # Reported separately too, and never part of `ok`: without the vote the definition fallback still serves.
    checks["neighbour_vote"] = neighbour_vote_state(rt)
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})


@router.get("/v1/taxonomy/status")
def taxonomy_status() -> dict[str, Any]:
    rt = get_runtime()
    try:
        corpus = _corpus_state()
    except Exception as exc:  # noqa: BLE001
        corpus = {"database": "unreachable", "error": type(exc).__name__}
    s = get_settings()
    return {
        "corpus": corpus,
        "models": {
            "state": rt.state,
            "embedding_provider": s.embedding_provider,
            "embedding_model": s.embedding_model_id if s.embedding_provider == "bge_m3" else s.gemma_embedding_model,
            "embedding_revision": s.embedding_model_revision if s.embedding_provider == "bge_m3" else None,
            "reranker_model": s.reranker_model_id,
            "reranker_revision": s.reranker_model_revision,
        },
        "calibration": rt.confidence.status() if rt.confidence else {"state": "not_calibrated"},
        "keyword_mode": "PostgreSQL full-text search (ts_rank_cd); not BM25",
        "source_warnings": corpus.get("validation", {}).get("source_quality", []),
        "neighbour_vote": neighbour_vote_state(rt),
    }
