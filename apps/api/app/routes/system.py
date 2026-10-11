"""GET /health (liveness, no model/DB work), GET /ready, GET /v1/taxonomy/status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.db import repository as repo
from app.db.session import api_session, database_configured
from app.runtime import get_runtime

router = APIRouter(tags=["system"])


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
    }
