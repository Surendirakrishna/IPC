"""Protected administration. Import/embedding/index/calibration jobs are CLI-only (`ipc --help`);
the API exposes read-only job status and a calibrator reload. Disabled unless ADMIN_TOKEN is set."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Header
from sqlalchemy import text

from app.config import get_settings
from app.db.session import api_session
from app.errors import ApiError
from app.runtime import get_runtime

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _require_admin(authorization: str | None) -> None:
    token = get_settings().admin_token
    if token is None:
        raise ApiError(404, "admin_disabled", "Administration API is disabled (ADMIN_TOKEN unset).")
    expected = f"Bearer {token.get_secret_value()}"
    if not authorization or not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise ApiError(401, "unauthorized", "Valid admin bearer token required.")


@router.get("/jobs")
def jobs(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_admin(authorization)
    with api_session() as session:
        rows = session.execute(
            text("SELECT job_id, kind, taxonomy_version, state, progress, errors, started_at, updated_at,"
                 " finished_at FROM ingestion_jobs ORDER BY started_at DESC LIMIT 50")
        ).mappings().all()
        runs = session.execute(
            text("SELECT run_id, provider, model_id, model_revision, storage_table, state, is_serving,"
                 " expected_count, completed_count, created_at, completed_at FROM embedding_runs"
                 " ORDER BY created_at DESC LIMIT 20")
        ).mappings().all()
    return {"ingestion_jobs": [dict(r) for r in rows], "embedding_runs": [dict(r) for r in runs]}


@router.post("/reload-confidence")
def reload_confidence(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_admin(authorization)
    rt = get_runtime()
    if rt.state != "ready":
        raise ApiError(503, "models_unavailable", "Models are not loaded.")
    rt.refresh_confidence()
    return {"calibration": rt.confidence.status(), "fingerprint_hash": rt.fingerprint_hash}
