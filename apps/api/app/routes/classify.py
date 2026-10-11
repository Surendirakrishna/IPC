"""POST /v1/classify."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from app.config import get_settings
from app.db.session import DatabaseNotConfigured, api_session
from app.errors import ApiError
from app.runtime import get_runtime
from app.schemas.classify import ClassifyRequest, ClassifyResponse
from app.services.classification.pipeline import InputRejected, PipelineUnavailable

router = APIRouter(prefix="/v1", tags=["classification"])


def _run(req: ClassifyRequest) -> ClassifyResponse:
    rt = get_runtime()
    with api_session() as session, session.begin():
        result = rt.pipeline.classify(session, req)
    s = get_settings()
    if s.retain_inputs:
        docs = {"abstract": req.abstract} if not s.store_input_documents else {
            k: getattr(req, k) for k in ("abstract", "title", "specification", "claims") if getattr(req, k)}
        with api_session() as session, session.begin():
            session.execute(
                text("INSERT INTO classification_audit (request_id, taxonomy_version, input_documents, response)"
                     " VALUES (:id, :v, CAST(:docs AS jsonb), CAST(:resp AS jsonb))"),
                {"id": str(uuid.uuid4()), "v": result.taxonomy_version, "docs": json.dumps(docs),
                 "resp": result.model_dump_json()},
            )
    return result


@router.post("/classify", response_model=ClassifyResponse, responses={422: {}, 413: {}, 503: {}, 504: {}})
async def classify(req: ClassifyRequest, request: Request) -> ClassifyResponse:
    s = get_settings()
    rt = get_runtime()
    if rt.state != "ready":
        raise ApiError(503, "models_unavailable",
                       f"Models are {rt.state}." + (f" {rt.error}" if rt.error else ""))
    sem = rt.semaphore()
    if sem.locked():
        raise ApiError(503, "busy", "Inference queue is full; retry shortly.")
    async with sem:
        try:
            return await asyncio.wait_for(run_in_threadpool(_run, req), timeout=s.classify_timeout_s)
        except TimeoutError as exc:
            raise ApiError(504, "timeout", f"Classification exceeded {s.classify_timeout_s:.0f}s.") from exc
        except InputRejected as exc:
            status = 413 if exc.code == "request_too_large" else 422
            raise ApiError(status, exc.code, str(exc), exc.details) from exc
        except PipelineUnavailable as exc:
            raise ApiError(503, exc.code, str(exc)) from exc
        except DatabaseNotConfigured as exc:
            raise ApiError(503, "database_unavailable", str(exc)) from exc
