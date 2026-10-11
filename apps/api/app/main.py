"""FastAPI entry point. Models load once per process in a background thread at startup, so /health is
immediately live and /ready reports loading state. Run: uv run uvicorn app.main:app --port 8000"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import errors
from app.config import get_settings
from app.routes import admin, classify, system, taxonomy
from app.runtime import get_runtime

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("IPC_SKIP_MODEL_LOAD") != "1":
        get_runtime().start_loading()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="IPC Classifier API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > s.max_request_bytes:
            return JSONResponse(status_code=413, content={
                "error": "request_too_large",
                "message": f"Request body exceeds {s.max_request_bytes} bytes.", "details": None})
        return await call_next(request)

    errors.install(app)
    app.include_router(system.router)
    app.include_router(classify.router)
    app.include_router(taxonomy.router)
    app.include_router(admin.router)
    return app


app = create_app()
