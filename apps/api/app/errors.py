"""Structured API errors: {"error": code, "message": text, "details": ...}."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def _body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": code, "message": message, "details": details}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=_body(exc.code, exc.message, exc.details))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errs = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        missing_abstract = any(e["loc"][-1:] == ["abstract"] for e in errs)
        code = "invalid_abstract" if missing_abstract else "invalid_request"
        msg = "abstract is required and must not be blank." if missing_abstract else "Request validation failed."
        return JSONResponse(status_code=422, content=_body(code, msg, errs))

    @app.exception_handler(OperationalError)
    async def _db(_: Request, exc: OperationalError) -> JSONResponse:
        return JSONResponse(status_code=503, content=_body("database_unavailable", "Database is unreachable."))
