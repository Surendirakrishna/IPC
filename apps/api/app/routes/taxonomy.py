"""GET /v1/ipc/search and GET /v1/ipc/{code}."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.db import repository as repo
from app.db.session import api_session
from app.errors import ApiError
from app.services.classification.scope import parent_path

router = APIRouter(prefix="/v1/ipc", tags=["taxonomy"])


def _version(session, version: str | None) -> str:
    rel = repo.active_release(session)
    if version:
        if repo.get_release(session, version) is None:
            raise ApiError(422, "invalid_taxonomy_version", f"Unknown taxonomy_version {version!r}.")
        return version
    if rel is None:
        raise ApiError(503, "corpus_unavailable", "No active taxonomy release.")
    return rel.taxonomy_version


@router.get("/search")
def search(
    q: str | None = Query(default=None, max_length=500),
    record_type: str | None = Query(default=None, pattern="^(section|class|subclass|maingroup|subgroup)$"),
    section: str | None = Query(default=None, pattern="^[A-H]$"),
    subclass: str | None = Query(default=None, pattern=r"^[A-H]\d{2}[A-Z]$"),
    taxonomy_version: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100_000),
) -> dict[str, Any]:
    with api_session() as session:
        v = _version(session, taxonomy_version)
        items, total = repo.search_records(session, v, q=q, record_type=record_type, section=section,
                                           subclass=subclass, limit=limit, offset=offset)
    return {"taxonomy_version": v, "total": total, "limit": limit, "offset": offset, "items": items,
            "ranking": "exact/prefix code lookup, else PostgreSQL full-text ts_rank_cd (not BM25)"}


@router.get("/{code:path}")
def get_code(code: str, taxonomy_version: str | None = None) -> dict[str, Any]:
    with api_session() as session:
        v = _version(session, taxonomy_version)
        rec = repo.get_record_by_any_code(session, v, code)
        if rec is None:
            raise ApiError(404, "code_not_found", f"{code!r} is not in taxonomy {v}.")
        parents = repo.get_records(session, v, rec.ancestor_codes)
        children = repo.get_records(session, v, rec.child_codes[:500])
        refs = repo.get_references(session, v, [rec.code]).get(rec.code, [])
        incoming = repo.get_incoming_references(session, v, rec.code)
        occurrences = repo.get_occurrences(session, v, rec.code)
    return {
        "taxonomy_version": v,
        "code": rec.code,
        "display_code": rec.display_code,
        "record_type": rec.record_type,
        "title": rec.title,
        "title_raw": rec.title_raw,
        "original_title": rec.original_title,
        "original_title_source": rec.original_title_source,
        "semantic_text": rec.semantic_text,
        "parent_code": rec.parent_code,
        "parent_path": parent_path(rec, parents),
        "children": [{"code": c, "display_code": children[c].display_code, "title": children[c].title}
                     for c in rec.child_codes if c in children],
        "duplicate_context": {
            "is_duplicate_symbol": rec.is_duplicate_symbol,
            "titles": rec.titles,
            "canonical_title": rec.canonical_title,
            "context_heading": rec.context_heading,
        },
        "source_occurrences": occurrences,
        "references": refs,
        "referenced_by": incoming,
    }
