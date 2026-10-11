"""pgvector cosine retrieval over the serving embedding table.

HNSW is the index; the embedding model creates the vectors. ef_search and iterative scan are set
with transaction-local set_config so request settings never leak through pooled connections.
An exact (sequential) search path exists for recall evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.retrieval.keyword import GROUP_TYPES

TABLES = {"ipc_embeddings", "ipc_embeddings_gemma"}


@dataclass
class VectorHit:
    code: str
    rank: int
    cosine_distance: float

    @property
    def cosine_similarity(self) -> float:
        return 1.0 - self.cosine_distance


def to_vector_literal(vec: np.ndarray) -> str:
    """Text form bound as a parameter and cast to vector in SQL (no string-concatenated SQL)."""
    return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"


def vector_search(
    session: Session,
    *,
    table: str,
    version: str,
    query_vector: np.ndarray,
    limit: int,
    ef_search: int,
    record_types: tuple[str, ...] = GROUP_TYPES,
    section: str | None = None,
    subclass: str | None = None,
    subclasses: list[str] | None = None,
    exact: bool = False,
) -> list[VectorHit]:
    if table not in TABLES:
        raise ValueError(f"unknown embedding table {table!r}")
    dim = 1024 if table == "ipc_embeddings" else 768
    ef = max(int(ef_search), int(limit))
    # Every vector row is joined to its official scheme entry and filtered to current K entries.
    filtered = True
    # Each request runs in its own transaction; set_config(..., true) is transaction-local.
    session.execute(text("SELECT set_config('hnsw.ef_search', :ef, true)"), {"ef": str(ef)})
    if filtered:
        session.execute(text("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"))
    if exact:
        session.execute(text("SELECT set_config('enable_indexscan', 'off', true)"))
        session.execute(text("SELECT set_config('enable_bitmapscan', 'off', true)"))
    order_cte = (
        f"SELECT e.code, e.embedding <=> CAST(:qv AS vector({dim})) AS cosine_distance"
        f" FROM {table} e JOIN ipc_records r"
        "   ON r.taxonomy_version = e.taxonomy_version AND r.code = e.code"
        " JOIN ipc_official_scheme_entries official"
        "   ON official.taxonomy_version = e.taxonomy_version AND official.code = e.code"
        " WHERE e.taxonomy_version = :version AND e.record_type = ANY(:types)"
        "   AND official.entry_type = 'K'"
        "   AND (CAST(:section AS text) IS NULL OR r.section_code = :section)"
        "   AND (CAST(:subclass AS text) IS NULL OR r.subclass_code = :subclass)"
        "   AND (CAST(:subclasses AS text[]) IS NULL OR r.subclass_code = ANY(:subclasses))"
        f" ORDER BY e.embedding <=> CAST(:qv AS vector({dim}))"
        " LIMIT :limit"
    )
    # relaxed_order may return slightly out-of-order rows; re-sort by the computed distance.
    sql = f"SELECT code, cosine_distance FROM ({order_cte}) s ORDER BY cosine_distance, code"
    rows = session.execute(
        text(sql),
        {"qv": to_vector_literal(query_vector), "version": version, "types": list(record_types),
         "section": section, "subclass": subclass, "subclasses": subclasses, "limit": limit},
    ).all()
    return [VectorHit(code=r[0], rank=i + 1, cosine_distance=float(r[1])) for i, r in enumerate(rows)]


def similarities_for_codes(
    session: Session, *, table: str, version: str, query_vector: np.ndarray, codes: list[str]
) -> dict[str, float]:
    """Exact cosine similarity for a bounded explicit code list (used for expansion)."""
    if table not in TABLES:
        raise ValueError(f"unknown embedding table {table!r}")
    if not codes:
        return {}
    dim = 1024 if table == "ipc_embeddings" else 768
    rows = session.execute(
        text(
            f"SELECT code, 1 - (embedding <=> CAST(:qv AS vector({dim}))) FROM {table}"
            " WHERE taxonomy_version = :version AND code = ANY(:codes)"
        ),
        {"qv": to_vector_literal(query_vector), "version": version, "codes": codes},
    ).all()
    return {r[0]: float(r[1]) for r in rows}
