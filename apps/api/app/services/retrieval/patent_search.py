"""Nearest labelled patents of one corpus, by PaECTER cosine similarity (ADR-0005).

Each corpus has its own partial HNSW index (``... USING hnsw (embedding halfvec_cosine_ops) WHERE
corpus_id = '<literal>'``, built by TASK-0007). The planner only uses a partial index when it can prove
the query's WHERE clause implies the index predicate, and it cannot prove that for a bound parameter
in a generic plan (psycopg prepares a statement after ``prepare_threshold`` executions, so the pooled
API connections would switch to generic plans). The corpus id and the limit are therefore inlined as
SQL literals with ``psycopg.sql.Literal`` (quoted by the driver, never by string formatting); the
query vector stays a bound parameter.

``hnsw.ef_search`` is set with transaction-local ``set_config`` so the value never leaks through a
pooled connection. An HNSW scan returns at most ``ef_search`` rows, so fewer than ``k`` rows means the
index or its settings are wrong (or the corpus is smaller than ``k``). That is raised as
``NeighbourShortfall`` instead of being returned as a smaller vote.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from psycopg import Connection, sql
from sqlalchemy.orm import Session

from app.services.patents.labels import ResolvedLabel

EMBEDDING_DIM = 1024
EF_SEARCH_MAX = 1000  # pgvector's upper bound for hnsw.ef_search


class PatentSearchError(RuntimeError):
    pass


class NeighbourShortfall(PatentSearchError):
    """The search returned fewer than ``k`` neighbours. ``neighbours`` holds what was found."""

    def __init__(self, corpus_id: str, k: int, neighbours: list[PatentNeighbour]) -> None:
        self.corpus_id = corpus_id
        self.k = k
        self.neighbours = neighbours
        super().__init__(
            f"patent search on corpus {corpus_id!r} returned {len(neighbours)} of k={k} neighbours; "
            "check that the corpus has its partial HNSW index, that hnsw.ef_search >= k, and that it holds "
            "at least k embedded documents"
        )


@dataclass(frozen=True)
class PatentNeighbour:
    """One corpus patent near the query. ``labels`` are resolved catalogue display codes, main label first."""

    document_id: str
    title: str | None
    similarity: float  # cosine similarity, 1 - cosine distance
    labels: tuple[ResolvedLabel, ...]


def _driver(conn: Connection | Session) -> Connection:
    if isinstance(conn, Session):
        return conn.connection().connection.driver_connection  # type: ignore[return-value]
    return conn


def halfvec_text(query_vector: Any) -> str:
    """Text form of a 1024-dim query vector, bound as a parameter and cast to halfvec(1024) in SQL."""
    v = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if v.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"query vector has {v.shape[0]} dimensions, expected {EMBEDDING_DIM}")
    if not np.isfinite(v).all():
        raise ValueError("query vector contains non-finite values")
    return "[" + ",".join(f"{float(x):.8g}" for x in v) + "]"


def search_query(corpus_id: str, k: int) -> sql.Composed:
    """The search statement for one corpus. Its single placeholder is the query vector (used twice by name)."""
    corpus = sql.Literal(corpus_id)
    return sql.SQL(
        "SELECT d.document_id, d.title, d.resolved_labels, 1 - n.distance AS similarity"
        " FROM (SELECT e.document_id, e.embedding <=> CAST(%(qv)s AS halfvec(1024)) AS distance"
        "       FROM patent_embeddings e"
        "       WHERE e.corpus_id = {corpus}"
        "       ORDER BY e.embedding <=> CAST(%(qv)s AS halfvec(1024))"
        "       LIMIT {k}) n"
        " JOIN patent_documents d ON d.corpus_id = {corpus} AND d.document_id = n.document_id"
        " ORDER BY n.distance, d.document_id"
    ).format(corpus=corpus, k=sql.Literal(int(k)))


def _labels(raw: Sequence[Sequence[str | None]]) -> tuple[ResolvedLabel, ...]:
    return tuple(ResolvedLabel(lab[0], lab[1], lab[2]) for lab in raw)  # type: ignore[arg-type]


def search_patents(
    session: Connection | Session,
    corpus_id: str,
    query_vector: Any,
    k: int,
    ef_search: int,
) -> list[PatentNeighbour]:
    """The ``k`` nearest documents of ``corpus_id`` to ``query_vector``, by descending similarity.

    ``session`` is a SQLAlchemy ``Session`` (the request path) or a psycopg ``Connection``. The search
    runs in the caller's transaction (an autocommit connection gets its own), and ``hnsw.ef_search`` is
    set for that transaction only. Raises ``NeighbourShortfall`` when fewer than ``k`` rows come back.
    """
    if not corpus_id:
        raise ValueError("corpus_id is required")
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 1 <= ef_search <= EF_SEARCH_MAX:
        raise ValueError(f"ef_search must be in [1, {EF_SEARCH_MAX}], got {ef_search}")
    if ef_search < k:
        raise ValueError(f"ef_search ({ef_search}) must be >= k ({k}): an HNSW scan returns at most ef_search rows")
    params = {"qv": halfvec_text(query_vector)}
    conn = _driver(session)

    def run() -> list[tuple[Any, ...]]:
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(int(ef_search))])
        return conn.execute(search_query(corpus_id, k), params).fetchall()

    if conn.autocommit:
        with conn.transaction():
            rows = run()
    else:
        rows = run()
    neighbours = [
        PatentNeighbour(document_id=doc_id, title=title, similarity=float(sim), labels=_labels(resolved))
        for doc_id, title, resolved, sim in rows
    ]
    if len(neighbours) < k:
        raise NeighbourShortfall(corpus_id, k, neighbours)
    return neighbours
