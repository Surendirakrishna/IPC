"""HNSW index management, kept separate from row import. Built after embeddings are loaded."""

from __future__ import annotations

import time
from typing import Any

from psycopg import Connection, sql

TABLES = {"ipc_embeddings", "ipc_embeddings_gemma"}


def index_name(table: str) -> str:
    return f"{table}_hnsw_cosine"


def build_hnsw(conn: Connection, table: str, *, m: int, ef_construction: int, maintenance_work_mem: str | None,
               concurrently: bool = False) -> dict[str, Any]:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    name = index_name(table)
    conn.autocommit = True
    try:
        if maintenance_work_mem:
            conn.execute(sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(maintenance_work_mem)))
        exists = conn.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", [name]).fetchone()
        if exists:
            return {"index": name, "status": "exists"}
        t0 = time.time()
        conn.execute(
            sql.SQL("CREATE INDEX {conc} {name} ON {table} USING hnsw (embedding vector_cosine_ops)"
                    " WITH (m = {m}, ef_construction = {ef})").format(
                conc=sql.SQL("CONCURRENTLY") if concurrently else sql.SQL(""),
                name=sql.Identifier(name), table=sql.Identifier(table),
                m=sql.Literal(int(m)), ef=sql.Literal(int(ef_construction)))
        )
        size = conn.execute("SELECT pg_relation_size(%s::regclass)", [name]).fetchone()[0]
        return {"index": name, "status": "created", "seconds": round(time.time() - t0, 1), "bytes": size,
                "m": m, "ef_construction": ef_construction}
    finally:
        conn.autocommit = False


def drop_hnsw(conn: Connection, table: str) -> None:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index_name(table))))
    conn.commit()


def explain_vector_query(conn: Connection, table: str, version: str, dim: int, ef_search: int) -> str:
    """Representative plan for review (uses a stored vector as the probe)."""
    probe = conn.execute(sql.SQL("SELECT embedding::text FROM {} WHERE taxonomy_version = %s LIMIT 1")
                         .format(sql.Identifier(table)), [version]).fetchone()
    if not probe:
        return "no vectors"
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(ef_search)])
        rows = conn.execute(
            sql.SQL("EXPLAIN (ANALYZE, BUFFERS) SELECT code, embedding <=> %s::vector({d}) AS d FROM {t}"
                    " WHERE taxonomy_version = %s AND record_type = ANY(%s)"
                    " ORDER BY embedding <=> %s::vector({d}) LIMIT 100").format(
                d=sql.Literal(dim), t=sql.Identifier(table)),
            [probe[0], version, ["maingroup", "subgroup"], probe[0]],
        ).fetchall()
    return "\n".join(r[0] for r in rows)
