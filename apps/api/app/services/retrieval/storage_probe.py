"""Measure real vector-table and HNSW index storage with synthetic unit vectors.

Synthetic random vectors have the same on-disk size as real embeddings, so this measures
capacity requirements before a full ingestion. They are written only to a scratch table
that is dropped afterwards and are never used for retrieval.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection, sql


def probe(conn: Connection, *, rows: int, dim: int, kind: str, m: int, ef_construction: int,
          seed: int = 7) -> dict[str, Any]:
    if kind not in ("vector", "halfvec"):
        raise ValueError(kind)
    register_vector(conn)
    table = sql.Identifier(f"storage_probe_{kind}_{dim}")
    rng = np.random.default_rng(seed)
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
    conn.execute(sql.SQL("CREATE TABLE {} (id integer PRIMARY KEY, embedding {}({}))").format(
        table, sql.SQL(kind), sql.Literal(dim)))
    with conn.cursor() as cur, cur.copy(
        sql.SQL("COPY {} (id, embedding) FROM STDIN WITH (FORMAT BINARY)").format(table)
    ) as copy:
        copy.set_types(["int4", kind])
        for start in range(0, rows, 5000):
            n = min(5000, rows - start)
            v = rng.standard_normal((n, dim)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            for i in range(n):
                if kind == "halfvec":
                    from pgvector import HalfVector
                    copy.write_row((start + i, HalfVector(v[i])))
                else:
                    copy.write_row((start + i, v[i]))
    conn.commit()
    ops = "halfvec_cosine_ops" if kind == "halfvec" else "vector_cosine_ops"
    conn.execute("SET maintenance_work_mem = '1GB'")
    t0 = time.time()
    conn.execute(sql.SQL("CREATE INDEX ON {} USING hnsw (embedding {}) WITH (m = {}, ef_construction = {})")
                 .format(table, sql.SQL(ops), sql.Literal(m), sql.Literal(ef_construction)))
    conn.commit()
    build_s = time.time() - t0
    total, indexes = conn.execute(
        sql.SQL("SELECT pg_total_relation_size({t}::regclass), pg_indexes_size({t}::regclass)").format(
            t=sql.Literal(table.as_string(conn))
        )
    ).fetchone()
    conn.execute(sql.SQL("DROP TABLE {}").format(table))
    conn.commit()
    return {"kind": kind, "dim": dim, "rows": rows, "m": m, "ef_construction": ef_construction,
            "table_plus_toast_bytes": total - indexes, "indexes_bytes": indexes,
            "total_bytes": total, "total_mib": round(total / 2**20, 1), "hnsw_build_s": round(build_s, 1)}
