"""Resumable taxonomy embedding runs.

A run is keyed by (release, provider/model, revision, input hash, storage table). Resuming picks up
records whose stored vector is missing, belongs to another run, or has a stale semantic_text hash.
A run becomes 'complete' only when every record of the release has a vector from this run with a
matching hash; 'validated' after dimension/norm/count checks; serving is a separate activation.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from app.services.embeddings.providers import EmbeddingProvider

log = logging.getLogger(__name__)

ALLOWED_TABLES = {"ipc_embeddings": 1024, "ipc_embeddings_gemma": 768}


class EmbeddingRunError(RuntimeError):
    pass


def release_input_hash(conn: Connection, version: str) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT code, semantic_text_sha256 FROM ipc_records WHERE taxonomy_version = %s ORDER BY code",
            [version],
        )
        for code, sha in cur:
            h.update(f"{code}:{sha}\n".encode())
            n += 1
    return h.hexdigest(), n


def _find_or_create_run(conn: Connection, version: str, provider: EmbeddingProvider) -> str:
    p = provider.policy
    if p.storage_table not in ALLOWED_TABLES or ALLOWED_TABLES[p.storage_table] != p.dimension:
        raise EmbeddingRunError(f"storage table {p.storage_table} does not hold {p.dimension}-dim vectors")
    input_hash, expected = release_input_hash(conn, version)
    if expected == 0:
        raise EmbeddingRunError(f"release {version} has no records")
    row = conn.execute(
        "SELECT run_id FROM embedding_runs WHERE taxonomy_version = %s AND storage_table = %s"
        " AND model_id = %s AND model_revision = %s AND input_hash = %s"
        " AND state IN ('pending','running','failed','complete','validated')"
        " ORDER BY created_at DESC LIMIT 1",
        [version, p.storage_table, p.model_id, p.model_revision, input_hash],
    ).fetchone()
    if row:
        return row[0]
    run_id = f"emb-{p.provider}-{uuid.uuid4().hex[:12]}"
    # Any other run in this table for this release is superseded; it stops serving immediately.
    conn.execute(
        "UPDATE embedding_runs SET state = 'superseded', is_serving = false, updated_at = now()"
        " WHERE taxonomy_version = %s AND storage_table = %s AND state <> 'superseded'",
        [version, p.storage_table],
    )
    conn.execute(
        "INSERT INTO embedding_runs (run_id, taxonomy_version, provider, model_id, model_revision, dimension,"
        " normalized, storage_table, encoding_policy, input_hash, expected_count, state)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending')",
        [run_id, version, p.provider, p.model_id, p.model_revision, p.dimension, p.normalized,
         p.storage_table, Jsonb(p.as_dict()), input_hash, expected],
    )  # fmt: skip
    conn.commit()
    return run_id


def _pending(conn: Connection, version: str, table: str, run_id: str, limit: int) -> list[tuple]:
    q = sql.SQL(
        "SELECT r.code, r.record_type, r.subclass_code, r.semantic_text, r.semantic_text_sha256"
        " FROM ipc_records r LEFT JOIN {t} e ON e.taxonomy_version = r.taxonomy_version AND e.code = r.code"
        " WHERE r.taxonomy_version = %s AND (e.code IS NULL OR e.embedding_run_id <> %s"
        "   OR e.semantic_text_sha256 <> r.semantic_text_sha256)"
        " ORDER BY r.code LIMIT %s"
    ).format(t=sql.Identifier(table))
    return conn.execute(q, [version, run_id, limit]).fetchall()


def _upsert(conn: Connection, table: str, version: str, run_id: str, rows: list[tuple], vecs: np.ndarray) -> None:
    t = sql.Identifier(table)
    stage = sql.Identifier(f"stage_{table}")
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE TEMP TABLE IF NOT EXISTS {s} (LIKE {t} INCLUDING DEFAULTS) ON COMMIT DELETE ROWS")
            .format(s=stage, t=t)
        )
        with cur.copy(
            sql.SQL("COPY {s} (taxonomy_version, code, record_type, subclass_code, embedding,"
                    " embedding_run_id, semantic_text_sha256) FROM STDIN WITH (FORMAT BINARY)").format(s=stage)
        ) as copy:
            copy.set_types(["text", "text", "text", "text", "vector", "text", "text"])
            for (code, rtype, subclass, _text, sha), v in zip(rows, vecs, strict=True):
                copy.write_row((version, code, rtype, subclass, v, run_id, sha))
        cur.execute(
            sql.SQL(
                "INSERT INTO {t} (taxonomy_version, code, record_type, subclass_code, embedding,"
                " embedding_run_id, semantic_text_sha256)"
                " SELECT taxonomy_version, code, record_type, subclass_code, embedding, embedding_run_id,"
                " semantic_text_sha256 FROM {s}"
                " ON CONFLICT (taxonomy_version, code) DO UPDATE SET embedding = EXCLUDED.embedding,"
                " embedding_run_id = EXCLUDED.embedding_run_id, record_type = EXCLUDED.record_type,"
                " subclass_code = EXCLUDED.subclass_code,"
                " semantic_text_sha256 = EXCLUDED.semantic_text_sha256, created_at = now()"
            ).format(t=t, s=stage)
        )


def run_embeddings(
    conn: Connection,
    version: str,
    provider: EmbeddingProvider,
    *,
    batch_records: int = 512,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Embed every record of a release (resumable). max_records bounds a partial/smoke run."""
    register_vector(conn)
    table = provider.policy.storage_table
    run_id = _find_or_create_run(conn, version, provider)
    conn.execute("UPDATE embedding_runs SET state = 'running', updated_at = now() WHERE run_id = %s", [run_id])
    conn.commit()
    done_now = 0
    try:
        while max_records is None or done_now < max_records:
            limit = batch_records if max_records is None else min(batch_records, max_records - done_now)
            rows = _pending(conn, version, table, run_id, limit)
            if not rows:
                break
            vecs = provider.encode_documents([r[3] for r in rows])
            _upsert(conn, table, version, run_id, rows, vecs)
            done_now += len(rows)
            completed = conn.execute(
                sql.SQL("SELECT count(*) FROM {t} WHERE taxonomy_version = %s AND embedding_run_id = %s")
                .format(t=sql.Identifier(table)), [version, run_id]
            ).fetchone()[0]
            conn.execute(
                "UPDATE embedding_runs SET completed_count = %s, updated_at = now() WHERE run_id = %s",
                [completed, run_id],
            )
            conn.commit()
            log.info("embedded %d (run total %d)", done_now, completed)
    except Exception as exc:
        conn.rollback()
        conn.execute(
            "UPDATE embedding_runs SET state = 'failed', errors = errors || %s, updated_at = now()"
            " WHERE run_id = %s",
            [Jsonb([{"at": datetime.now(UTC).isoformat(), "error": str(exc)[:2000]}]), run_id],
        )
        conn.commit()
        raise
    return finalize_run(conn, run_id)


def finalize_run(conn: Connection, run_id: str) -> dict[str, Any]:
    """Mark complete+validated only if every record has a fresh vector from this run."""
    run = conn.execute(
        "SELECT taxonomy_version, storage_table, expected_count, dimension FROM embedding_runs WHERE run_id = %s",
        [run_id],
    ).fetchone()
    version, table, expected, dim = run
    t = sql.Identifier(table)
    fresh = conn.execute(
        sql.SQL(
            "SELECT count(*) FROM {t} e JOIN ipc_records r USING (taxonomy_version, code)"
            " WHERE e.taxonomy_version = %s AND e.embedding_run_id = %s"
            " AND e.semantic_text_sha256 = r.semantic_text_sha256 AND vector_dims(e.embedding) = %s"
        ).format(t=t), [version, run_id, dim]
    ).fetchone()[0]
    bad_norm = conn.execute(
        sql.SQL("SELECT count(*) FROM {t} WHERE taxonomy_version = %s AND embedding_run_id = %s"
                " AND NOT (vector_norm(embedding) > 0.99 AND vector_norm(embedding) < 1.01)").format(t=t),
        [version, run_id],
    ).fetchone()[0]
    complete = fresh == expected and bad_norm == 0
    state = "validated" if complete else "running"
    conn.execute(
        "UPDATE embedding_runs SET completed_count = %s, state = %s, updated_at = now(),"
        " completed_at = CASE WHEN %s THEN now() ELSE completed_at END WHERE run_id = %s",
        [fresh, state, complete, run_id],
    )
    conn.commit()
    return {"run_id": run_id, "storage_table": table, "expected": expected, "fresh_vectors": fresh,
            "bad_norm": bad_norm, "state": state}


def activate_run(conn: Connection, run_id: str) -> None:
    """Serve a validated run. Requires the HNSW index for its table to exist."""
    run = conn.execute(
        "SELECT taxonomy_version, storage_table, state FROM embedding_runs WHERE run_id = %s", [run_id]
    ).fetchone()
    if run is None:
        raise EmbeddingRunError(f"run {run_id} not found")
    version, table, state = run
    if state != "validated":
        raise EmbeddingRunError(f"run {run_id} is {state!r}; only a validated run can serve")
    has_index = conn.execute(
        "SELECT count(*) FROM pg_indexes WHERE tablename = %s AND indexdef ILIKE '%%USING hnsw%%'", [table]
    ).fetchone()[0]
    if not has_index:
        raise EmbeddingRunError(f"no HNSW index on {table}; run `ipc build-index` first")
    conn.execute(
        "UPDATE embedding_runs SET is_serving = false WHERE taxonomy_version = %s AND storage_table = %s",
        [version, table],
    )
    conn.execute("UPDATE embedding_runs SET is_serving = true, updated_at = now() WHERE run_id = %s", [run_id])
    conn.commit()
