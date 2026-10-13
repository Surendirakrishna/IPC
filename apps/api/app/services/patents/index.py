"""Per-corpus HNSW index on patent_embeddings: build, recall validation, activation and retirement (ADR-0005).

One partial index per corpus (user decision, 2026-10-13):

    CREATE INDEX patent_hnsw_<slug>_<hash> ON patent_embeddings
        USING hnsw (embedding halfvec_cosine_ops) WITH (m = ..., ef_construction = ...)
        WHERE corpus_id = '<corpus id literal>'

A table-wide index would filter by corpus after the graph scan and return fewer than k rows while a second
corpus is loaded next to the serving one. With a partial index each corpus has its own graph, so a new
corpus can be embedded, indexed and validated beside the serving one. ``patent_search.search_patents``
inlines the corpus id as a literal so the planner can prove the query implies the index predicate.

Lifecycle: ``build_patent_index`` (corpus embedded) -> ``validate_corpus`` (recall@k of the HNSW search
against exact search, document count == embedding count -> state ``validated``) -> ``activate_corpus``
(validated *and* its index exists -> serving; at most one corpus serves) -> ``retire_corpus`` (stops
serving if asked, drops the index, back to ``embedded``).

All functions take a psycopg ``Connection`` that is idle (no open transaction) and manage their own
transactions; values are bound or composed with ``psycopg.sql`` (Identifier / Literal), never formatted in.
Corpus ids used here must match ``CORPUS_ID_RE``, so the index predicate compares exactly with the catalogue's
rendering of it.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from psycopg import Connection, sql
from psycopg.pq import TransactionStatus

from app.db import patents
from app.db.patents import Corpus, CorpusNotFound, CorpusStateError
from app.services.retrieval.patent_search import EF_SEARCH_MAX, NeighbourShortfall, search_patents, search_query

INDEX_PREFIX = "patent_hnsw_"
CORPUS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
BUILDABLE_STATES = ("embedded", "validated")  # 'loading' vectors are still changing; 'failed' needs a re-run
DEFAULT_RECALL_THRESHOLD = 0.95
DEFAULT_PROBES = 200
DEFAULT_K = 100
TIE_EPSILON = 1e-6  # an HNSW hit at the k-th exact distance (a tie) counts as found
MAX_NAME = 63  # PostgreSQL identifier limit
# Above this many vectors the planner must choose the HNSW index *unforced* (as a vote request runs the search);
# below it a primary-key scan plus a sort is legitimately cheaper (gotcha: tiny test tables).
DEFAULT_PLANNER_GATE_ROWS = 10_000
# DROP INDEX waits at most this long for its lock, then fails cleanly instead of queueing vote searches behind it.
DEFAULT_DROP_LOCK_TIMEOUT = "5s"


class PatentIndexError(RuntimeError):
    pass


def _check_corpus_id(corpus_id: str) -> str:
    if not isinstance(corpus_id, str) or not CORPUS_ID_RE.match(corpus_id):
        raise ValueError(f"corpus id {corpus_id!r} must match {CORPUS_ID_RE.pattern} to get a partial index")
    return corpus_id


def index_name(corpus_id: str) -> str:
    """Deterministic index name for a corpus: readable slug plus a hash of the exact id (slugs can collide)."""
    _check_corpus_id(corpus_id)
    digest = hashlib.md5(corpus_id.encode(), usedforsecurity=False).hexdigest()[:8]
    slug = re.sub(r"[^a-z0-9]+", "_", corpus_id.lower()).strip("_")
    slug = slug[: MAX_NAME - len(INDEX_PREFIX) - len(digest) - 1]
    return f"{INDEX_PREFIX}{slug}_{digest}"


def index_predicate(corpus_id: str) -> str:
    """The index predicate as PostgreSQL renders it (``pg_get_expr(indpred, indrelid)``)."""
    return f"(corpus_id = '{_check_corpus_id(corpus_id)}'::text)"


def _require_idle(conn: Connection) -> None:
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise PatentIndexError("the connection has an open transaction; commit or roll back before index work")


@contextmanager
def _autocommit(conn: Connection) -> Iterator[Connection]:
    _require_idle(conn)
    previous = conn.autocommit
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.autocommit = previous


def _corpus(conn: Connection, corpus_id: str) -> Corpus:
    corpus = patents.get_corpus(conn, corpus_id)
    conn.rollback()
    if corpus is None:
        raise CorpusNotFound(f"corpus {corpus_id!r} not found")
    return corpus


def corpus_index(conn: Connection, corpus_id: str) -> dict[str, Any] | None:
    """The corpus's own index, if it exists and is what the search needs; else None.

    Checks the name, the access method and operator class, the predicate, and that the index is valid (a
    failed concurrent build leaves an invalid index behind)."""
    name = index_name(corpus_id)
    row = conn.execute(
        "SELECT am.amname, pg_get_indexdef(i.indexrelid), pg_get_expr(i.indpred, i.indrelid), i.indisvalid,"
        " pg_relation_size(i.indexrelid)"
        " FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_am am ON am.oid = c.relam"
        " WHERE c.relname = %s AND i.indrelid = 'patent_embeddings'::regclass",
        [name],
    ).fetchone()
    conn.rollback()
    if row is None:
        return None
    amname, indexdef, predicate, valid, size = row
    problems = []
    if amname != "hnsw" or "halfvec_cosine_ops" not in indexdef:
        problems.append(f"not an HNSW halfvec_cosine_ops index: {indexdef}")
    if predicate != index_predicate(corpus_id):
        problems.append(f"predicate {predicate!r} is not {index_predicate(corpus_id)!r}")
    if not valid:
        problems.append("index is marked invalid (an interrupted build)")
    return {"index": name, "bytes": int(size), "definition": indexdef, "valid": not problems, "problems": problems}


def drop_index_statement(name: str) -> sql.Composed:
    """``DROP INDEX CONCURRENTLY IF EXISTS``: a plain DROP INDEX takes an ACCESS EXCLUSIVE lock on all of
    patent_embeddings, queues behind in-flight classify transactions and then blocks every new vote search.
    CONCURRENTLY only waits for transactions using the index and never blocks readers; it needs autocommit."""
    return sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(sql.Identifier(name))


def _drop_index(conn: Connection, name: str, lock_timeout: str) -> None:
    """Drop on an autocommit connection, waiting at most ``lock_timeout`` (then psycopg raises LockNotAvailable /
    QueryCanceled and nothing is dropped)."""
    if not conn.autocommit:
        raise PatentIndexError("DROP INDEX CONCURRENTLY needs an autocommit connection")
    conn.execute(sql.SQL("SET lock_timeout = {}").format(sql.Literal(lock_timeout)))
    try:
        conn.execute(drop_index_statement(name))
    finally:
        conn.execute("RESET lock_timeout")


def _database_size(conn: Connection) -> int:
    size = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
    return int(size)


def _counts(conn: Connection, corpus_id: str) -> tuple[int, int]:
    docs = conn.execute("SELECT count(*) FROM patent_documents WHERE corpus_id = %s", [corpus_id]).fetchone()[0]
    embs = conn.execute("SELECT count(*) FROM patent_embeddings WHERE corpus_id = %s", [corpus_id]).fetchone()[0]
    return int(docs), int(embs)


def _server_versions(conn: Connection) -> dict[str, str | None]:
    pg = conn.execute("SHOW server_version").fetchone()[0]
    ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    return {"postgresql": pg, "pgvector": ext[0] if ext else None}


def build_patent_index(
    conn: Connection,
    corpus_id: str,
    *,
    m: int,
    ef_construction: int,
    maintenance_work_mem: str | None = None,
    concurrently: bool = False,
    replace: bool = False,
    lock_timeout: str = "60s",
    drop_lock_timeout: str = DEFAULT_DROP_LOCK_TIMEOUT,
) -> dict[str, Any]:
    """Build the corpus's partial HNSW index and return the build manifest.

    The corpus must be ``embedded`` (or ``validated``, to rebuild) and have vectors, and must not be serving: a
    rebuild drops the index every vote search depends on. An existing valid index is kept (``status:
    "exists"``) unless ``replace``; an invalid one is dropped and rebuilt. A validated corpus whose index is
    (re)built goes back to ``embedded`` *before* anything is dropped, so the new index must be validated again
    and the corpus cannot be activated in between. Old indexes are dropped with ``DROP INDEX CONCURRENTLY``
    (waiting at most ``drop_lock_timeout``). ``maintenance_work_mem`` is set for this session only (sakthi's
    server default is 512 MB; ipc_user may SET it). ``concurrently`` avoids blocking writes to other corpora
    during the build, at the cost of a slower build. The manifest records build seconds, index size and
    ``pg_database_size`` before and after.
    """
    _check_corpus_id(corpus_id)
    if not 2 <= int(m) <= 100:
        raise ValueError("m must be in [2, 100] (pgvector)")
    if not 4 <= int(ef_construction) <= 1000 or int(ef_construction) < 2 * int(m):
        raise ValueError("ef_construction must be in [4, 1000] and >= 2 * m (pgvector)")
    _require_idle(conn)
    corpus = _corpus(conn, corpus_id)
    if corpus.is_serving:
        raise CorpusStateError(f"corpus {corpus_id!r} is serving; its index cannot be built or replaced while vote "
                               "searches use it (activate another corpus or retire it with stop_serving=True)")
    if corpus.state not in BUILDABLE_STATES:
        raise CorpusStateError(f"corpus {corpus_id!r} is {corpus.state!r}; build its index once it is embedded")
    name = index_name(corpus_id)
    manifest: dict[str, Any] = {
        "corpus_id": corpus_id, "index": name, "table": "patent_embeddings", "method": "hnsw",
        "opclass": "halfvec_cosine_ops", "m": int(m), "ef_construction": int(ef_construction),
        "maintenance_work_mem": maintenance_work_mem, "concurrently": concurrently, "state_before": corpus.state,
    }  # fmt: skip
    existing = corpus_index(conn, corpus_id)
    if existing is not None and existing["valid"] and not replace:
        return {**manifest, "status": "exists", "bytes": existing["bytes"], "definition": existing["definition"],
                "state": corpus.state}  # fmt: skip
    if corpus.state == "validated":
        # set_corpus_state refuses a serving corpus, so this also closes the race with a concurrent activation.
        patents.set_corpus_state(conn, corpus_id, "embedded")
    with _autocommit(conn):
        manifest["server"] = _server_versions(conn)
        manifest["document_count"], manifest["embedding_count"] = _counts(conn, corpus_id)
        if manifest["embedding_count"] == 0:
            raise CorpusStateError(f"corpus {corpus_id!r} has no embeddings to index")
        manifest["database_bytes_before"] = _database_size(conn)
        if existing is not None:
            manifest["replaced"] = existing["definition"]
            _drop_index(conn, name, drop_lock_timeout)
        try:
            conn.execute(sql.SQL("SET lock_timeout = {}").format(sql.Literal(lock_timeout)))
            conn.execute("SET statement_timeout = 0")  # a large build takes far longer than any request timeout
            if maintenance_work_mem:
                conn.execute(sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(maintenance_work_mem)))
            manifest["maintenance_work_mem_effective"] = conn.execute("SHOW maintenance_work_mem").fetchone()[0]
            t0 = time.perf_counter()
            conn.execute(sql.SQL(
                "CREATE INDEX {conc} {name} ON patent_embeddings USING hnsw (embedding halfvec_cosine_ops)"
                " WITH (m = {m}, ef_construction = {ef}) WHERE corpus_id = {corpus}"
            ).format(conc=sql.SQL("CONCURRENTLY") if concurrently else sql.SQL(""), name=sql.Identifier(name),
                     m=sql.Literal(int(m)), ef=sql.Literal(int(ef_construction)), corpus=sql.Literal(corpus_id)))
            manifest["build_seconds"] = round(time.perf_counter() - t0, 2)
            # Planner statistics: per-corpus row counts decide whether the HNSW path is chosen.
            conn.execute("ANALYZE patent_embeddings")
        finally:
            for setting in ("lock_timeout", "statement_timeout", "maintenance_work_mem"):
                conn.execute(sql.SQL("RESET {}").format(sql.SQL(setting)))  # fixed names above
        built = corpus_index(conn, corpus_id)
        if built is None or not built["valid"]:
            raise PatentIndexError(f"index {name} is missing or invalid after the build: {built}")
        manifest["bytes"] = built["bytes"]
        manifest["definition"] = built["definition"]
        manifest["database_bytes_after"] = _database_size(conn)
    manifest["status"] = "created"
    manifest["state"] = "embedded"  # a freshly built index is unmeasured: validate_corpus decides
    return manifest


def _vector(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=np.float32)


def _plan(conn: Connection, corpus_id: str, vector_text: str, k: int) -> str:
    rows = conn.execute(sql.SQL("EXPLAIN ") + search_query(corpus_id, k), {"qv": vector_text}).fetchall()
    return "\n".join(r[0] for r in rows)


def _force_hnsw(conn: Connection) -> None:
    """Make the HNSW scan the only cheap path (transaction-local). On a small corpus the planner otherwise
    prefers the primary key plus a sort, and validation would measure exact search instead of the index."""
    # One statement: on the LAN to sakthi every round trip costs tens of milliseconds, times 200 probes.
    conn.execute("SELECT set_config('enable_seqscan', 'off', true), set_config('enable_bitmapscan', 'off', true),"
                 " set_config('enable_sort', 'off', true)")


def _is_hit(n: Any, exact_ids: set[str], kth: float) -> bool:
    """An HNSW result counts if exact search has it, or it ties the k-th exact distance."""
    return n.document_id in exact_ids or (1.0 - n.similarity) <= kth + TIE_EPSILON


def _exact_mode(conn: Connection) -> None:
    """Exact reference search (transaction-local): no index scans, so never the HNSW graph (pgvector's HNSW
    has no bitmap scan). The result is a full scan of the corpus's rows, sorted by distance."""
    conn.execute("SELECT set_config('enable_indexscan', 'off', true)")


def validate_corpus(
    conn: Connection,
    corpus_id: str,
    *,
    probes: int = DEFAULT_PROBES,
    k: int = DEFAULT_K,
    ef_search: int | None = None,
    threshold: float = DEFAULT_RECALL_THRESHOLD,
    seed: int = 20261013,
    planner_gate_rows: int = DEFAULT_PLANNER_GATE_ROWS,
) -> dict[str, Any]:
    """Measure recall@k of the corpus's HNSW search against exact search, and mark it ``validated`` if it passes.

    Probes are ``probes`` corpus vectors chosen deterministically from ``seed``. For each, the real search
    (``patent_search.search_patents`` with ``ef_search``, default PATENT_HNSW_EF_SEARCH) is compared with an
    exact scan (index scans off) of the same corpus. An HNSW hit at the k-th exact distance counts (ties).
    Each probe is itself a corpus vector at distance 0, which any search finds; ``recall_at_k`` counts only the
    other neighbours (the probe's own document is left out on both sides) and is what the threshold gates.
    ``recall_at_k_with_self`` is recorded too.

    The recall is measured on a forced HNSW plan (a small corpus would otherwise be scanned exactly). Above
    ``planner_gate_rows`` embeddings the planner must also choose the index *unforced*, as a vote request runs
    the search; both plan checks are recorded.

    Passes when: the corpus's index exists and is valid, the planner uses it, every probe returned k rows,
    mean recall@k >= ``threshold``, the corpus has at least k documents, and document count == embedding count.
    Then the state becomes ``validated`` with the measured ``document_count`` (provenance reads it). A failure
    leaves the state unchanged (rebuild with larger m / ef_construction, or fix the load) and lists reasons.
    """
    _check_corpus_id(corpus_id)
    if probes < 1:
        raise ValueError("probes must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if ef_search is None:
        from app.config import get_settings

        ef_search = get_settings().patent_hnsw_ef_search
    if not k <= ef_search <= EF_SEARCH_MAX:
        raise ValueError(f"ef_search must be in [k, {EF_SEARCH_MAX}], got {ef_search} for k={k}")
    _require_idle(conn)
    corpus = _corpus(conn, corpus_id)
    if corpus.state not in BUILDABLE_STATES:
        raise CorpusStateError(f"corpus {corpus_id!r} is {corpus.state!r}; only an embedded corpus can be validated")
    t_start = time.perf_counter()
    reasons: list[str] = []
    index = corpus_index(conn, corpus_id)
    doc_count, emb_count = _counts(conn, corpus_id)
    conn.rollback()
    manifest: dict[str, Any] = {
        "corpus_id": corpus_id, "index": index_name(corpus_id), "probes_requested": probes, "k": k,
        "ef_search": ef_search, "threshold": threshold, "seed": seed, "document_count": doc_count,
        "embedding_count": emb_count, "state_before": corpus.state,
    }  # fmt: skip
    if index is None:
        reasons.append(f"no index {index_name(corpus_id)} for this corpus (run build_patent_index)")
    elif not index["valid"]:
        reasons.append("index unusable: " + "; ".join(index["problems"]))
    if doc_count != emb_count:
        reasons.append(f"document count {doc_count} != embedding count {emb_count}")
    if doc_count < k:
        reasons.append(f"corpus has {doc_count} documents, fewer than k={k}: every vote would fall short")

    recalls: list[float] = []  # self-hit excluded (gated)
    recalls_with_self: list[float] = []
    latencies: list[float] = []
    shortfalls = 0
    if index is not None and index["valid"] and emb_count >= k:
        rows = conn.execute(
            "SELECT document_id, embedding::text FROM patent_embeddings WHERE corpus_id = %s"
            " ORDER BY md5(document_id || %s), document_id LIMIT %s",
            [corpus_id, str(seed), int(probes)],
        ).fetchall()
        conn.rollback()
        manifest["probes"] = len(rows)
        # Every read below runs in the connection's own (implicit) transaction and is rolled back, so the
        # transaction-local planner settings never outlive one statement group.
        name = index_name(corpus_id)
        if rows:
            probe = rows[0][1]
            unforced = _plan(conn, corpus_id, probe, k)
            conn.rollback()
            _force_hnsw(conn)
            plan = _plan(conn, corpus_id, probe, k)
            conn.rollback()
            _exact_mode(conn)
            exact_plan = _plan(conn, corpus_id, probe, k)
            conn.rollback()
            manifest["planner_default_uses_index"] = f"using {name}" in unforced
            manifest["plan_uses_index"] = f"using {name}" in plan
            manifest["exact_plan_avoids_index"] = name not in exact_plan
            manifest["planner_gate_rows"] = planner_gate_rows
            manifest["planner_gate_applied"] = emb_count > planner_gate_rows
            if manifest["planner_gate_applied"] and not manifest["planner_default_uses_index"]:
                reasons.append(f"with {emb_count} vectors (> {planner_gate_rows}) the planner does not choose the "
                               "corpus index for the unforced search statement, so vote searches would scan:\n"
                               + unforced)  # fmt: skip
            if not manifest["plan_uses_index"]:
                reasons.append("the planner cannot use the corpus index for the search statement:\n" + plan)
            if not manifest["exact_plan_avoids_index"]:
                reasons.append("the exact reference search still used the HNSW index:\n" + exact_plan)
        measurable = bool(rows) and manifest["plan_uses_index"] and manifest["exact_plan_avoids_index"]
        for probe_id, vector_text in rows if measurable else []:
            vec = _vector(vector_text)
            _exact_mode(conn)
            exact = conn.execute(
                sql.SQL("SELECT document_id, embedding <=> CAST(%(qv)s AS halfvec(1024)) AS d"
                        " FROM patent_embeddings WHERE corpus_id = {corpus}"
                        " ORDER BY d, document_id LIMIT {k}").format(corpus=sql.Literal(corpus_id), k=sql.Literal(int(k))),
                {"qv": vector_text},
            ).fetchall()
            conn.rollback()
            exact_ids = {r[0] for r in exact}
            kth = max(float(r[1]) for r in exact)
            try:
                _force_hnsw(conn)
                t = time.perf_counter()  # the search itself (set ef_search + query), as a vote request runs it
                try:
                    found = search_patents(conn, corpus_id, vec, k, ef_search)
                except NeighbourShortfall as exc:
                    shortfalls += 1
                    found = exc.neighbours
                latencies.append((time.perf_counter() - t) * 1000)
            finally:
                conn.rollback()
            hits = [_is_hit(n, exact_ids, kth) for n in found]
            recalls_with_self.append(min(1.0, sum(hits) / k))
            # Without the probe's own document (distance 0, found by any search): the k-1 real neighbours.
            wanted = len(exact_ids - {probe_id}) or 1
            others = sum(h for n, h in zip(found, hits, strict=True) if n.document_id != probe_id)
            recalls.append(min(1.0, others / wanted))

    if recalls:
        manifest["recall_at_k"] = round(statistics.fmean(recalls), 4)
        manifest["recall_at_k_with_self"] = round(statistics.fmean(recalls_with_self), 4)
        manifest["recall_at_k_min"] = round(min(recalls), 4)
        manifest["probes_below_threshold"] = sum(1 for r in recalls if r < threshold)
        manifest["hnsw_ms_p50"] = round(statistics.median(latencies), 2)
        manifest["hnsw_ms_p95"] = round(sorted(latencies)[max(0, int(round(0.95 * len(latencies))) - 1)], 2)
        if manifest["recall_at_k"] < threshold:
            reasons.append(f"recall@{k} {manifest['recall_at_k']} < threshold {threshold}")
    elif not reasons:
        reasons.append("no probe vectors")
    manifest["shortfalls"] = shortfalls
    if shortfalls:
        reasons.append(f"{shortfalls} probe(s) returned fewer than k={k} rows")
    manifest["seconds"] = round(time.perf_counter() - t_start, 2)
    manifest["passed"] = not reasons
    manifest["reasons"] = reasons
    if manifest["passed"]:
        updated = patents.set_corpus_state(conn, corpus_id, "validated", document_count=doc_count)
        manifest["state"] = updated.state
    else:
        manifest["state"] = corpus.state
    return manifest


def activate_corpus(conn: Connection, corpus_id: str) -> Corpus:
    """Serve this corpus (and stop serving any other). Refused unless it is ``validated`` and its own index
    exists and is valid."""
    _require_idle(conn)
    corpus = _corpus(conn, corpus_id)
    if corpus.state != "validated":
        raise CorpusStateError(f"corpus {corpus_id!r} is {corpus.state!r}; validate it before activating")
    index = corpus_index(conn, corpus_id)
    if index is None or not index["valid"]:
        why = "does not exist" if index is None else "; ".join(index["problems"])
        raise CorpusStateError(f"corpus {corpus_id!r} has no usable index {index_name(corpus_id)} ({why})")
    return patents.set_serving_corpus(conn, corpus_id)


def retire_corpus(conn: Connection, corpus_id: str, *, stop_serving: bool = False,
                  drop_lock_timeout: str = DEFAULT_DROP_LOCK_TIMEOUT) -> dict[str, Any]:  # fmt: skip
    """Drop the corpus's index and set it back to ``embedded`` (it must be re-indexed and re-validated to serve).

    A serving corpus is refused unless ``stop_serving``: then serving stops first and the classifier falls back
    to definition matching until another corpus is activated. The state goes to ``embedded`` before the drop (no
    activation in between), and the drop is ``DROP INDEX CONCURRENTLY`` waiting at most ``drop_lock_timeout``.
    """
    _check_corpus_id(corpus_id)
    _require_idle(conn)
    corpus = _corpus(conn, corpus_id)
    out: dict[str, Any] = {"corpus_id": corpus_id, "index": index_name(corpus_id), "was_serving": corpus.is_serving,
                           "state_before": corpus.state}  # fmt: skip
    if corpus.is_serving:
        if not stop_serving:
            raise CorpusStateError(f"corpus {corpus_id!r} is serving; pass stop_serving=True to retire it")
        patents.unset_serving_corpus(conn)
    if corpus.state == "validated":
        out["state"] = patents.set_corpus_state(conn, corpus_id, "embedded").state
    else:
        out["state"] = corpus.state
    with _autocommit(conn):
        out["dropped"] = corpus_index(conn, corpus_id) is not None
        _drop_index(conn, index_name(corpus_id), drop_lock_timeout)
    return out
