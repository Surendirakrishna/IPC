"""TASK-0007: per-corpus partial HNSW index on patent_embeddings, recall validation, activation, retirement.

Unit tests need no database. Integration tests write to sakthi only with an explicit opt-in (there is no
local docker database on this machine):

    IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1 uv run pytest -m integration tests/test_patent_index.py

(or IPC_DB_TARGET=local against a local database). They touch only the patent tables and only corpus ids
starting with "it-test-"; every index they build is that corpus's own ``patent_hnsw_it_test_*`` index, dropped
in teardown together with the corpus rows. They refuse to run when a real (non-it-test) corpus is serving,
loading or embedded. Activating an it-test corpus makes it the serving corpus for the length of the test.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from psycopg.pq import TransactionStatus

from app.db.patents import Corpus, CorpusStateError
from app.services.patents import index as ix
from app.services.retrieval.patent_search import NeighbourShortfall

# --- units -------------------------------------------------------------------------------------------------


def test_index_name_is_deterministic_bounded_and_collision_free() -> None:
    name = ix.index_name("hupd-2017")
    assert name == ix.index_name("hupd-2017")
    assert name.startswith("patent_hnsw_hupd_2017_") and len(name) <= 63
    assert ix.index_name("a-b") != ix.index_name("a.b")  # same slug, different ids
    long = ix.index_name("x" * 100)
    assert len(long) <= 63 and long.startswith("patent_hnsw_x")
    assert ix.index_name("it-test-abc").startswith("patent_hnsw_it_test_abc_")


@pytest.mark.parametrize("bad", ["", "it's", "a b", "-lead", "x" * 101, "a;drop"])
def test_corpus_ids_that_cannot_be_a_clean_literal_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        ix.index_name(bad)


def test_index_predicate_matches_postgres_rendering() -> None:
    assert ix.index_predicate("hupd-2017") == "(corpus_id = 'hupd-2017'::text)"


@pytest.mark.parametrize(("m", "efc"), [(1, 64), (101, 400), (16, 3), (16, 1001), (32, 63)])
def test_build_rejects_bad_hnsw_parameters_before_touching_the_database(m: int, efc: int) -> None:
    with pytest.raises(ValueError):
        ix.build_patent_index(object(), "it-test-x", m=m, ef_construction=efc)  # type: ignore[arg-type]


@pytest.mark.parametrize(("kwargs"), [{"probes": 0}, {"k": 0}, {"threshold": 0.0}, {"threshold": 1.5},
                                      {"k": 100, "ef_search": 99}, {"ef_search": 1001}])  # fmt: skip
def test_validate_rejects_bad_arguments_before_touching_the_database(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("ef_search", 200)
    with pytest.raises(ValueError):
        ix.validate_corpus(object(), "it-test-x", **kwargs)  # type: ignore[arg-type]


class _IdleConn:
    info = SimpleNamespace(transaction_status=TransactionStatus.IDLE)
    autocommit = False

    def rollback(self) -> None:
        pass


def _corpus(state: str, serving: bool = False) -> Corpus:
    now = datetime(2026, 10, 13, tzinfo=UTC)
    return Corpus("it-test-u", "hupd", None, "compact", None, None, 0, "m", "r", None, state, serving, now, now, None)


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch):
    state: dict[str, Any] = {"corpus": _corpus("validated"), "index": {"valid": True, "problems": [], "bytes": 1},
                             "served": [], "unset": 0}  # fmt: skip
    monkeypatch.setattr(ix.patents, "get_corpus", lambda conn, cid: state["corpus"])
    monkeypatch.setattr(ix, "corpus_index", lambda conn, cid: state["index"])

    def serve(conn, cid):  # noqa: ANN001
        state["served"].append(cid)
        return replace(state["corpus"], is_serving=True)

    monkeypatch.setattr(ix.patents, "set_serving_corpus", serve)
    monkeypatch.setattr(ix.patents, "unset_serving_corpus", lambda conn: state.__setitem__("unset", 1))
    return state


def test_activate_requires_validated_and_its_own_index(fake_db) -> None:
    conn = _IdleConn()
    fake_db["corpus"] = _corpus("embedded")
    with pytest.raises(CorpusStateError, match="validate it before activating"):
        ix.activate_corpus(conn, "it-test-u")  # type: ignore[arg-type]
    fake_db["corpus"] = _corpus("validated")
    fake_db["index"] = None
    with pytest.raises(CorpusStateError, match="no usable index .*does not exist"):
        ix.activate_corpus(conn, "it-test-u")  # type: ignore[arg-type]
    fake_db["index"] = {"valid": False, "problems": ["index is marked invalid"], "bytes": 1}
    with pytest.raises(CorpusStateError, match="marked invalid"):
        ix.activate_corpus(conn, "it-test-u")  # type: ignore[arg-type]
    assert fake_db["served"] == []
    fake_db["index"] = {"valid": True, "problems": [], "bytes": 1}
    assert ix.activate_corpus(conn, "it-test-u").is_serving  # type: ignore[arg-type]
    assert fake_db["served"] == ["it-test-u"]


def test_retire_refuses_a_serving_corpus_unless_asked(fake_db) -> None:
    fake_db["corpus"] = _corpus("validated", serving=True)
    with pytest.raises(CorpusStateError, match="stop_serving=True"):
        ix.retire_corpus(_IdleConn(), "it-test-u")  # type: ignore[arg-type]
    assert fake_db["unset"] == 0


def test_index_work_refuses_an_open_transaction() -> None:
    conn = SimpleNamespace(info=SimpleNamespace(transaction_status=TransactionStatus.INTRANS))
    with pytest.raises(ix.PatentIndexError, match="open transaction"):
        ix.activate_corpus(conn, "it-test-u")  # type: ignore[arg-type]


# --- integration: sakthi with an explicit opt-in (or a local database) ---------------------------------------

TEST_PREFIX = "it-test-"
BUSY_STATES = ("loading", "embedded")
PATENT_TABLES = ("patent_corpora", "patent_documents", "patent_embeddings")


def _integration_target() -> str | None:
    target = os.environ.get("IPC_DB_TARGET")
    if target == "local":
        return target
    if target == "sakthi" and os.environ.get("IPC_INTEGRATION_ALLOW_SAKTHI") == "1":
        return target
    return None


def _refusal(corpora: list[tuple[str, str, bool]]) -> str | None:
    """Why the integration tests must not run against this database, or None.

    They build indexes on, ANALYZE and serve from the shared patent tables, which would disturb a live load or
    take over serving. So they refuse when any real (non-it-test) corpus is serving, loading or embedded.
    """
    for corpus_id, state, is_serving in corpora:
        if corpus_id.startswith(TEST_PREFIX):
            continue
        if is_serving:
            return f"refusing to run: corpus {corpus_id!r} is serving"
        if state in BUSY_STATES:
            return f"refusing to run: corpus {corpus_id!r} is {state!r} (a load may be running)"
    return None


def test_integration_refuses_when_a_real_corpus_is_serving_or_busy() -> None:
    assert _refusal([]) is None
    assert _refusal([("it-test-a", "loading", False), ("it-test-b", "validated", True)]) is None
    assert _refusal([("hupd-2017", "validated", False), ("old", "failed", False)]) is None
    assert "serving" in _refusal([("hupd-2017", "validated", True)])  # type: ignore[operator]
    assert "'loading'" in _refusal([("uspto-2021", "loading", False)])  # type: ignore[operator]
    assert "'embedded'" in _refusal([("uspto-2021", "embedded", False)])  # type: ignore[operator]


def _table_counts(conn) -> dict[str, int]:  # noqa: ANN001
    out = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in PATENT_TABLES}  # fixed names
    out["patent_hnsw_indexes"] = conn.execute(
        "SELECT count(*) FROM pg_indexes WHERE tablename = 'patent_embeddings' AND indexname LIKE 'patent_hnsw_%'"
    ).fetchone()[0]
    conn.rollback()
    return out


@pytest.fixture
def db_conn():
    """(conn, created_corpora). Teardown stops serving / drops the index / deletes rows of it-test-* corpora only."""
    target = _integration_target()
    if target is None:
        pytest.skip("set IPC_DB_TARGET=local (or IPC_DB_TARGET=sakthi with IPC_INTEGRATION_ALLOW_SAKTHI=1) "
                    "to run DB integration tests")
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().ipc_db_target == target
    from app.db.session import connect_direct

    conn = connect_direct()
    rows = conn.execute("SELECT corpus_id, state, is_serving FROM patent_corpora").fetchall()
    conn.commit()
    reason = _refusal(rows)
    if reason is not None:
        conn.close()
        pytest.fail(reason)
    before = _table_counts(conn)
    created: list[str] = []
    try:
        yield conn, created
    finally:
        from psycopg import sql

        assert all(c.startswith(TEST_PREFIX) for c in created), created
        conn.rollback()
        conn.autocommit = True
        for corpus_id in created:
            conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(ix.index_name(corpus_id))))
        conn.execute("UPDATE patent_corpora SET is_serving = false WHERE corpus_id = ANY(%s) AND corpus_id LIKE %s",
                     [created, TEST_PREFIX + "%"])
        conn.execute("DELETE FROM patent_corpora WHERE corpus_id = ANY(%s) AND corpus_id LIKE %s",
                     [created, TEST_PREFIX + "%"])
        conn.autocommit = False
        after = _table_counts(conn)
        left = conn.execute(
            "SELECT (SELECT count(*) FROM patent_corpora WHERE corpus_id = ANY(%(c)s))"
            " + (SELECT count(*) FROM patent_documents WHERE corpus_id = ANY(%(c)s))"
            " + (SELECT count(*) FROM patent_embeddings WHERE corpus_id = ANY(%(c)s))",
            {"c": created},
        ).fetchone()[0]
        conn.rollback()
        conn.close()
        # Other sessions may write to the shared tables meanwhile, so the exact check is on this test's rows.
        print(f"\npatent table counts before={before} after={after}")
        assert left == 0, f"{left} rows of {created} left behind"


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _load(conn, created: list[str], n: int, rng: np.random.Generator, *, embedded: bool = True,  # noqa: ANN001
          extra_docs: int = 0) -> str:  # fmt: skip
    from app.db import patents
    from app.db.patents import CorpusSpec, PatentDocument

    corpus_id = f"{TEST_PREFIX}{uuid.uuid4().hex[:10]}"
    created.append(corpus_id)
    patents.create_or_get_corpus(conn, CorpusSpec(corpus_id, "hupd", "compact", "mpi-inno-comp/paecter", "rev-test"))
    ids = [f"d{i:04d}" for i in range(n + extra_docs)]
    docs = [PatentDocument(d, f"Title {d}", "A synthetic abstract for the index test.", date(2017, 1, 1), None,
                           ["H04L908"], [("H04L", "H04L 9/00", "H04L 9/08")]) for d in ids]  # fmt: skip
    patents.upsert_documents(conn, corpus_id, docs, batch_size=500)
    patents.insert_embeddings(conn, corpus_id, [(d, _unit(rng)) for d in ids[:n]], batch_size=500)
    if embedded:
        patents.set_corpus_state(conn, corpus_id, "embedded")
    return corpus_id


@pytest.mark.integration
def test_build_validate_activate_beside_a_coexisting_corpus(db_conn) -> None:
    from app.db import patents

    conn, created = db_conn
    rng = np.random.default_rng(20261013)
    a = _load(conn, created, 300, rng)
    b = _load(conn, created, 200, rng)  # coexists: its rows must never leak into a's k results

    built = ix.build_patent_index(conn, a, m=16, ef_construction=64, maintenance_work_mem="64MB")
    print("\nbuild manifest:", built)
    assert built["status"] == "created"
    assert "USING hnsw (embedding halfvec_cosine_ops)" in built["definition"]
    assert f"WHERE (corpus_id = '{a}'::text)" in built["definition"]
    assert built["build_seconds"] >= 0 and built["bytes"] > 0
    assert built["database_bytes_before"] > 0 and built["database_bytes_after"] > 0
    assert built["maintenance_work_mem_effective"] == "64MB"
    assert built["embedding_count"] == built["document_count"] == 300
    assert ix.build_patent_index(conn, a, m=16, ef_construction=64)["status"] == "exists"  # idempotent
    assert ix.build_patent_index(conn, b, m=16, ef_construction=64)["status"] == "created"
    assert conn.execute("SHOW maintenance_work_mem").fetchone()[0] != "64MB"  # session setting was reset
    conn.rollback()

    with pytest.raises(CorpusStateError, match="validate it before activating"):
        ix.activate_corpus(conn, a)  # embedded, indexed, but not validated

    report = ix.validate_corpus(conn, a, probes=200, k=100, ef_search=200)
    print("validation manifest:", report)
    assert report["passed"], report["reasons"]
    assert report["probes"] == 200 and report["shortfalls"] == 0
    assert report["recall_at_k"] >= 0.95 and report["plan_uses_index"] and report["exact_plan_avoids_index"]
    assert report["state"] == "validated"
    corpus = patents.get_corpus(conn, a)
    conn.rollback()
    assert corpus.state == "validated" and corpus.document_count == 300  # provenance reads the real count

    served = ix.activate_corpus(conn, a)
    assert served.is_serving
    with pytest.raises(CorpusStateError):
        ix.activate_corpus(conn, b)  # b is still only embedded
    assert ix.validate_corpus(conn, b, probes=50, k=100, ef_search=200)["passed"]
    ix.activate_corpus(conn, b)
    serving = conn.execute("SELECT corpus_id FROM patent_corpora WHERE is_serving").fetchall()
    conn.rollback()
    assert serving == [(b,)]  # one serving corpus at most

    retired = ix.retire_corpus(conn, b, stop_serving=True)
    assert retired == {"corpus_id": b, "index": ix.index_name(b), "was_serving": True, "state_before": "validated",
                       "dropped": True, "state": "embedded"}  # fmt: skip
    assert ix.corpus_index(conn, b) is None
    with pytest.raises(CorpusStateError):
        ix.activate_corpus(conn, b)  # needs re-indexing and re-validation
    assert ix.corpus_index(conn, a)["valid"]  # retiring b left a's index alone


@pytest.mark.integration
def test_refusals_and_failed_validation(db_conn, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.db import patents

    conn, created = db_conn
    rng = np.random.default_rng(7)
    loading = _load(conn, created, 120, rng, embedded=False)
    with pytest.raises(CorpusStateError, match="once it is embedded"):
        ix.build_patent_index(conn, loading, m=16, ef_construction=64)

    # validated without an index (state set by hand): activation still refuses
    no_index = _load(conn, created, 120, rng)
    patents.set_corpus_state(conn, no_index, "validated")
    with pytest.raises(CorpusStateError, match="no usable index"):
        ix.activate_corpus(conn, no_index)

    # a document without a vector: document count != embedding count -> not validated
    partial = _load(conn, created, 150, rng, extra_docs=1)
    ix.build_patent_index(conn, partial, m=16, ef_construction=64)
    report = ix.validate_corpus(conn, partial, probes=20, k=100, ef_search=200)
    assert not report["passed"] and report["state"] == "embedded"
    assert "document count 151 != embedding count 150" in report["reasons"]

    # recall below the threshold, and short searches: not validated, state unchanged
    good = _load(conn, created, 150, rng)
    ix.build_patent_index(conn, good, m=16, ef_construction=64)
    real_search = ix.search_patents

    def half_wrong(conn_, corpus_id, vec, k, ef):  # noqa: ANN001
        found = real_search(conn_, corpus_id, vec, k, ef)
        return found[: k // 2] + [replace(n, document_id=f"not-{n.document_id}", similarity=-1.0) for n in found[k // 2:]]

    monkeypatch.setattr(ix, "search_patents", half_wrong)
    report = ix.validate_corpus(conn, good, probes=10, k=100, ef_search=200)
    assert not report["passed"] and report["recall_at_k"] == 0.5 and report["state"] == "embedded"
    assert any(r.startswith("recall@100 0.5 < threshold 0.95") for r in report["reasons"])

    def short(conn_, corpus_id, vec, k, ef):  # noqa: ANN001
        raise NeighbourShortfall(corpus_id, k, real_search(conn_, corpus_id, vec, k, ef)[:90])

    monkeypatch.setattr(ix, "search_patents", short)
    report = ix.validate_corpus(conn, good, probes=5, k=100, ef_search=200)
    assert not report["passed"] and report["shortfalls"] == 5
    assert "5 probe(s) returned fewer than k=100 rows" in report["reasons"]
    corpus = patents.get_corpus(conn, good)
    conn.rollback()
    assert corpus.state == "embedded" and not corpus.is_serving
