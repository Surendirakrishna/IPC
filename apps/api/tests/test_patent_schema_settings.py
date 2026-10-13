"""TASK-0002: patent corpus settings, migration 0004 and storage helpers.

Unit tests need no database. Integration tests run against the local docker database:
    IPC_DB_TARGET=local uv run pytest -m integration tests/test_patent_schema_settings.py
(after `uv run ipc --target local migrate`). They only touch rows whose corpus id starts with
"it-test-" and delete them afterwards.
"""

from __future__ import annotations

import importlib.util
import os
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings
from app.db import patents
from app.db.patents import CorpusSpec, CorpusStateError, PatentDocument

PATENT_ENV = (
    "NEIGHBOUR_VOTE_ENABLED", "PATENT_ENCODER_MODEL_ID", "PATENT_ENCODER_MODEL_REVISION", "NEIGHBOUR_K",
    "NEIGHBOUR_TEMPERATURE", "NEIGHBOUR_BETA", "NEIGHBOUR_MIN_VOTE_SHARE", "PATENT_HNSW_EF_SEARCH",
    "USPTO_ODP_API_KEY", "HUPD_DATA_DIR",
)
MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "0004_patent_corpus.py"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in PATENT_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# --- settings ----------------------------------------------------------------------------------------


def test_patent_settings_defaults(clean_env: pytest.MonkeyPatch) -> None:
    s = Settings(_env_file=None)
    assert s.neighbour_vote_enabled is False
    assert s.patent_encoder_model_id == "mpi-inno-comp/paecter"
    assert s.patent_encoder_model_revision == "1f355d9e9a34dc4859e22d05ce27e970eeacfbd2"
    assert s.neighbour_k == 100
    assert s.neighbour_temperature == 0.005
    assert s.neighbour_beta == 1.0
    assert s.neighbour_min_vote_share == 0.2
    assert s.patent_hnsw_ef_search == 200
    assert s.uspto_odp_api_key is None
    assert s.hupd_data_dir == Path("D:/ipc_knn/raw/data")


def test_patent_settings_from_environment(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("NEIGHBOUR_VOTE_ENABLED", "true")
    clean_env.setenv("NEIGHBOUR_K", "50")
    clean_env.setenv("NEIGHBOUR_TEMPERATURE", "0.01")
    clean_env.setenv("HUPD_DATA_DIR", "E:/hupd")
    s = Settings(_env_file=None)
    assert s.neighbour_vote_enabled is True and s.neighbour_k == 50 and s.neighbour_temperature == 0.01
    assert s.hupd_data_dir == Path("E:/hupd")


def test_uspto_api_key_is_secret(clean_env: pytest.MonkeyPatch) -> None:
    secret = "odp-test-" + uuid.uuid4().hex
    clean_env.setenv("USPTO_ODP_API_KEY", secret)
    s = Settings(_env_file=None)
    assert isinstance(s.uspto_odp_api_key, SecretStr)
    assert s.uspto_odp_api_key.get_secret_value() == secret
    for rendered in (repr(s), str(s), repr(s.uspto_odp_api_key), str(s.uspto_odp_api_key), s.model_dump_json()):
        assert secret not in rendered
    assert "**********" in str(s.uspto_odp_api_key)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_uspto_api_key_is_unset(clean_env: pytest.MonkeyPatch, blank: str) -> None:
    clean_env.setenv("USPTO_ODP_API_KEY", blank)
    assert Settings(_env_file=None).uspto_odp_api_key is None


@pytest.mark.parametrize(
    ("name", "value"),
    [("NEIGHBOUR_K", "0"), ("NEIGHBOUR_TEMPERATURE", "0"), ("NEIGHBOUR_BETA", "-0.5"),
     ("NEIGHBOUR_MIN_VOTE_SHARE", "1.5"), ("PATENT_HNSW_EF_SEARCH", "0")],
)
def test_patent_settings_reject_out_of_range(clean_env: pytest.MonkeyPatch, name: str, value: str) -> None:
    clean_env.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_ef_search_must_cover_k(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("NEIGHBOUR_K", "300")
    with pytest.raises(ValidationError, match="PATENT_HNSW_EF_SEARCH"):
        Settings(_env_file=None)
    clean_env.setenv("PATENT_HNSW_EF_SEARCH", "300")
    assert Settings(_env_file=None).patent_hnsw_ef_search == 300


# --- migration 0004 (SQL captured, no database) ---------------------------------------------------------


def _migration_sql(func: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    spec = importlib.util.spec_from_file_location("migration_0004", MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executed: list[str] = []
    monkeypatch.setattr(module.op, "execute", lambda sql: executed.append(" ".join(str(sql).split())))
    getattr(module, func)()
    assert module.revision == "0004" and module.down_revision == "0003"
    return executed


def test_migration_upgrade_creates_the_patent_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = "\n".join(_migration_sql("upgrade", monkeypatch))
    for table in ("patent_corpora", "patent_documents", "patent_embeddings"):
        assert f"CREATE TABLE {table}" in sql
    assert "embedding halfvec(1024) NOT NULL" in sql
    assert "CREATE UNIQUE INDEX patent_corpora_one_serving ON patent_corpora (is_serving) WHERE is_serving" in sql
    assert "CHECK (NOT is_serving OR state = 'validated')" in sql
    assert "state IN ('loading', 'embedded', 'validated', 'failed')" in sql
    assert "PRIMARY KEY (corpus_id, document_id)" in sql
    assert "REFERENCES patent_documents (corpus_id, document_id)" in sql
    assert "hnsw" not in sql.lower()  # the HNSW index is built by service code (TASK-0007)
    for existing in ("taxonomy_releases", "ipc_records", "embedding_runs"):  # additive only
        assert existing not in sql


def test_migration_downgrade_drops_in_dependency_order(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _migration_sql("downgrade", monkeypatch) == [
        "DROP TABLE patent_embeddings", "DROP TABLE patent_documents", "DROP TABLE patent_corpora",
    ]


# --- helpers without a database ----------------------------------------------------------------------------


class _Result:
    def __init__(self, row: tuple | None) -> None:
        self._row = row

    def fetchone(self) -> tuple | None:
        return self._row


class _FakeConn:
    """Answers the corpus SELECT with a fixed row and records every statement."""

    def __init__(self, state: str) -> None:
        now = date(2026, 10, 13)
        self.row = ("c1", "hupd", None, "compact", None, None, 0, "m", "r", None, state, False, now, now, None)
        self.statements: list[str] = []
        self.commits = self.rollbacks = 0

    def execute(self, query: str, params: object = None) -> _Result:
        self.statements.append(query)
        return _Result(self.row if query.lstrip().startswith("SELECT") else None)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


@pytest.mark.parametrize("state", ["loading", "embedded", "failed"])
def test_set_serving_refuses_unvalidated_corpus_without_writing(state: str) -> None:
    conn = _FakeConn(state)
    with pytest.raises(CorpusStateError, match="only a validated corpus can serve"):
        patents.set_serving_corpus(conn, "c1")  # type: ignore[arg-type]
    assert not any(s.lstrip().startswith("UPDATE") for s in conn.statements)
    assert conn.commits == 0 and conn.rollbacks == 1


def test_set_serving_accepts_validated_corpus() -> None:
    conn = _FakeConn("validated")
    patents.set_serving_corpus(conn, "c1")  # type: ignore[arg-type]
    updates = [s for s in conn.statements if s.lstrip().startswith("UPDATE")]
    assert len(updates) == 2 and "is_serving = true" in updates[1]
    assert conn.commits == 1


def test_batches_and_vector_checks() -> None:
    assert list(patents._batches(range(5), 2)) == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        list(patents._batches([1], 0))
    assert len(patents._halfvec("d", np.ones(1024, dtype=np.float32)).to_list()) == 1024
    with pytest.raises(ValueError, match="768 dimensions"):
        patents._halfvec("d", np.ones(768))
    bad = np.ones(1024)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        patents._halfvec("d", bad)
    with pytest.raises(ValueError, match="resolved label"):
        patents._labels_json([("H04L", None)])


# --- integration: local docker database ------------------------------------------------------------------


@pytest.fixture
def db_conn():
    if os.environ.get("IPC_DB_TARGET") != "local":
        pytest.skip("set IPC_DB_TARGET=local to run DB integration tests")
    from app.config import get_settings

    get_settings.cache_clear()
    from app.db.session import connect_direct

    conn = connect_direct()
    created: list[str] = []
    yield conn, created
    conn.rollback()
    conn.execute("UPDATE patent_corpora SET is_serving = false WHERE corpus_id = ANY(%s)", [created])
    conn.execute("DELETE FROM patent_corpora WHERE corpus_id = ANY(%s)", [created])
    conn.commit()
    conn.close()


def _spec(created: list[str], **overrides: object) -> CorpusSpec:
    corpus_id = f"it-test-{uuid.uuid4().hex[:10]}"
    created.append(corpus_id)
    fields = {"corpus_id": corpus_id, "source": "hupd", "label_scheme": "compact",
              "encoder_model_id": "mpi-inno-comp/paecter", "encoder_model_revision": "rev-1"}
    fields.update(overrides)
    return CorpusSpec(**fields)  # type: ignore[arg-type]


def _doc(doc_id: str, abstract: str = "A method of distributing quantum keys between grid nodes.") -> PatentDocument:
    return PatentDocument(doc_id, "Title " + doc_id, abstract, date(2017, 1, 5), date(2018, 7, 5),
                          ["H04L1218", "H04L908"], [("H04L", "H04L 12/00", "H04L 12/18"), ("H04L", None, None)])


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.integration
def test_serving_an_unvalidated_corpus_is_refused(db_conn) -> None:
    conn, created = db_conn
    corpus = patents.create_or_get_corpus(conn, _spec(created))
    assert corpus.state == "loading" and not corpus.is_serving
    for state in ("loading", "embedded", "failed"):
        patents.set_corpus_state(conn, corpus.corpus_id, state)  # type: ignore[arg-type]
        with pytest.raises(CorpusStateError):
            patents.set_serving_corpus(conn, corpus.corpus_id)
        assert patents.get_corpus(conn, corpus.corpus_id).is_serving is False  # type: ignore[union-attr]
    # The database refuses it too, even without the helper.
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE patent_corpora SET is_serving = true WHERE corpus_id = %s", [corpus.corpus_id])
    conn.rollback()


@pytest.mark.integration
def test_corpus_round_trip_and_single_serving(db_conn) -> None:
    conn, created = db_conn
    previous = patents.get_serving_corpus(conn)
    spec = _spec(created)
    corpus = patents.create_or_get_corpus(conn, spec)
    assert patents.create_or_get_corpus(conn, spec).created_at == corpus.created_at  # get, not a second row
    with pytest.raises(patents.CorpusConflict):
        patents.create_or_get_corpus(conn, CorpusSpec(**{**spec.__dict__, "encoder_model_revision": "rev-2"}))

    assert patents.upsert_documents(conn, corpus.corpus_id, [_doc("d1"), _doc("d2"), _doc("d3")], batch_size=2) == 3
    assert patents.upsert_documents(conn, corpus.corpus_id, [_doc("d1")]) == 1  # idempotent
    assert patents.insert_embeddings(conn, corpus.corpus_id, [("d1", _unit(1)), ("d2", _unit(2)), ("d3", _unit(3))]) == 3
    got = patents.fetch_documents(conn, corpus.corpus_id, ["d2", "d1", "missing"])
    assert set(got) == {"d1", "d2"}
    assert got["d1"].resolved_labels[0] == ("H04L", "H04L 12/00", "H04L 12/18")  # main label first
    assert got["d1"].raw_labels == ["H04L1218", "H04L908"] and got["d1"].filing_date == date(2017, 1, 5)

    # A changed abstract drops the stale vector; an unchanged one keeps it.
    patents.upsert_documents(conn, corpus.corpus_id, [_doc("d2", "A different abstract."), _doc("d3")])
    have = {r[0] for r in conn.execute(
        "SELECT document_id FROM patent_embeddings WHERE corpus_id = %s", [corpus.corpus_id]).fetchall()}
    assert have == {"d1", "d3"}
    conn.commit()

    patents.set_corpus_state(conn, corpus.corpus_id, "validated", document_count=3, input_hash="h1")
    with pytest.raises(CorpusStateError):  # a validated corpus is read-only
        patents.upsert_documents(conn, corpus.corpus_id, [_doc("d4")])
    served = patents.set_serving_corpus(conn, corpus.corpus_id)
    assert served.is_serving and served.document_count == 3 and served.input_hash == "h1"
    assert patents.get_serving_corpus(conn).corpus_id == corpus.corpus_id  # type: ignore[union-attr]
    with pytest.raises(CorpusStateError):  # a serving corpus must stop serving before it changes state
        patents.set_corpus_state(conn, corpus.corpus_id, "loading")

    other = patents.create_or_get_corpus(conn, _spec(created))
    patents.set_corpus_state(conn, other.corpus_id, "validated")
    patents.set_serving_corpus(conn, other.corpus_id)
    serving = conn.execute("SELECT corpus_id FROM patent_corpora WHERE is_serving").fetchall()
    assert serving == [(other.corpus_id,)]  # at most one serving corpus

    assert patents.unset_serving_corpus(conn) == other.corpus_id
    if previous is not None:
        patents.set_serving_corpus(conn, previous.corpus_id)
