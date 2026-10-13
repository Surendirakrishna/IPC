"""TASK-0014: POST /v1/classify end to end on a real database with a small synthetic patent corpus.

What is real: the FastAPI app and route (``TestClient``), ``ClassificationPipeline`` with the vote enabled, the
database (``patent_corpora`` / ``patent_documents`` / ``patent_embeddings``, a real partial HNSW index), the
validation and activation services (``validate_corpus``, ``activate_corpus``), the real ``DbPatentVoteStore`` and the
real vote. What is faked: PaECTER (a fixed query vector) and the definition path's catalogue and BGE/reranker (the
same fakes as ``test_pipeline_neighbour.py``), so no model loads and no taxonomy data is needed.

The test builds an ``it-test-*`` corpus of 300 documents in three label clusters, activates it, and checks the
contract fields of the vote response; then it stops serving the corpus and checks that the very same request
takes the definition fallback with ``fallback_reason == "no_serving_corpus"``.

Run (sakthi has no local docker database; user-approved, opt-in):
    IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1 uv run pytest -m integration tests/test_integration_neighbour.py -s

Safety: only ``it-test-*`` corpora and their indexes are created, served or deleted (try/finally). It refuses to run
when any other corpus is serving, loading or embedded, and it prints the patent table counts before and after (they
must match).
"""

from __future__ import annotations

import os
import uuid
from datetime import date
from typing import Any

import numpy as np
import pytest

from app.config import Settings
from app.schemas.classify import ClassifyRequest, ClassifyResponse
from app.services.classification import pipeline as pl
from app.services.classification.pipeline import ClassificationPipeline, ServingState
from app.services.confidence.calibrator import ConfidenceService
from app.services.embeddings.paecter import PAECTER_MODEL_ID, PAECTER_REVISION
from app.services.fields.mapping import load_mapping

# Helpers shared with the pipeline unit tests (tests/ is on sys.path; no package).
from test_pipeline_neighbour import ABSTRACT, TITLE, FakeEncoder, FakeRepo, FakeReranker, patch_catalogue  # isort: skip

TEST_PREFIX = "it-test-"
PATENT_TABLES = ("patent_corpora", "patent_documents", "patent_embeddings")
BUSY_STATES = ("loading", "embedded")
K = 50
N_A, N_B, N_C = 150, 60, 90  # H04L 9/08 cluster, H02J 13/00 cluster, G06Q 50/06 filler
DIM = 1024


def _integration_target() -> str | None:
    target = os.environ.get("IPC_DB_TARGET")
    if target == "local" or (target == "sakthi" and os.environ.get("IPC_INTEGRATION_ALLOW_SAKTHI") == "1"):
        return target
    return None


def _refusal(corpora: list[tuple[str, str, bool]]) -> str | None:
    """Why this test must not run against this database, or None: any real corpus serving, loading or embedded."""
    for corpus_id, state, is_serving in corpora:
        if corpus_id.startswith(TEST_PREFIX):
            continue
        if is_serving:
            return f"refusing to run: corpus {corpus_id!r} is serving"
        if state in BUSY_STATES:
            return f"refusing to run: corpus {corpus_id!r} is {state!r} (a load may be running)"
    return None


def test_refusal_rule() -> None:
    assert _refusal([]) is None
    assert _refusal([("it-test-a", "loading", True), ("it-test-b", "embedded", False)]) is None
    assert _refusal([("hupd-2017", "validated", False), ("old", "failed", False)]) is None
    assert "serving" in (_refusal([("hupd-2017", "validated", True)]) or "")
    assert "'loading'" in (_refusal([("hupd-2017", "loading", False)]) or "")
    assert "'embedded'" in (_refusal([("uspto-2021", "embedded", False)]) or "")


def _counts(conn: Any) -> dict[str, int]:
    out = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in PATENT_TABLES}  # fixed names
    conn.rollback()
    return out


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(centre: np.ndarray, rng: np.random.Generator, spread: float) -> np.ndarray:
    v = centre + spread * _unit(rng)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FixedPatentEncoder:
    """PaECTER stand-in: every text encodes to the same vector, near the H04L cluster's centre."""

    model_id = PAECTER_MODEL_ID
    revision = PAECTER_REVISION

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.tile(self.vector, (len(texts), 1))


@pytest.fixture
def db():
    """(conn, created_corpora, created_indexes) on the opted-in target; teardown removes only it-test-* rows."""
    target = _integration_target()
    if target is None:
        pytest.skip("set IPC_DB_TARGET=local (or IPC_DB_TARGET=sakthi with IPC_INTEGRATION_ALLOW_SAKTHI=1)")
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().ipc_db_target == target
    from app.db.session import connect_direct

    conn = connect_direct()
    reason = _refusal(conn.execute("SELECT corpus_id, state, is_serving FROM patent_corpora").fetchall())
    conn.commit()
    if reason is not None:
        conn.close()
        pytest.fail(reason)
    created: list[str] = []
    try:
        yield conn, created
    finally:
        from psycopg import sql

        from app.services.patents import index as ix

        assert all(c.startswith(TEST_PREFIX) for c in created), created
        conn.rollback()
        conn.autocommit = True
        for cid in created:
            conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(ix.index_name(cid))))
        conn.execute("UPDATE patent_corpora SET is_serving = false WHERE corpus_id = ANY(%s) AND corpus_id LIKE %s",
                     [created, TEST_PREFIX + "%"])
        conn.execute("DELETE FROM patent_corpora WHERE corpus_id = ANY(%s) AND corpus_id LIKE %s",
                     [created, TEST_PREFIX + "%"])
        conn.close()
        try:
            from app.db.session import get_pooled_engine

            get_pooled_engine().dispose()
        except Exception:  # noqa: BLE001 - cleanup only
            pass


def _build_serving_corpus(conn: Any, created: list[str], rng: np.random.Generator) -> tuple[str, np.ndarray, dict]:
    """Load, embed (synthetic vectors), index, validate and activate a 300-document corpus through the services.
    Returns (corpus_id, the query vector, the validation manifest)."""
    from app.db import patents
    from app.db.patents import CorpusSpec, PatentDocument
    from app.services.patents import index as ix

    centre_a, centre_b = _unit(rng), _unit(rng)
    corpus_id = f"{TEST_PREFIX}neighbour-{uuid.uuid4().hex[:8]}"
    created.append(corpus_id)
    patents.create_or_get_corpus(conn, CorpusSpec(corpus_id, "hupd", "compact", PAECTER_MODEL_ID, PAECTER_REVISION,
                                                  label_ipc_edition="2017.01"))  # fmt: skip
    docs: list[PatentDocument] = []
    vectors: dict[str, np.ndarray] = {}
    for i in range(N_A):
        labels = [("H04L", "H04L 9/00", "H04L 9/08")]
        if i < 60:
            labels.append(("H04L", "H04L 9/00", "H04L 9/32"))  # secondary label on 60 of the 150
        docs.append(PatentDocument(f"a{i:03d}", f"Key distribution {i}", "A synthetic abstract about key distribution.",
                                   date(2017, 1, 3), None, ["H04L908"], labels))  # fmt: skip
        vectors[f"a{i:03d}"] = _near(centre_a, rng, 0.3)
    for i in range(N_B):
        docs.append(PatentDocument(f"b{i:03d}", f"Grid {i}", "A synthetic abstract about power grids.",
                                   date(2017, 1, 3), None, ["H02J1300"], [("H02J", "H02J 13/00", None)]))  # fmt: skip
        vectors[f"b{i:03d}"] = _near(centre_b, rng, 0.3)
    for i in range(N_C):
        docs.append(PatentDocument(f"c{i:03d}", f"Energy {i}", "A synthetic abstract about energy supply.",
                                   date(2017, 1, 3), None, ["G06Q5006"], [("G06Q", "G06Q 50/00", "G06Q 50/06")]))  # fmt: skip
        vectors[f"c{i:03d}"] = _unit(rng)
    assert patents.upsert_documents(conn, corpus_id, docs, batch_size=500) == N_A + N_B + N_C
    assert patents.insert_embeddings(conn, corpus_id, vectors.items(), batch_size=500) == N_A + N_B + N_C
    assert patents.set_corpus_state(conn, corpus_id, "embedded", document_count=len(docs)).state == "embedded"

    built = ix.build_patent_index(conn, corpus_id, m=16, ef_construction=64)
    assert built["status"] == "created" and built["embedding_count"] == len(docs)
    conn.commit()
    validated = ix.validate_corpus(conn, corpus_id, probes=20, k=K, threshold=0.95)
    conn.commit()
    assert validated["passed"] is True, validated
    activated = ix.activate_corpus(conn, corpus_id)
    assert activated.is_serving and activated.state == "validated"
    query = _near(centre_a, rng, 0.05)  # nearest by far: the 150 H04L patents
    return corpus_id, query, validated


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, encoder: FixedPatentEncoder):
    """The real app with a ready runtime holding a vote-enabled pipeline: real database store, faked models."""
    from fastapi.testclient import TestClient

    from app import runtime as runtime_mod
    from app.main import create_app

    s = Settings(_env_file=None, neighbour_vote_enabled=True, neighbour_k=K, neighbour_temperature=0.005,
                 neighbour_beta=1.0, neighbour_min_vote_share=0.2, patent_hnsw_ef_search=200)  # fmt: skip
    assert not s.retain_inputs  # the route must not write classification_audit rows to a real database
    repo = FakeRepo()
    patch_catalogue(monkeypatch, pl, repo)
    monkeypatch.setattr(pl, "serving_run", lambda session, release, provider: ServingState(release, "run-test",
                                                                                          "ipc_embeddings", {}))  # fmt: skip
    pipeline = ClassificationPipeline(s, FakeEncoder(), FakeReranker(), ConfidenceService(tmp_path, "fp-test"),
                                      load_mapping(s.field_mapping_path), patent_encoder=encoder)  # fmt: skip
    rt = runtime_mod.Runtime(s)
    rt.state, rt.pipeline = "ready", pipeline
    monkeypatch.setattr(runtime_mod, "_runtime", rt)
    monkeypatch.setattr("app.routes.classify.get_settings", lambda: s)
    return TestClient(create_app())  # no `with`: the lifespan (model loading) must not run


@pytest.mark.integration
def test_classify_uses_the_neighbour_vote_then_falls_back_when_the_corpus_stops_serving(db, monkeypatch, tmp_path) -> None:
    conn, created = db
    rng = np.random.default_rng(20261014)
    before = _counts(conn)
    corpus_id, query, validated = _build_serving_corpus(conn, created, rng)
    print(f"\ncorpus {corpus_id}: recall@{K} = {validated['recall_at_k']} over {validated.get('probes')} probes")
    encoder = FixedPatentEncoder(query)
    client = _client(monkeypatch, tmp_path, encoder)
    body = ClassifyRequest(abstract=ABSTRACT, title=TITLE, target_level="subgroup", max_results=3).model_dump(
        exclude_none=True)  # fmt: skip

    # --- the vote serves -----------------------------------------------------------------------------------
    r = client.post("/v1/classify", json=body)
    assert r.status_code == 200, r.text
    resp = ClassifyResponse.model_validate(r.json())  # the contract validators run on the wire form
    wire = r.json()
    assert wire["classifier"] == "neighbour_vote" and wire["fallback_reason"] is None
    assert encoder.calls == 1
    vote = resp.vote
    assert vote is not None and vote.first_code == "H04L 9/08" and vote.first_code_level == "subgroup"
    assert vote.first_code_share.subclass == pytest.approx(1.0, abs=1e-3)  # all 50 nearest carry H04L
    settings = resp.provenance.vote_settings
    assert settings is not None and settings.k == K == vote.neighbour_count
    assert vote.first_code_share.subclass >= settings.min_vote_share
    # provenance names the synthetic corpus and the encoder
    pc = resp.provenance.patent_corpus
    assert pc is not None and pc.corpus_id == corpus_id and pc.document_count == N_A + N_B + N_C
    assert pc.source == "hupd" and pc.labels_ipc_edition == "2017.01"
    pe = resp.provenance.patent_encoder
    assert pe is not None and (pe.model_id, pe.model_revision) == (PAECTER_MODEL_ID, PAECTER_REVISION)

    # candidates are in vote order: the first code leads, the rest follow the vote's own exact ranking
    assert resp.candidates and resp.candidates[0].display_code == vote.first_code
    assert all(c.origin == ["neighbour_vote"] and c.confidence.status == "vote_share_uncalibrated"
               for c in resp.candidates)  # fmt: skip
    exact = [c.vote_share.exact for c in resp.candidates]
    assert exact[0] == pytest.approx(1.0, abs=1e-3) and exact == sorted(exact, reverse=True)
    assert all(c.confidence.score == c.vote_share.exact for c in resp.candidates)
    ranking = [c.display_code for c in vote.top_codes.exact]
    codes = [c.display_code for c in resp.candidates]
    assert [c for c in ranking if c in codes][: len(codes)] == codes  # same relative order as the vote ranking
    assert "H04L 9/32" in codes  # the 60 secondary labels (beta = 1) make it the second subgroup

    # nearest patents: 10, most similar first, all from the H04L cluster, carrying its label
    nb = vote.neighbours
    assert len(nb) == 10
    assert [n.similarity for n in nb] == sorted((n.similarity for n in nb), reverse=True)
    assert all(n.document_id.startswith("a") and "H04L 9/08" in n.ipc_codes for n in nb), nb
    assert all(0 < n.vote_weight <= 1 for n in nb)

    # --- remove the serving corpus: the same request takes the fallback ---------------------------------------
    from app.db import patents

    assert patents.unset_serving_corpus(conn) == corpus_id
    r = client.post("/v1/classify", json=body)
    assert r.status_code == 200, r.text
    fb = ClassifyResponse.model_validate(r.json())
    assert fb.classifier == "definition_fallback" and fb.fallback_reason == "no_serving_corpus"
    assert fb.vote is None and fb.provenance.patent_corpus is None
    assert encoder.calls == 1  # no corpus: the vote never encoded the query
    assert all("neighbour_vote" not in c.origin and c.vote_share.is_empty() for c in fb.candidates)
    assert fb.candidates, "the definition fallback must still answer"
    assert any("No validated patent corpus is serving" in w for w in fb.warnings)

    # --- serving again restores the vote (activation needs no restart) -------------------------------------
    from app.services.patents import index as ix

    ix.activate_corpus(conn, corpus_id)
    again = ClassifyResponse.model_validate(client.post("/v1/classify", json=body).json())
    assert again.classifier == "neighbour_vote" and again.vote is not None and again.vote.first_code == "H04L 9/08"
    patents.unset_serving_corpus(conn)

    # --- cleanup is verified by the fixture; record that the tables return to their starting counts ----------
    from psycopg import sql

    conn.autocommit = True
    conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(ix.index_name(corpus_id))))
    conn.execute("DELETE FROM patent_corpora WHERE corpus_id = %s AND corpus_id LIKE %s", [corpus_id, TEST_PREFIX + "%"])
    conn.autocommit = False
    after = _counts(conn)
    print(f"patent table counts before={before} after={after}")
    assert before == after
