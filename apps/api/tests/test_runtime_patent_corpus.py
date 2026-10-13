"""TASK-0011: the patent encoder at startup, /ready and /v1/taxonomy/status, 503 details, fingerprint.

No database and no models: the runtime's loaders, the serving-corpus reader and the API session are patched.
"""

from __future__ import annotations

import contextlib
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import runtime as runtime_mod
from app.config import Settings
from app.db.patents import Corpus
from app.routes import classify as classify_route
from app.routes import system
from app.runtime import Runtime
from app.schemas.classify import ClassifyRequest
from app.services.classification.pipeline import ClassificationPipeline, PipelineUnavailable
from app.services.confidence import fingerprint as fp_mod
from app.services.embeddings.paecter import PAECTER_MODEL_ID, PAECTER_REVISION
from app.services.embeddings.providers import EncodingPolicy

ABSTRACT = "Quantum keys are distributed among smart grid nodes for encrypted communication between substations."


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def _corpus(**overrides: Any) -> Corpus:
    now = datetime(2026, 10, 14, tzinfo=UTC)
    values = dict(corpus_id="it-test-runtime", source="hupd", description=None, label_scheme="compact",
                  label_ipc_edition="2017.01", labels_taxonomy_version="20260101", document_count=0,
                  encoder_model_id=PAECTER_MODEL_ID, encoder_model_revision=PAECTER_REVISION, input_hash="ih-1",
                  state="validated", is_serving=True, created_at=now, updated_at=now, served_at=now)  # fmt: skip
    values.update(overrides)
    return Corpus(**values)


class FakeProvider:
    policy = EncodingPolicy(provider="fake", model_id="fake/bge", model_revision="rev-1", dimension=8,
                            storage_table="ipc_embeddings", normalized=True, document_template="{text}",
                            query_template="{text}", taxonomy_max_tokens=512, query_chunk_tokens=1024,
                            dtype="float32")  # fmt: skip


class FakeReranker:
    model_id = "fake/reranker"
    revision = "rr-1"


class FakePatentEncoder:
    def __init__(self, revision: str = PAECTER_REVISION) -> None:
        self.model_id = PAECTER_MODEL_ID
        self.revision = revision
        self.device = "cpu"
        self.source = "injected"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 1024), 1 / 32, dtype=np.float32)


class FakeStore:
    def __init__(self, corpus: Corpus | None, count: int = 293_711) -> None:
        self.corpus, self.count, self.searches = corpus, count, 0

    def serving_corpus(self, session: Any) -> Corpus | None:
        return self.corpus

    def document_count(self, session: Any, corpus: Corpus) -> int:
        return self.count

    def search(self, *args: Any) -> list[Any]:
        self.searches += 1
        return []


@pytest.fixture
def loaders(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> dict[str, Any]:
    """Patch the runtime's model loaders and the fingerprint's DB reads; count encoder loads."""
    calls: dict[str, Any] = {"encoder_loads": 0, "encoder": FakePatentEncoder(), "error": None, "fingerprint": []}

    def load_encoder(settings: Settings) -> FakePatentEncoder:
        calls["encoder_loads"] += 1
        if calls["error"] is not None:
            raise calls["error"]
        return calls["encoder"]

    def fingerprint(settings: Settings, provider: Any, reranker: Any) -> tuple[dict[str, Any], str]:
        calls["fingerprint"].append(settings.neighbour_vote_enabled)
        return {"fake": True}, "fp-hash"

    monkeypatch.setattr("app.services.embeddings.providers.get_provider", lambda s: FakeProvider())
    monkeypatch.setattr("app.services.classification.reranker.get_reranker", lambda s: FakeReranker())
    monkeypatch.setattr(runtime_mod, "load_patent_encoder", load_encoder)
    monkeypatch.setattr(fp_mod, "current_fingerprint", fingerprint)
    monkeypatch.setattr("app.db.session.database_configured", lambda: False)  # startup mismatch log: no DB
    return calls


def _runtime(tmp_path: Any, **settings: Any) -> Runtime:
    return Runtime(_settings(calibration_dir=tmp_path, **settings))


# --- startup ------------------------------------------------------------------------------------------------


def test_encoder_loads_once_at_startup_and_reaches_the_pipeline(loaders: dict[str, Any], tmp_path: Any) -> None:
    rt = _runtime(tmp_path, neighbour_vote_enabled=True)
    rt._load()
    assert rt.wait_for_patent_encoder(5)
    assert rt.state == "ready", rt.error
    assert loaders["encoder_loads"] == 1
    assert rt.patent_encoder is loaders["encoder"]
    assert isinstance(rt.pipeline, ClassificationPipeline)
    assert rt.pipeline.patent_encoder is loaders["encoder"]
    assert rt.pipeline.s.neighbour_vote_enabled is True
    rt.start_loading()  # already ready: no second load
    assert loaders["encoder_loads"] == 1


def test_vote_disabled_loads_no_encoder(loaders: dict[str, Any], tmp_path: Any) -> None:
    rt = _runtime(tmp_path, neighbour_vote_enabled=False)
    rt._load()
    assert rt.wait_for_patent_encoder(5)
    assert rt.state == "ready", rt.error
    assert loaders["encoder_loads"] == 0
    assert rt.patent_encoder is None and rt.pipeline.patent_encoder is None


def test_encoder_failure_keeps_the_runtime_ready_and_is_reported(loaders: dict[str, Any], tmp_path: Any,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    loaders["error"] = RuntimeError("CUDA error: out of memory")
    rt = _runtime(tmp_path, neighbour_vote_enabled=True)
    rt._load()
    assert rt.wait_for_patent_encoder(5)
    assert rt.state == "ready", rt.error  # the definition fallback still serves
    assert rt.patent_encoder is None
    assert rt.patent_encoder_error == "RuntimeError: CUDA error: out of memory"
    monkeypatch.setattr(system, "database_configured", lambda: True)
    monkeypatch.setattr(system, "_serving_patent_corpus", lambda rt_: (True, _corpus(), 293_711))
    nv = system.neighbour_vote_state(rt)
    assert nv["state"] == "patent_encoder_unavailable" and nv["degraded"] is True
    assert "CUDA error: out of memory" in nv["reason"]
    assert nv["patent_encoder"] is None
    assert nv["patent_encoder_error"] == "RuntimeError: CUDA error: out of memory"


def test_definition_only_runtime_keeps_the_served_fingerprint(loaders: dict[str, Any], tmp_path: Any) -> None:
    """For the calibration CLI: the vote is off in the pipeline, no encoder loads, but the fingerprint is computed
    from the configured settings (vote enabled), so a fitted artifact matches the serving API."""
    rt = Runtime(_settings(calibration_dir=tmp_path, neighbour_vote_enabled=True), definition_only=True)
    rt._load()
    assert rt.wait_for_patent_encoder(5)
    assert rt.state == "ready", rt.error
    assert loaders["encoder_loads"] == 0
    assert rt.pipeline.s.neighbour_vote_enabled is False
    assert rt.settings.neighbour_vote_enabled is True
    assert loaders["fingerprint"] == [True]
    assert rt.neighbour_vote_enabled is False


def test_revision_mismatch_disables_the_vote_with_a_pipeline_warning(loaders: dict[str, Any], tmp_path: Any,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    loaders["encoder"] = FakePatentEncoder(revision="0" * 40)
    rt = _runtime(tmp_path, neighbour_vote_enabled=True)
    rt._load()
    assert rt.wait_for_patent_encoder(5)
    store = FakeStore(_corpus())
    rt.pipeline.patent_store = store
    att = rt.pipeline._attempt_vote(SimpleNamespace(), ClassifyRequest(abstract=ABSTRACT), {})
    assert att.reason == "patent_encoder_unavailable"
    assert any("was embedded with" in w and "0" * 40 in w for w in att.warnings), att.warnings
    assert store.searches == 0  # never searched with an incomparable vector
    monkeypatch.setattr(system, "database_configured", lambda: True)
    monkeypatch.setattr(system, "_serving_patent_corpus", lambda rt_: (True, _corpus(), 293_711))
    nv = system.neighbour_vote_state(rt)
    assert nv["state"] == "patent_encoder_unavailable" and nv["degraded"] is True
    assert PAECTER_REVISION in nv["reason"] and "0" * 40 in nv["reason"]


def test_startup_logs_a_corpus_revision_mismatch(loaders: dict[str, Any], tmp_path: Any,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 caplog: pytest.LogCaptureFixture) -> None:  # fmt: skip
    loaders["encoder"] = FakePatentEncoder(revision="0" * 40)
    monkeypatch.setattr("app.db.session.database_configured", lambda: True)
    monkeypatch.setattr("app.db.session.api_session", lambda: contextlib.nullcontext(object()))
    monkeypatch.setattr("app.db.patents.get_serving_corpus", lambda session: _corpus())
    rt = _runtime(tmp_path, neighbour_vote_enabled=True)
    with caplog.at_level("WARNING", logger="app.runtime"):
        rt._load()
        assert rt.wait_for_patent_encoder(5)
    assert rt.state == "ready"
    assert any("neighbour vote is disabled until they match" in r.getMessage() for r in caplog.records)


def test_startup_reaches_ready_while_the_encoder_load_blocks(loaders: dict[str, Any], tmp_path: Any,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """A stalled PaECTER load (e.g. a hung hub download) must not hold the API in `loading` (wave 4 review)."""
    release, entered = threading.Event(), threading.Event()
    enc = FakePatentEncoder()

    def blocking_load(settings: Settings) -> FakePatentEncoder:
        entered.set()
        assert release.wait(10), "test never released the encoder load"
        return enc

    monkeypatch.setattr(runtime_mod, "load_patent_encoder", blocking_load)
    rt = _runtime(tmp_path, neighbour_vote_enabled=True)
    rt._load()  # returns although the encoder load is still blocked
    try:
        assert entered.wait(5)
        assert rt.state == "ready"
        assert rt.patent_encoder_state == "loading" and rt.patent_encoder is None
        assert rt.pipeline.patent_encoder is None
        # meanwhile the vote reports the encoder as unavailable and the fallback serves
        rt.pipeline.patent_store = FakeStore(_corpus())
        att = rt.pipeline._attempt_vote(SimpleNamespace(), ClassifyRequest(abstract=ABSTRACT), {})
        assert att.reason == "patent_encoder_unavailable"
        monkeypatch.setattr(system, "database_configured", lambda: True)
        monkeypatch.setattr(system, "_serving_patent_corpus", lambda rt_: (True, _corpus(), 293_711))
        nv = system.neighbour_vote_state(rt)
        assert nv["state"] == "patent_encoder_unavailable" and "still loading" in nv["reason"]
        assert nv["patent_encoder_state"] == "loading"
    finally:
        release.set()
    assert rt.wait_for_patent_encoder(5)
    assert rt.patent_encoder_state == "ready"
    assert rt.patent_encoder is enc and rt.pipeline.patent_encoder is enc  # attached once loaded
    assert system.neighbour_vote_state(rt)["state"] == "serving"


def test_local_path_from_env_file_is_honoured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PATENT_ENCODER_LOCAL_PATH in .env reaches the loader (it used to be read from os.environ only)."""
    monkeypatch.delenv("PATENT_ENCODER_LOCAL_PATH", raising=False)
    env = tmp_path / ".env"
    env.write_text("PATENT_ENCODER_LOCAL_PATH=D:/ipc_knn/models/paecter\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.patent_encoder_local_path == Path("D:/ipc_knn/models/paecter")

    def no_hub(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a configured local path must not touch the Hugging Face hub or its cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", no_hub)
    seen: dict[str, Any] = {}
    monkeypatch.setattr("app.services.embeddings.paecter.PaecterEncoder.load",
                        lambda **kw: seen.update(kw) or FakePatentEncoder())
    runtime_mod.load_patent_encoder(s)
    assert seen["local_path"] == Path("D:/ipc_knn/models/paecter")
    assert (seen["model_id"], seen["revision"]) == (PAECTER_MODEL_ID, PAECTER_REVISION)
    assert _settings().patent_encoder_local_path is None  # no .env: unset


def test_resolve_uses_the_hf_cache_without_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def snapshot(repo: str, **kw: Any) -> str:
        calls.append(kw)
        assert kw.get("local_files_only") is True, "only the cache may be consulted"
        return str(tmp_path / "snapshots" / PAECTER_REVISION)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    assert runtime_mod.resolve_patent_encoder_path(_settings()) == tmp_path / "snapshots" / PAECTER_REVISION
    assert calls[0]["revision"] == PAECTER_REVISION


def test_hub_download_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def snapshot(repo: str, **kw: Any) -> str:
        if kw.get("local_files_only"):
            raise FileNotFoundError("not cached")
        release.wait(30)  # a stalled download
        return "never"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    t0 = time.perf_counter()
    try:
        with pytest.raises(runtime_mod.PatentEncoderUnavailable, match="did not finish within 0 s.*PATENT_ENCODER_LOCAL_PATH"):
            runtime_mod.resolve_patent_encoder_path(_settings(patent_encoder_download_timeout_s=0.2))
        assert time.perf_counter() - t0 < 5
    finally:
        release.set()


def test_hub_download_error_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def snapshot(repo: str, **kw: Any) -> str:
        raise OSError("connection reset")

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    with pytest.raises(runtime_mod.PatentEncoderUnavailable, match="OSError: connection reset"):
        runtime_mod.resolve_patent_encoder_path(_settings())


def test_stalled_download_leaves_the_api_serving(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the real resolver and loader: no local path, not cached, the download stalls -> the
    runtime is ready at once and the encoder ends `failed` with the timeout message."""
    release = threading.Event()

    def snapshot(repo: str, **kw: Any) -> str:
        if kw.get("local_files_only"):
            raise FileNotFoundError("not cached")
        release.wait(30)
        return "never"

    monkeypatch.delenv("PATENT_ENCODER_LOCAL_PATH", raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr("app.services.embeddings.providers.get_provider", lambda s: FakeProvider())
    monkeypatch.setattr("app.services.classification.reranker.get_reranker", lambda s: FakeReranker())
    monkeypatch.setattr(fp_mod, "current_fingerprint", lambda s, p, r: ({}, "fp"))
    rt = _runtime(tmp_path, neighbour_vote_enabled=True, patent_encoder_download_timeout_s=0.3)
    try:
        rt._load()
        assert rt.state == "ready"
        assert rt.wait_for_patent_encoder(5)
        assert rt.patent_encoder_state == "failed" and rt.patent_encoder is None
        assert "did not finish within" in rt.patent_encoder_error
    finally:
        release.set()


# --- readiness ----------------------------------------------------------------------------------------------


def _ready_runtime(tmp_path: Any, encoder: Any = None, **settings: Any) -> Runtime:
    rt = _runtime(tmp_path, **settings)
    rt.state, rt.patent_encoder = "ready", encoder
    return rt


@pytest.mark.parametrize(
    ("read", "encoder", "state", "reason_part"),
    [
        ((False, None, None), FakePatentEncoder(), "no_serving_corpus", "migration 0004"),
        ((True, None, None), FakePatentEncoder(), "no_serving_corpus", "No validated patent corpus is serving"),
        ((True, _corpus(), 293_711), None, "patent_encoder_unavailable", "not loaded (still loading)"),
        ((True, _corpus(encoder_model_revision="abc"), 293_711), FakePatentEncoder(), "patent_encoder_unavailable",
         "was embedded with"),
        ((True, _corpus(), 99), FakePatentEncoder(), "corpus_too_small", "fewer than NEIGHBOUR_K=100"),
        ((True, _corpus(), 100), FakePatentEncoder(), "serving", None),
    ],
    ids=["no-tables", "no-corpus", "no-encoder", "revision-mismatch", "too-small", "serving"],
)
def test_neighbour_vote_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, read: tuple[Any, ...], encoder: Any,
                              state: str, reason_part: str | None) -> None:  # fmt: skip
    monkeypatch.setattr(system, "database_configured", lambda: True)
    monkeypatch.setattr(system, "_serving_patent_corpus", lambda rt_: read)
    nv = system.neighbour_vote_state(_ready_runtime(tmp_path, encoder, neighbour_vote_enabled=True))
    assert nv["enabled"] is True
    assert nv["state"] == state
    assert nv["degraded"] is (state != "serving")
    if reason_part is None:
        assert nv["reason"] is None
    else:
        assert reason_part in nv["reason"]
    assert nv["vote_settings"] == {"k": 100, "temperature": 0.005, "beta": 1.0, "min_vote_share": 0.2,
                                   "hnsw_ef_search": 200}
    if read[1] is not None:
        assert nv["patent_corpus"] == {"corpus_id": "it-test-runtime", "source": "hupd", "document_count": read[2],
                                       "labels_ipc_edition": "2017.01", "state": "validated"}
    else:
        assert nv["patent_corpus"] is None
    if encoder is not None:
        assert nv["patent_encoder"] == {"model_id": PAECTER_MODEL_ID, "model_revision": PAECTER_REVISION}


def test_neighbour_vote_state_disabled_reads_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    def boom(rt_: Any) -> Any:
        raise AssertionError("no patent-corpus read while the vote is disabled")

    monkeypatch.setattr(system, "_serving_patent_corpus", boom)
    nv = system.neighbour_vote_state(_ready_runtime(tmp_path, neighbour_vote_enabled=False))
    assert nv == {"enabled": False, "state": "disabled", "degraded": False, "reason": nv["reason"],
                  "patent_encoder": None, "patent_encoder_state": "disabled", "patent_encoder_error": None,
                  "patent_corpus": None, "vote_settings": None}
    assert "NEIGHBOUR_VOTE_ENABLED=false" in nv["reason"]


def test_neighbour_vote_state_unreadable_corpus_is_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    def unreachable(rt_: Any) -> Any:
        raise TimeoutError("connection timeout")

    monkeypatch.setattr(system, "database_configured", lambda: True)
    monkeypatch.setattr(system, "_serving_patent_corpus", unreachable)
    nv = system.neighbour_vote_state(_ready_runtime(tmp_path, FakePatentEncoder(), neighbour_vote_enabled=True))
    assert nv["state"] == "unknown" and nv["degraded"] is True and "TimeoutError" in nv["reason"]


READY_CORPUS = {"database": "ok", "schema": "ok", "active_release": "20260101", "serving_embedding_run": "run-1",
                "validation": {"ok": True, "source_quality": []}}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("IPC_SKIP_MODEL_LOAD", "1")
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_missing_corpus_is_degraded_but_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Any) -> None:  # fmt: skip
    rt = _ready_runtime(tmp_path, FakePatentEncoder(), neighbour_vote_enabled=True)
    monkeypatch.setattr(system, "get_runtime", lambda: rt)
    monkeypatch.setattr(system, "_corpus_state", lambda: READY_CORPUS)
    monkeypatch.setattr(system, "database_configured", lambda: True)
    monkeypatch.setattr(system, "_serving_patent_corpus", lambda rt_: (True, None, None))
    r = client.get("/ready")
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["ready"] is True
    nv = body["checks"]["neighbour_vote"]
    assert nv["state"] == "no_serving_corpus" and nv["degraded"] is True
    assert nv["patent_encoder"] == {"model_id": PAECTER_MODEL_ID, "model_revision": PAECTER_REVISION}

    status = client.get("/v1/taxonomy/status").json()
    assert status["neighbour_vote"] == nv


def test_status_reports_the_serving_corpus(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    rt = _ready_runtime(tmp_path, FakePatentEncoder(), neighbour_vote_enabled=True)
    monkeypatch.setattr(system, "get_runtime", lambda: rt)
    monkeypatch.setattr(system, "_corpus_state", lambda: READY_CORPUS)
    monkeypatch.setattr(system, "database_configured", lambda: True)
    monkeypatch.setattr(system, "_serving_patent_corpus", lambda rt_: (True, _corpus(), 293_711))
    nv = client.get("/v1/taxonomy/status").json()["neighbour_vote"]
    assert nv["state"] == "serving" and nv["degraded"] is False
    assert nv["patent_corpus"] == {"corpus_id": "it-test-runtime", "source": "hupd", "document_count": 293_711,
                                   "labels_ipc_edition": "2017.01", "state": "validated"}
    assert nv["patent_encoder"]["model_revision"] == PAECTER_REVISION
    assert nv["vote_settings"]["k"] == 100


def test_skip_model_load_starts_the_app(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    rt = _runtime(tmp_path, neighbour_vote_enabled=True)
    monkeypatch.setattr(system, "get_runtime", lambda: rt)
    monkeypatch.setattr(system, "database_configured", lambda: False)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 503
    checks = r.json()["checks"]
    assert checks["models"] == "not_started"
    assert checks["neighbour_vote"]["state"] == "unknown"  # no database: the corpus cannot be read
    assert rt.patent_encoder is None and rt.state == "not_started"


# --- POST /v1/classify 503 details --------------------------------------------------------------------------


class RaisingPipeline:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def classify(self, session: Any, req: ClassifyRequest) -> Any:
        raise self.exc


class _Session:
    def begin(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


@pytest.mark.parametrize(
    ("exc", "details"),
    [
        (PipelineUnavailable("patent_corpus_unavailable",
                             "The neighbour vote could not run (no_serving_corpus) and the definition fallback could "
                             "not serve (embeddings_unavailable: none)",
                             {"reason": "no_serving_corpus", "fallback_error": "embeddings_unavailable"}),
         {"reason": "no_serving_corpus", "fallback_error": "embeddings_unavailable"}),
        (PipelineUnavailable("embeddings_unavailable", "No serving embedding run."), None),
    ],
    ids=["patent-corpus-unavailable", "definition-error"],
)
def test_503_carries_pipeline_details(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
                                      exc: PipelineUnavailable, details: Any) -> None:  # fmt: skip
    rt = _ready_runtime(tmp_path, neighbour_vote_enabled=True)
    rt.pipeline = RaisingPipeline(exc)
    monkeypatch.setattr(classify_route, "get_runtime", lambda: rt)
    monkeypatch.setattr(classify_route, "api_session", lambda: contextlib.nullcontext(_Session()))
    r = client.post("/v1/classify", json={"abstract": ABSTRACT})
    assert r.status_code == 503
    assert r.json() == {"error": exc.code, "message": str(exc), "details": details}


# --- fingerprint --------------------------------------------------------------------------------------------


class _Row:
    def __init__(self, value: Any) -> None:
        self.value = value

    def first(self) -> Any:
        return self.value

    def scalar_one(self) -> Any:
        return self.value


class FingerprintSession:
    def __init__(self, corpus: tuple[str, str] | None, tables: bool = True) -> None:
        self.corpus, self.tables, self.sql = corpus, tables, []

    def execute(self, stmt: Any, params: Any = None) -> _Row:
        sql = str(stmt)
        self.sql.append(sql)
        if "embedding_runs" in sql:
            return _Row(("run-input-hash",))
        if "to_regclass('public.patent_corpora')" in sql:
            return _Row(self.tables)
        if "FROM patent_corpora" in sql:
            return _Row(self.corpus)
        raise AssertionError(sql)


@pytest.fixture
def fingerprint_db(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    db: dict[str, Any] = {"session": FingerprintSession(("hupd-2017", "ih-1"))}
    release = SimpleNamespace(taxonomy_version="20260101", source_hashes={"scheme": "s1"})
    monkeypatch.setattr("app.db.session.api_session", lambda: contextlib.nullcontext(db["session"]))
    monkeypatch.setattr("app.db.repository.active_release", lambda session: release)
    return db


def _fp(settings: Settings) -> tuple[dict[str, Any], str]:
    return fp_mod.current_fingerprint(settings, FakeProvider(), FakeReranker())


def test_fingerprint_changes_with_corpus_id_input_hash_and_encoder_revision(fingerprint_db: dict[str, Any]) -> None:
    on = _settings(neighbour_vote_enabled=True)
    comp, base = _fp(on)
    assert comp["neighbour_vote"]["patent_corpus"] == {"corpus_id": "hupd-2017", "input_hash": "ih-1"}
    assert comp["neighbour_vote"]["patent_encoder"]["model_revision"] == PAECTER_REVISION
    assert comp["neighbour_vote"]["vote_settings"] == {
        "neighbour_k": 100, "neighbour_temperature": 0.005, "neighbour_beta": 1.0, "neighbour_min_vote_share": 0.2,
        "patent_hnsw_ef_search": 200}  # fmt: skip
    assert _fp(on)[1] == base  # deterministic

    fingerprint_db["session"] = FingerprintSession(("uspto-2021-2025", "ih-1"))
    assert _fp(on)[1] != base  # corpus id
    fingerprint_db["session"] = FingerprintSession(("hupd-2017", "ih-2"))
    assert _fp(on)[1] != base  # corpus input hash
    fingerprint_db["session"] = FingerprintSession(("hupd-2017", "ih-1"))
    assert _fp(_settings(neighbour_vote_enabled=True, patent_encoder_model_revision="0" * 40))[1] != base  # revision
    assert _fp(_settings(neighbour_vote_enabled=True, neighbour_k=50, patent_hnsw_ef_search=200))[1] != base
    fingerprint_db["session"] = FingerprintSession(None)
    comp_none, h_none = _fp(on)
    assert h_none != base and comp_none["neighbour_vote"]["patent_corpus"] == {"corpus_id": None, "input_hash": None}
    fingerprint_db["session"] = FingerprintSession(None, tables=False)
    assert _fp(on)[1] == h_none  # no patent tables == no serving corpus


def test_fingerprint_without_the_vote_is_unchanged_and_ignores_the_corpus(fingerprint_db: dict[str, Any]) -> None:
    off = _settings(neighbour_vote_enabled=False)
    comp, h = _fp(off)
    assert "neighbour_vote" not in comp
    assert not any("patent_corpora" in sql for sql in fingerprint_db["session"].sql)
    fingerprint_db["session"] = FingerprintSession(("uspto-2021-2025", "ih-9"))
    assert _fp(off)[1] == h
    assert _fp(_settings(neighbour_vote_enabled=False, patent_encoder_model_revision="0" * 40))[1] == h
    # the 1.x components are exactly what they were before TASK-0011
    assert set(comp) == {"embedding", "embedding_run", "reranker", "scope_formatter", "candidate_generation",
                         "taxonomy", "field_mapping_version"}
