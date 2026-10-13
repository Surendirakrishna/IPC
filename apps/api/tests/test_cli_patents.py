"""TASK-0009: patent corpus CLI commands (Typer CliRunner, services monkeypatched, no database, no models).

The connection is a ``FakeConn`` that answers only what cli.py itself asks (advisory locks, the active release,
the catalogue); every service (ingest, embed, index, validate, activate) and ``db.patents.get_corpus`` is patched
with a fake that records its calls and moves the fake corpus through its states.

One integration test takes the advisory locks on a real database (two sessions); it is opt-in (see the bottom).
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from app import cli
from app.config import Settings
from app.db.patents import Corpus, CorpusStateError
from app.services.embeddings.paecter import PAECTER_MODEL_ID, PAECTER_REVISION, PaecterError
from app.services.patents.index import PatentIndexError
from app.services.patents.ingest import HoldoutLeak

runner = CliRunner()
VERSION = "20260101"
NOW = datetime(2026, 10, 14, tzinfo=UTC)
HNSW = ["--m", "16", "--ef-construction", "64"]


def _corpus(corpus_id: str, state: str, count: int = 0, *, serving: bool = False) -> Corpus:
    return Corpus(corpus_id, "hupd", None, "compact", None, VERSION, count, PAECTER_MODEL_ID, PAECTER_REVISION,
                  "ih" if state in ("embedded", "validated") else None, state, serving, NOW, NOW, None)  # fmt: skip


class _Res:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    def fetchone(self) -> Any:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class FakeConn:
    """Answers cli.py's own statements. ``busy`` lists lock namespaces another session holds."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.autocommit = False
        self.closed = False

    def __enter__(self) -> FakeConn:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True

    def execute(self, query: str, params: Any = None) -> _Res:
        w = self.world
        w.sql.append(query)
        if "pg_try_advisory_lock" in query:
            w.locks.append(("lock", params[0], params[1] if len(params) > 1 else None))
            return _Res([(params[0] not in w.busy,)])
        if "pg_advisory_unlock" in query:
            w.locks.append(("unlock", params[0], params[1] if len(params) > 1 else None))
            return _Res([(True,)])
        if "FROM taxonomy_releases" in query:
            return _Res([(VERSION,)])
        if "FROM ipc_records" in query:
            return _Res([(c,) for c in ("H04L", "H04L 9/00", "H04L 9/08")])
        if "to_regclass('public.patent_corpora')" in query:
            return _Res([(True,)])
        if "FROM patent_corpora" in query:
            return _Res([tuple(c.__dict__.values()) for c in w.corpora.values()])
        if "FROM patent_documents GROUP BY" in query:
            return _Res([(cid, n) for cid, n in w.documents.items()])
        if "FROM patent_embeddings GROUP BY" in query:
            return _Res([(cid, n) for cid, n in w.embedded.items()])
        raise AssertionError(f"unexpected SQL from cli.py: {query}")

    def rollback(self) -> None:
        pass

    def commit(self) -> None:
        pass


@dataclass
class World:
    """The fake database state and every recorded service call."""

    corpora: dict[str, Corpus] = field(default_factory=dict)
    documents: dict[str, int] = field(default_factory=dict)  # stored documents per corpus
    embedded: dict[str, int] = field(default_factory=dict)  # documents with a vector
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)
    locks: list[tuple[str, int, Any]] = field(default_factory=list)
    busy: set[int] = field(default_factory=set)
    fail: dict[str, BaseException] = field(default_factory=dict)  # service name -> exception to raise
    stored: int = 5  # documents an ingest writes
    validate_passes: bool = True
    imported: int | None = None  # documents import_precomputed gives a vector (None: all)
    encoder_loads: list[Any] = field(default_factory=list)

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def call(self, name: str) -> dict[str, Any]:
        return next(kw for n, kw in self.calls if n == name)

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail:
            raise self.fail.pop(name)

    def set_state(self, corpus_id: str, state: str, count: int | None = None) -> None:
        c = self.corpora[corpus_id]
        self.corpora[corpus_id] = replace(c, state=state, document_count=c.document_count if count is None else count)


class FakeStore:
    def __init__(self, world: World) -> None:
        self.w = world

    def corpus(self, corpus_id: str) -> Corpus | None:
        return self.w.corpora.get(corpus_id)

    def counts(self, corpus_id: str) -> tuple[int, int]:
        return self.w.documents.get(corpus_id, 0), self.w.embedded.get(corpus_id, 0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    source = tmp_path / "source"
    source.mkdir()
    (source / "wipo_ipc_concordance_20260101.zip").write_bytes(b"")  # presence is what the CLI checks
    return Settings(_env_file=None, manifests_dir=tmp_path / "manifests", source_dir=source,
                    hupd_data_dir=tmp_path / "hupd", uspto_odp_api_key="test-key-not-real",
                    patent_encoder_local_path=tmp_path / "paecter")  # fmt: skip


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> World:
    w = World()
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("app.db.session.connect_direct", lambda: FakeConn(w))
    monkeypatch.setattr("app.db.patents.get_corpus", lambda conn, cid: w.corpora.get(cid))

    # --- sources and concordance
    def for_years(data_dir: Any, years: Any) -> Any:
        w.calls.append(("hupd_reader", {"data_dir": data_dir, "years": list(years)}))
        return [SimpleNamespace(document_id=f"h{i}") for i in range(10)]

    class FakeUspto:
        def __init__(self, years: Any, api_key: Any, *, limit: int | None = None) -> None:
            w.calls.append(("uspto_reader", {"years": list(years), "api_key": api_key, "limit": limit}))

    monkeypatch.setattr("app.services.patents.sources.hupd.HupdReader.for_years", for_years)
    monkeypatch.setattr("app.services.patents.sources.uspto_odp.UsptoOdpReader", FakeUspto)

    def load_concordance(path: Any, catalogue: Any = None) -> Any:
        w.calls.append(("load_concordance", {"path": path, "catalogue": catalogue}))
        return SimpleNamespace(sources=["wipo_ipc_concordance_20260101.zip"], catalogue=catalogue)

    monkeypatch.setattr("app.services.patents.concordance.load_concordance", load_concordance)

    # --- ingest
    def ingest(reader: Any, corpus_id: str, catalogue: Any, **kw: Any) -> dict[str, Any]:
        w.calls.append(("ingest", {"reader": reader, "corpus_id": corpus_id, "catalogue": catalogue, **kw}))
        w._maybe_fail("ingest")
        w.corpora.setdefault(corpus_id, _corpus(corpus_id, "loading"))
        w.documents[corpus_id] = w.stored
        w.set_state(corpus_id, "loading", w.stored)
        manifest = {"corpus_id": corpus_id, "counts": {"stored": w.stored}, "corpus_document_count": w.stored,
                    "state": "loading", "holdout_ids": ["x1", "x2"],
                    "concordance": None if kw["concordance"] is None else {"sources": kw["concordance"].sources}}
        if kw.get("holdout_size") and not kw.get("allow_holdout_leak") and "leak" in corpus_id:
            raise HoldoutLeak(f"2 hold-out id(s) are in corpus {corpus_id!r}", {**manifest, "holdout_leak": ["x1"]})
        return manifest

    monkeypatch.setattr("app.services.patents.ingest.ingest", ingest)

    # --- embed
    def rep(corpus_id: str, **extra: Any) -> SimpleNamespace:
        return SimpleNamespace(as_dict=lambda: {"corpus_id": corpus_id, **extra})

    def finish(corpus_id: str) -> None:
        if w.embedded.get(corpus_id, 0) == w.documents.get(corpus_id, 0) > 0:
            w.set_state(corpus_id, "embedded")

    def embed_corpus(store: Any, corpus_id: str, encoder: Any, *, page_size: int) -> Any:
        w.calls.append(("embed_corpus", {"corpus_id": corpus_id, "encoder": encoder, "page_size": page_size}))
        w._maybe_fail("embed")
        w.embedded[corpus_id] = w.documents.get(corpus_id, 0)
        finish(corpus_id)
        return rep(corpus_id, embedded_now=w.documents.get(corpus_id, 0))

    def import_precomputed(store: Any, corpus_id: str, src: Any, *, batch_size: int) -> Any:
        w.calls.append(("import_precomputed", {"corpus_id": corpus_id, "source": src, "batch_size": batch_size}))
        n = w.documents.get(corpus_id, 0)
        w.embedded[corpus_id] = n if w.imported is None else w.imported
        finish(corpus_id)
        return rep(corpus_id, imported=w.embedded[corpus_id])

    def finalize_corpus(store: Any, corpus_id: str, policy: Any) -> Any:
        w.calls.append(("finalize_corpus", {"corpus_id": corpus_id, "policy": policy}))
        finish(corpus_id)
        return rep(corpus_id)

    def from_directory(directory: Any, id_list: Any, *, with_texts: bool = True) -> Any:
        w.calls.append(("from_directory", {"directory": directory, "id_list": id_list, "with_texts": with_texts}))
        return SimpleNamespace(directory=directory)

    def load_encoder(**kw: Any) -> Any:
        w.encoder_loads.append(kw)
        if "encoder" in w.fail:
            raise w.fail.pop("encoder")
        return SimpleNamespace(model_id=kw["model_id"], revision=kw["revision"])

    monkeypatch.setattr("app.services.patents.embed.DbEmbeddingStore", lambda conn: FakeStore(w))
    monkeypatch.setattr("app.services.patents.embed.embed_corpus", embed_corpus)
    monkeypatch.setattr("app.services.patents.embed.import_precomputed", import_precomputed)
    monkeypatch.setattr("app.services.patents.embed.finalize_corpus", finalize_corpus)
    monkeypatch.setattr("app.services.patents.embed.PrecomputedSource.from_directory", from_directory)
    monkeypatch.setattr("app.services.embeddings.paecter.PaecterEncoder.load", load_encoder)

    # --- index / validate / activate / retire
    def build(conn: Any, corpus_id: str, **kw: Any) -> dict[str, Any]:
        w.calls.append(("build_patent_index", {"corpus_id": corpus_id, **kw}))
        w._maybe_fail("index")
        c = w.corpora[corpus_id]
        if c.is_serving:
            raise CorpusStateError(f"corpus {corpus_id!r} is serving; its index cannot be built or replaced")
        return {"corpus_id": corpus_id, "status": "created", "bytes": 4096, "state": "embedded"}

    def validate(conn: Any, corpus_id: str, **kw: Any) -> dict[str, Any]:
        w.calls.append(("validate_corpus", {"corpus_id": corpus_id, **kw}))
        w._maybe_fail("validate")
        if w.validate_passes:
            w.set_state(corpus_id, "validated")
            return {"corpus_id": corpus_id, "passed": True, "reasons": [], "recall_at_k": 0.99, "state": "validated"}
        return {"corpus_id": corpus_id, "passed": False, "reasons": ["recall@100 0.9 < threshold 0.95"],
                "recall_at_k": 0.9, "state": "embedded"}

    def activate(conn: Any, corpus_id: str) -> Corpus:
        w.calls.append(("activate_corpus", {"corpus_id": corpus_id}))
        c = replace(w.corpora[corpus_id], is_serving=True, served_at=NOW)
        w.corpora[corpus_id] = c
        return c

    def retire(conn: Any, corpus_id: str, *, stop_serving: bool) -> dict[str, Any]:
        w.calls.append(("retire_corpus", {"corpus_id": corpus_id, "stop_serving": stop_serving}))
        return {"corpus_id": corpus_id, "state": "embedded", "dropped": True}

    monkeypatch.setattr("app.services.patents.index.build_patent_index", build)
    monkeypatch.setattr("app.services.patents.index.validate_corpus", validate)
    monkeypatch.setattr("app.services.patents.index.activate_corpus", activate)
    monkeypatch.setattr("app.services.patents.index.retire_corpus", retire)
    monkeypatch.setattr("app.services.patents.index.corpus_index",
                        lambda conn, cid: {"index": f"patent_hnsw_{cid}", "bytes": 3 * 2**20, "valid": True})
    return w


def _manifest(settings: Settings, step: str, corpus_id: str) -> dict[str, Any]:
    return json.loads((settings.manifests_dir / "patents" / f"{step}_{corpus_id}.json").read_text(encoding="utf-8"))


def _invoke(*args: str) -> Any:
    return runner.invoke(cli.app, list(args))


# --- ingest-patents -----------------------------------------------------------------------------------------


def test_ingest_hupd_parses_arguments_and_writes_the_manifest(world: World, settings: Settings) -> None:
    r = _invoke("ingest-patents", "--source", "hupd", "--corpus", "it-hupd", "--years", "2017", "--sample", "4",
                "--seed", "7", "--holdout", "2", "--label-ipc-edition", "2017.01", "--ingest-batch", "500")
    assert r.exit_code == 0, r.output
    kw = world.call("ingest")
    assert kw["corpus_id"] == "it-hupd"
    assert (kw["seed"], kw["sample_size"], kw["holdout_size"], kw["batch_size"]) == (7, 4, 2, 500)
    assert kw["labels_taxonomy_version"] == VERSION and kw["label_ipc_edition"] == "2017.01"
    assert (kw["encoder_model_id"], kw["encoder_model_revision"]) == (PAECTER_MODEL_ID, PAECTER_REVISION)
    assert kw["catalogue"] == {"H04L", "H04L 9/00", "H04L 9/08"}
    assert kw["allow_holdout_leak"] is False
    assert kw["source_parameters"] == {"source": "hupd", "years": [2017], "limit": None,
                                       "data_dir": str(settings.hupd_data_dir)}
    assert world.call("hupd_reader") == {"data_dir": settings.hupd_data_dir, "years": [2017]}
    # the concordance is loaded from data/source/ with the release's catalogue and passed to ingest
    assert world.call("load_concordance") == {"path": settings.source_dir, "catalogue": kw["catalogue"]}
    assert kw["concordance"].catalogue == kw["catalogue"]
    m = _manifest(settings, "ingest", "it-hupd")
    assert m["concordance_applied"] is True and m["concordance_note"].startswith("applied: 1 WIPO")
    assert m["completed"] is True and m["holdout_ids"] == ["x1", "x2"]
    assert m["parameters"] == {"source": "hupd", "years": [2017], "limit": None, "sample": 4, "seed": 7,
                               "holdout": 2, "concordance": True}
    out = json.loads(r.stdout)
    assert out["holdout_ids"] == 2  # a count on screen; the ids are in the manifest


def test_ingest_uspto_warns_that_the_endpoint_is_unverified(world: World, settings: Settings) -> None:
    r = _invoke("ingest-patents", "--source", "uspto-odp", "--corpus", "it-uspto", "--years", "2021-2025",
                "--limit", "1000000")
    assert r.exit_code == 0, r.output
    assert "UNVERIFIED" in r.stderr and "USPTO_ODP_API_KEY" in r.stderr
    kw = world.call("uspto_reader")
    assert kw["years"] == [2021, 2022, 2023, 2024, 2025] and kw["limit"] == 1_000_000
    assert kw["api_key"].get_secret_value() == "test-key-not-real"
    assert "test-key-not-real" not in r.output


def test_ingest_without_concordance_files_records_it_was_not_applied(world: World, settings: Settings) -> None:
    for f in settings.source_dir.iterdir():
        f.unlink()
    r = _invoke("ingest-patents", "--source", "hupd", "--corpus", "it-noconc", "--years", "2017")
    assert r.exit_code == 0, r.output
    assert world.call("ingest")["concordance"] is None
    assert "load_concordance" not in world.names()
    m = _manifest(settings, "ingest", "it-noconc")
    assert m["concordance_applied"] is False and "no wipo_ipc_concordance_*.zip" in m["concordance_note"]


def test_ingest_no_concordance_flag(world: World, settings: Settings) -> None:
    r = _invoke("ingest-patents", "--source", "hupd", "--corpus", "it-off", "--years", "2017", "--no-concordance")
    assert r.exit_code == 0, r.output
    assert world.call("ingest")["concordance"] is None
    assert _manifest(settings, "ingest", "it-off")["concordance_note"] == "disabled (--no-concordance)"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--source", "patentsview", "--corpus", "c", "--years", "2017"], "--source must be one of"),
        (["--source", "hupd", "--corpus", "c", "--years", "17"], "is not YYYY or YYYY-YYYY"),
        (["--source", "hupd", "--corpus", "c", "--years", "2025-2021"], "ends before it starts"),
        (["--source", "hupd", "--corpus", "c"], "Missing option '--years'"),
        (["--source", "hupd", "--corpus", "c", "--years", "2017", "--holdout", "-1"], "--holdout"),
    ],
)
def test_ingest_rejects_bad_arguments(world: World, args: list[str], message: str) -> None:
    r = _invoke("ingest-patents", *args)
    assert r.exit_code == 2, r.output
    assert message in r.output
    assert "ingest" not in world.names()


def test_holdout_leak_fails_clearly_and_keeps_the_manifest(world: World, settings: Settings) -> None:
    r = _invoke("ingest-patents", "--source", "hupd", "--corpus", "it-leak", "--years", "2017", "--holdout", "2")
    assert r.exit_code == 1
    assert "HoldoutLeak" in r.stderr and "--allow-holdout-leak" in r.stderr
    m = _manifest(settings, "ingest", "it-leak")
    assert m["completed"] is False and m["holdout_leak"] == ["x1"]
    r = _invoke("ingest-patents", "--source", "hupd", "--corpus", "it-leak", "--years", "2017", "--holdout", "2",
                "--allow-holdout-leak")
    assert r.exit_code == 0, r.output
    assert world.calls[-1][1]["allow_holdout_leak"] is True


def test_manifest_name_of_a_corpus_id_with_a_colon(world: World, settings: Settings) -> None:
    r = _invoke("ingest-patents", "--source", "hupd", "--corpus", "it:colon", "--years", "2017")
    assert r.exit_code == 0, r.output
    assert (settings.manifests_dir / "patents" / "ingest_it_colon.json").is_file()


def test_limited_reader_is_reiterable() -> None:
    src = list(range(10))
    limited = cli.LimitedReader(src, 3)
    assert list(limited) == [0, 1, 2] and list(limited) == [0, 1, 2]
    assert iter(limited) is not limited  # ingest treats it as re-iterable (two-pass sampling)


def test_hupd_limit_wraps_the_reader(world: World) -> None:
    r = _invoke("ingest-patents", "--source", "hupd", "--corpus", "it-lim", "--years", "2016,2017", "--limit", "3")
    assert r.exit_code == 0, r.output
    reader = world.call("ingest")["reader"]
    assert isinstance(reader, cli.LimitedReader) and len(list(reader)) == 3
    assert world.call("hupd_reader")["years"] == [2016, 2017]


# --- embed-patents ------------------------------------------------------------------------------------------


def test_embed_encodes_pending_documents_and_is_resumable(world: World, settings: Settings) -> None:
    world.corpora["it-e"] = _corpus("it-e", "loading", 5)
    world.documents["it-e"], world.embedded["it-e"] = 5, 2  # an earlier run stopped after 2
    r = _invoke("embed-patents", "--corpus", "it-e", "--batch", "64", "--model-path", "D:/models/paecter")
    assert r.exit_code == 0, r.output
    assert world.call("embed_corpus")["page_size"] == 64
    assert world.encoder_loads == [{"local_path": Path("D:/models/paecter"), "model_id": PAECTER_MODEL_ID,
                                    "revision": PAECTER_REVISION, "device": "auto"}]
    out = json.loads(r.stdout)
    assert (out["documents"], out["embedded"], out["state"]) == (5, 5, "embedded")
    assert _manifest(settings, "embed", "it-e")["state"] == "embedded"
    # re-run: nothing pending, already embedded -> no model load, no encoding
    world.calls.clear()
    r = _invoke("embed-patents", "--corpus", "it-e")
    assert r.exit_code == 0, r.output
    assert world.names() == [] and len(world.encoder_loads) == 1


def test_embed_from_npy_imports_then_embeds_the_rest(world: World) -> None:
    world.corpora["it-n"] = _corpus("it-n", "loading", 5)
    world.documents["it-n"] = 5
    world.imported = 4  # one document's text differs from the vector's: skipped by the import
    r = _invoke("embed-patents", "--corpus", "it-n", "--from-npy", "D:/ipc_knn/emb/paecter", "--ids",
                "D:/ipc_knn/corpus.jsonl", "--batch", "2000")
    assert r.exit_code == 0, r.output
    assert world.names() == ["from_directory", "import_precomputed", "embed_corpus"]
    assert world.call("from_directory") == {"directory": Path("D:/ipc_knn/emb/paecter"),
                                            "id_list": Path("D:/ipc_knn/corpus.jsonl"), "with_texts": True}
    assert world.call("import_precomputed")["batch_size"] == 2000
    assert json.loads(r.stdout)["state"] == "embedded"


def test_embed_from_npy_complete_needs_no_model(world: World) -> None:
    world.corpora["it-n2"] = _corpus("it-n2", "loading", 5)
    world.documents["it-n2"] = 5
    r = _invoke("embed-patents", "--corpus", "it-n2", "--from-npy", "D:/x", "--ids", "D:/x/ids.jsonl")
    assert r.exit_code == 0, r.output
    assert "embed_corpus" not in world.names() and world.encoder_loads == []


def test_embed_finalizes_without_a_model_when_all_vectors_exist(world: World) -> None:
    world.corpora["it-f"] = _corpus("it-f", "loading", 3)
    world.documents["it-f"] = world.embedded["it-f"] = 3
    r = _invoke("embed-patents", "--corpus", "it-f")
    assert r.exit_code == 0, r.output
    assert world.names() == ["finalize_corpus"] and world.encoder_loads == []
    assert world.call("finalize_corpus")["policy"].model_revision == PAECTER_REVISION


def test_embed_from_npy_needs_ids(world: World) -> None:
    world.corpora["it-x"] = _corpus("it-x", "loading", 3)
    r = _invoke("embed-patents", "--corpus", "it-x", "--from-npy", "D:/x")
    assert r.exit_code == 1
    assert "--from-npy and --ids go together" in r.stderr


def test_embed_incomplete_exits_nonzero(world: World) -> None:
    world.corpora["it-i"] = _corpus("it-i", "loading", 5)
    world.documents["it-i"] = 5
    world.imported = 3
    world.fail["encoder"] = PaecterError("D:/models/paecter is not a SentenceTransformer directory")
    r = _invoke("embed-patents", "--corpus", "it-i", "--from-npy", "D:/x", "--ids", "D:/x/ids.jsonl")
    assert r.exit_code == 1
    assert "PaecterError: D:/models/paecter is not a SentenceTransformer directory" in r.stderr


def test_embed_without_a_local_copy_fails_instead_of_hanging(world: World, settings: Settings,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """No --model-path, no PATENT_ENCODER_LOCAL_PATH, not cached, the hub stalls: a bounded wait, one clear line."""
    import threading

    from app.runtime import load_patent_encoder

    release = threading.Event()

    def snapshot(repo: str, **kw: Any) -> str:
        if kw.get("local_files_only"):
            raise FileNotFoundError("not cached")
        release.wait(30)
        return "never"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    no_local = settings.model_copy(update={"patent_encoder_local_path": None, "patent_encoder_download_timeout_s": 0.2})
    monkeypatch.setattr(cli, "get_settings", lambda: no_local)
    assert cli._patent_encoder.__doc__ and load_patent_encoder.__module__ == "app.runtime"  # the real resolver runs
    world.corpora["it-h"] = _corpus("it-h", "loading", 5)
    world.documents["it-h"] = 5
    try:
        r = _invoke("embed-patents", "--corpus", "it-h")
    finally:
        release.set()
    assert r.exit_code == 1
    assert "PatentEncoderUnavailable" in r.stderr and "PATENT_ENCODER_LOCAL_PATH" in r.stderr
    assert "embed_corpus" not in world.names() and world.encoder_loads == []


# --- index / validate / activate / retire -------------------------------------------------------------------


def test_build_index_requires_m_and_ef_construction(world: World) -> None:
    r = _invoke("build-patent-index", "--corpus", "it-b")
    assert r.exit_code == 2 and "Missing option '--m'" in r.output


def test_build_index_passes_options(world: World, settings: Settings) -> None:
    world.corpora["it-b"] = _corpus("it-b", "embedded", 5)
    r = _invoke("build-patent-index", "--corpus", "it-b", "--m", "24", "--ef-construction", "200",
                "--maintenance-work-mem", "2GB", "--replace", "--concurrently")
    assert r.exit_code == 0, r.output
    assert world.call("build_patent_index") == {"corpus_id": "it-b", "m": 24, "ef_construction": 200,
                                                "maintenance_work_mem": "2GB", "replace": True, "concurrently": True}
    assert _manifest(settings, "index", "it-b")["status"] == "created"


def test_build_index_on_a_serving_corpus_fails_cleanly(world: World) -> None:
    world.corpora["it-s"] = _corpus("it-s", "validated", 5, serving=True)
    r = _invoke("build-patent-index", "--corpus", "it-s", *HNSW)
    assert r.exit_code == 1
    assert r.stderr.startswith("error: CorpusStateError: corpus 'it-s' is serving")
    assert "Traceback" not in r.output


def test_index_drop_lock_timeout_is_explained(world: World) -> None:
    import psycopg

    world.corpora["it-t"] = _corpus("it-t", "embedded", 5)
    world.fail["index"] = psycopg.errors.LockNotAvailable("canceling statement due to lock timeout")
    r = _invoke("build-patent-index", "--corpus", "it-t", *HNSW, "--replace")
    assert r.exit_code == 1
    assert "LockNotAvailable" in r.stderr and "nothing was dropped" in r.stderr


def test_validate_passes_and_fails(world: World, settings: Settings) -> None:
    world.corpora["it-v"] = _corpus("it-v", "embedded", 5)
    r = _invoke("validate-patent-corpus", "--corpus", "it-v", "--probes", "50", "--min-recall", "0.9")
    assert r.exit_code == 0, r.output
    assert world.call("validate_corpus") == {"corpus_id": "it-v", "probes": 50, "k": 100, "threshold": 0.9}
    world.validate_passes = False
    world.corpora["it-v"] = _corpus("it-v", "embedded", 5)
    r = _invoke("validate-patent-corpus", "--corpus", "it-v", "--k", "50")
    assert r.exit_code == 1
    assert "failed validation: recall@100 0.9 < threshold 0.95" in r.stderr
    assert world.calls[-1][1]["k"] == 50
    assert _manifest(settings, "validate", "it-v")["passed"] is False


def test_activate_takes_the_global_and_the_corpus_lock(world: World) -> None:
    world.corpora["it-a"] = _corpus("it-a", "validated", 5)
    r = _invoke("activate-patent-corpus", "--corpus", "it-a")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["is_serving"] is True
    assert world.locks == [("lock", cli.LOCK_NAMESPACE_GLOBAL, None), ("lock", cli.LOCK_NAMESPACE_CORPUS, "it-a"),
                           ("unlock", cli.LOCK_NAMESPACE_CORPUS, "it-a"), ("unlock", cli.LOCK_NAMESPACE_GLOBAL, None)]


def test_retire_passes_stop_serving(world: World) -> None:
    world.corpora["it-r"] = _corpus("it-r", "validated", 5, serving=True)
    r = _invoke("retire-patent-corpus", "--corpus", "it-r", "--stop-serving")
    assert r.exit_code == 0, r.output
    assert world.call("retire_corpus") == {"corpus_id": "it-r", "stop_serving": True}
    assert ("lock", cli.LOCK_NAMESPACE_GLOBAL, None) in world.locks


# --- advisory locks -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["ingest-patents", "--source", "hupd", "--corpus", "it-l", "--years", "2017"],
        ["embed-patents", "--corpus", "it-l"],
        ["build-patent-index", "--corpus", "it-l", *HNSW],
        ["validate-patent-corpus", "--corpus", "it-l"],
        ["activate-patent-corpus", "--corpus", "it-l"],
        ["retire-patent-corpus", "--corpus", "it-l"],
        ["load-patents", "--source", "hupd", "--corpus", "it-l", "--years", "2017", *HNSW],
    ],
    ids=["ingest", "embed", "index", "validate", "activate", "retire", "load"],
)
def test_a_second_command_on_a_locked_corpus_fails_fast(world: World, args: list[str]) -> None:
    world.corpora["it-l"] = _corpus("it-l", "embedded", 5)
    world.busy = {cli.LOCK_NAMESPACE_CORPUS}
    r = _invoke(*args)
    assert r.exit_code == 1
    assert "another ipc command holds the patent corpus 'it-l' lock" in r.stderr
    assert "Nothing was changed" in r.stderr
    assert world.calls == [] or world.names() == []


def test_activation_fails_fast_while_another_serving_change_runs(world: World) -> None:
    world.corpora["it-g"] = _corpus("it-g", "validated", 5)
    world.busy = {cli.LOCK_NAMESPACE_GLOBAL}
    r = _invoke("activate-patent-corpus", "--corpus", "it-g")
    assert r.exit_code == 1 and "patent serving (activation) lock" in r.stderr
    assert "activate_corpus" not in world.names()


def test_lock_is_released_when_the_command_fails(world: World) -> None:
    world.corpora["it-u"] = _corpus("it-u", "embedded", 5)
    world.fail["index"] = PatentIndexError("index is missing or invalid after the build")
    r = _invoke("build-patent-index", "--corpus", "it-u", *HNSW)
    assert r.exit_code == 1
    assert world.locks[-1] == ("unlock", cli.LOCK_NAMESPACE_CORPUS, "it-u")


# --- load-patents -------------------------------------------------------------------------------------------

LOAD = ["load-patents", "--source", "uspto-odp", "--years", "2021-2025", "--limit", "1000000", "--corpus", "it-load",
        *HNSW]  # fmt: skip
SERVICE_ORDER = ["uspto_reader", "ingest", "embed_corpus", "build_patent_index", "validate_corpus"]


def _services(world: World) -> list[str]:
    return [n for n in world.names() if n not in ("load_concordance", "hupd_reader")]


def test_load_runs_ingest_embed_index_validate_in_order(world: World, settings: Settings) -> None:
    r = _invoke(*LOAD)
    assert r.exit_code == 0, r.output
    assert _services(world) == SERVICE_ORDER
    assert "activate_corpus" not in world.names()  # activation stays a separate, deliberate step
    assert "UNVERIFIED" in r.stderr
    out = json.loads(r.stdout)
    assert [(s["step"], s["status"]) for s in out["steps"]] == [("ingest", "done"), ("embed", "done"),
                                                                ("index", "done"), ("validate", "done")]
    assert out["steps"][0]["concordance_applied"] is True
    assert out["next"] == "ipc activate-patent-corpus --corpus it-load"
    assert world.corpora["it-load"].state == "validated"
    for step in ("ingest", "embed", "index", "validate", "load"):
        assert (settings.manifests_dir / "patents" / f"{step}_it-load.json").is_file(), step
    # one lock for the whole run, released at the end
    assert world.locks == [("lock", cli.LOCK_NAMESPACE_CORPUS, "it-load"), ("unlock", cli.LOCK_NAMESPACE_CORPUS, "it-load")]
    assert world.call("validate_corpus")["k"] == 100


@pytest.mark.parametrize(
    ("step", "exc", "ran"),
    [
        ("ingest", HoldoutLeak("1 hold-out id(s) are in corpus", {}), ["uspto_reader", "ingest"]),
        ("embed", PaecterError("CUDA error: out of memory"), ["uspto_reader", "ingest", "embed_corpus"]),
        ("index", CorpusStateError("corpus is 'loading'"), SERVICE_ORDER[:4]),
        ("validate", PatentIndexError("probe failed"), SERVICE_ORDER),
    ],
)
def test_load_stops_at_the_first_failure(world: World, step: str, exc: BaseException, ran: list[str]) -> None:
    world.fail[step] = exc
    r = _invoke(*LOAD)
    assert r.exit_code == 1
    assert _services(world) == ran
    assert f"load-patents stopped at step '{step}': {type(exc).__name__}" in r.stderr
    assert "re-run the same command to resume" in r.stderr
    summary = json.loads(r.stdout)
    assert summary["steps"][-1]["step"] == step and summary["steps"][-1]["status"] == "failed"
    assert world.locks[-1] == ("unlock", cli.LOCK_NAMESPACE_CORPUS, "it-load")


def test_load_stops_when_validation_does_not_pass(world: World) -> None:
    world.validate_passes = False
    r = _invoke(*LOAD)
    assert r.exit_code == 1
    assert "stopped at step 'validate': recall@100 0.9 < threshold 0.95" in r.stderr
    assert "--ef-construction" in r.stderr


def test_load_resumes_after_a_failed_embed(world: World) -> None:
    world.fail["embed"] = PaecterError("CUDA error: out of memory")
    assert _invoke(*LOAD).exit_code == 1
    world.calls.clear()
    r = _invoke(*LOAD)  # same command again
    assert r.exit_code == 0, r.output
    # the completed ingest (same parameters, corpus unchanged) is not repeated: no second USPTO download
    assert _services(world) == ["embed_corpus", "build_patent_index", "validate_corpus"]
    steps = json.loads(r.stdout)["steps"]
    assert steps[0]["status"] == "skipped" and "completed ingest" in steps[0]["why"]


def test_load_reingests_when_parameters_change_or_asked(world: World) -> None:
    world.fail["embed"] = PaecterError("CUDA error: out of memory")
    assert _invoke(*LOAD).exit_code == 1
    world.calls.clear()
    other = [a if a != "1000000" else "500000" for a in LOAD]
    assert _invoke(*other).exit_code == 0
    assert "ingest" in world.names()
    world.calls.clear()
    world.corpora["it-load"] = _corpus("it-load", "loading", world.stored)
    assert _invoke(*other, "--reingest").exit_code == 0
    assert "ingest" in world.names()


def test_load_skips_ingest_and_embed_for_an_embedded_corpus(world: World) -> None:
    world.corpora["it-load"] = _corpus("it-load", "embedded", 5)
    world.documents["it-load"] = world.embedded["it-load"] = 5
    r = _invoke(*LOAD)
    assert r.exit_code == 0, r.output
    assert _services(world) == ["build_patent_index", "validate_corpus"]


def test_load_on_a_validated_corpus_does_nothing(world: World) -> None:
    world.corpora["it-load"] = _corpus("it-load", "validated", 5, serving=True)
    r = _invoke(*LOAD)
    assert r.exit_code == 0, r.output
    assert _services(world) == []
    assert all(s["status"] == "skipped" for s in json.loads(r.stdout)["steps"])


def test_load_hupd_with_precomputed_vectors(world: World, settings: Settings) -> None:
    """The TASK-0014 shape: HUPD 2017, seed 20261013, hold-out 2000, the offline vectors imported text-checked."""
    r = _invoke("load-patents", "--source", "hupd", "--years", "2017", "--corpus", "it-hupd2017", "--seed", "20261013",
                "--holdout", "2000", "--from-npy", "D:/ipc_knn/emb/paecter", "--ids", "D:/ipc_knn/corpus.jsonl", *HNSW)
    assert r.exit_code == 0, r.output
    assert "UNVERIFIED" not in r.stderr
    assert [n for n in world.names() if n != "load_concordance"] == [
        "hupd_reader", "ingest", "from_directory", "import_precomputed", "build_patent_index", "validate_corpus"]
    kw = world.call("ingest")
    assert (kw["seed"], kw["holdout_size"], kw["sample_size"]) == (20261013, 2000, None)
    assert world.call("from_directory")["with_texts"] is True
    assert world.encoder_loads == []


def test_load_rejects_from_npy_without_ids(world: World) -> None:
    r = _invoke(*LOAD, "--from-npy", "D:/x")
    assert r.exit_code == 2 and "--from-npy and --ids go together" in r.output
    assert world.calls == []


# --- patent-status ------------------------------------------------------------------------------------------


def test_patent_status_lists_corpora(world: World) -> None:
    world.corpora["it-a"] = _corpus("it-a", "validated", 5, serving=True)
    world.corpora["it-b"] = _corpus("it-b", "loading", 0)
    world.documents.update({"it-a": 5, "it-b": 7})
    world.embedded.update({"it-a": 5})
    r = _invoke("patent-status")
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    a, b = out["corpora"]
    assert (a["corpus_id"], a["state"], a["is_serving"], a["documents"], a["embeddings"]) == ("it-a", "validated",
                                                                                             True, 5, 5)
    assert a["encoder"] == {"model_id": PAECTER_MODEL_ID, "model_revision": PAECTER_REVISION}
    assert a["index"] == {"name": "patent_hnsw_it-a", "bytes": 3 * 2**20, "mib": 3.0, "valid": True}
    assert (b["state"], b["is_serving"], b["documents"], b["embeddings"]) == ("loading", False, 7, 0)


# --- calibration CLI: definition path only --------------------------------------------------------------------


def test_calibration_pipeline_is_definition_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app import runtime as runtime_mod
    from app.services.embeddings.providers import EncodingPolicy

    s = Settings(_env_file=None, calibration_dir=tmp_path, neighbour_vote_enabled=True)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    policy = EncodingPolicy(provider="fake", model_id="fake/bge", model_revision="r", dimension=8,
                            storage_table="ipc_embeddings", normalized=True, document_template="{text}",
                            query_template="{text}", taxonomy_max_tokens=512, query_chunk_tokens=1024,
                            dtype="float32")  # fmt: skip
    monkeypatch.setattr("app.services.embeddings.providers.get_provider", lambda st: SimpleNamespace(policy=policy))
    monkeypatch.setattr("app.services.classification.reranker.get_reranker",
                        lambda st: SimpleNamespace(model_id="fake/rr", revision="1"))
    fingerprinted: list[bool] = []

    def fingerprint(st: Settings, provider: Any, reranker: Any) -> tuple[dict, str]:
        fingerprinted.append(st.neighbour_vote_enabled)
        return {}, "fp"

    monkeypatch.setattr("app.services.confidence.fingerprint.current_fingerprint", fingerprint)

    def no_encoder(st: Settings) -> Any:
        raise AssertionError("the calibration CLI must not load PaECTER")

    monkeypatch.setattr(runtime_mod, "load_patent_encoder", no_encoder)
    rt = cli._load_pipeline()
    assert rt.definition_only is True
    assert rt.pipeline.s.neighbour_vote_enabled is False  # every classify takes the definition path
    assert rt.pipeline.patent_encoder is None
    assert fingerprinted == [True]  # ... but the fingerprint is the served (vote-enabled) pipeline's


def test_generate_pairs_classifies_on_the_definition_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    s = Settings(_env_file=None, calibration_dir=tmp_path, neighbour_vote_enabled=True)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    seen: list[bool] = []

    class FakePipeline:
        def __init__(self, settings: Settings) -> None:
            self.s = settings

        def classify(self, session: Any, req: Any) -> Any:
            seen.append(self.s.neighbour_vote_enabled)
            return SimpleNamespace(candidates=[], alternatives=[])

    def fake_load(self: Any) -> None:
        pipeline_settings = self.settings.model_copy(update={"neighbour_vote_enabled": False}) if \
            self.definition_only else self.settings  # fmt: skip
        self.pipeline, self.state, self.fingerprint, self.fingerprint_hash = FakePipeline(pipeline_settings), "ready", {}, "fp"

    class _Session:
        def begin(self) -> Any:
            return contextlib.nullcontext()

    monkeypatch.setattr("app.runtime.Runtime._load", fake_load)
    monkeypatch.setattr("app.db.session.api_session", lambda: contextlib.nullcontext(_Session()))
    doc = {"abstract": "A quantum key distribution network.", "taxonomy_version": VERSION, "requested_level": "group",
           "split": "fit", "labels": [], "judgement_completeness": "complete", "document_id": "d1"}
    monkeypatch.setattr("app.services.confidence.evaluation.load_eval_jsonl", lambda path: [doc])
    r = _invoke("generate-pairs", "--eval-file", str(tmp_path / "eval.jsonl"), "--out", str(tmp_path / "pairs.jsonl"))
    assert r.exit_code == 0, r.output
    assert seen == [False]


# --- integration: advisory locks on a real database (opt-in) --------------------------------------------------
# IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1 (or IPC_DB_TARGET=local). Takes only advisory locks on an
# it-test-* corpus id; writes no rows. Refuses when a real corpus is serving, loading or embedded (as the other
# patent integration tests do), and prints the patent table counts before and after.

TEST_PREFIX = "it-test-"
PATENT_TABLES = ("patent_corpora", "patent_documents", "patent_embeddings")


def _integration_target() -> str | None:
    target = os.environ.get("IPC_DB_TARGET")
    if target == "local" or (target == "sakthi" and os.environ.get("IPC_INTEGRATION_ALLOW_SAKTHI") == "1"):
        return target
    return None


def _counts(conn: Any) -> dict[str, int]:
    out = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in PATENT_TABLES}  # fixed names
    conn.rollback()
    return out


def _connect_guarded() -> Any:
    """A direct connection to the opted-in target, after refusing when a real corpus is serving or busy."""
    target = _integration_target()
    if target is None:
        pytest.skip("set IPC_DB_TARGET=local (or IPC_DB_TARGET=sakthi with IPC_INTEGRATION_ALLOW_SAKTHI=1)")
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().ipc_db_target == target
    from app.db.session import connect_direct

    conn = connect_direct()
    rows = conn.execute("SELECT corpus_id, state, is_serving FROM patent_corpora").fetchall()
    conn.rollback()
    busy = [(c, st, sv) for c, st, sv in rows if not c.startswith(TEST_PREFIX) and (sv or st in ("loading", "embedded"))]
    if busy:
        conn.close()
        pytest.fail(f"refusing to run: real corpus busy or serving: {busy}")
    return conn


@pytest.mark.integration
def test_advisory_lock_refuses_a_second_session() -> None:
    from app.db.session import connect_direct

    first = _connect_guarded()
    second = connect_direct()
    corpus_id = f"{TEST_PREFIX}lock-{uuid.uuid4().hex[:8]}"
    try:
        before = _counts(first)
        what = f"patent corpus {corpus_id!r}"
        with cli._advisory_lock(first, cli.LOCK_NAMESPACE_CORPUS, corpus_id, what):
            with (pytest.raises(cli.PatentLockBusy, match="another ipc command holds"),
                  cli._advisory_lock(second, cli.LOCK_NAMESPACE_CORPUS, corpus_id, what)):  # fmt: skip
                pass
            # another corpus is not blocked
            with cli._advisory_lock(second, cli.LOCK_NAMESPACE_CORPUS, corpus_id + "-other", "other"):
                pass
            # the lock survives the transaction ends cli.py issues (the connection is idle for the services)
            assert first.info.transaction_status.name == "IDLE"
        # released: the second session gets it now
        with cli._advisory_lock(second, cli.LOCK_NAMESPACE_CORPUS, corpus_id, f"patent corpus {corpus_id!r}"):
            pass
        after = _counts(first)
        print(f"\npatent table counts before={before} after={after}")
        assert before == after
    finally:
        first.close()
        second.close()


PAECTER_LOCAL = Path(os.environ.get("PATENT_ENCODER_LOCAL_PATH", "D:/ipc_knn/models/paecter"))


@pytest.mark.integration
def test_load_patents_end_to_end_on_a_tiny_hupd_corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real services behind load-patents: the first 400 HUPD 2017 records into an it-test corpus, with the
    WIPO concordance, PaECTER embeddings, a small partial HNSW index and a 20-probe validation. Needs the HUPD
    2017 tarball and a local PaECTER copy. Manifests go to tmp_path. The corpus and its index are removed after."""
    from psycopg import sql

    from app.config import get_settings
    from app.services.patents import index as ix

    conn = _connect_guarded()
    real = get_settings()
    if not (Path(real.hupd_data_dir) / "2017.tar.gz").is_file() or not (PAECTER_LOCAL / "modules.json").is_file():
        conn.close()
        pytest.skip("needs HUPD_DATA_DIR/2017.tar.gz and a local PaECTER copy")
    monkeypatch.setattr(cli, "get_settings", lambda: real.model_copy(update={"manifests_dir": tmp_path}))
    corpus_id = f"{TEST_PREFIX}cli-{uuid.uuid4().hex[:8]}"
    before = _counts(conn)
    args = ["load-patents", "--source", "hupd", "--years", "2017", "--limit", "400", "--corpus", corpus_id,
            "--m", "16", "--ef-construction", "64", "--probes", "20", "--batch", "128", "--model-path",
            str(PAECTER_LOCAL)]  # fmt: skip
    try:
        r = runner.invoke(cli.app, args)
        print("\n" + r.stdout)  # JSON, ASCII; stderr has tqdm bars a cp1252 console cannot print
        assert r.exit_code == 0, r.stderr.encode("ascii", "replace").decode()
        out = json.loads(r.stdout)
        assert [(s["step"], s["status"]) for s in out["steps"]] == [(n, "done") for n in cli.LOAD_STEPS]
        assert out["steps"][0]["concordance_applied"] is True
        ingest_manifest = json.loads((tmp_path / "patents" / f"ingest_{corpus_id}.json").read_text(encoding="utf-8"))
        assert ingest_manifest["concordance"]["sources"] and ingest_manifest["completed"] is True
        # re-run: everything is done, nothing is repeated
        again = json.loads(runner.invoke(cli.app, args).stdout)
        assert all(s["status"] == "skipped" for s in again["steps"])
        status = json.loads(runner.invoke(cli.app, ["patent-status"]).stdout)
        mine = next(c for c in status["corpora"] if c["corpus_id"] == corpus_id)
        print(json.dumps(mine, default=str))
        assert mine["state"] == "validated" and mine["is_serving"] is False
        assert mine["documents"] == mine["embeddings"] == mine["document_count"] >= 100
        assert mine["index"]["valid"] is True and mine["index"]["bytes"] > 0
    finally:
        conn.rollback()
        conn.autocommit = True
        conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(ix.index_name(corpus_id))))
        conn.execute("UPDATE patent_corpora SET is_serving = false WHERE corpus_id = %s AND corpus_id LIKE %s",
                     [corpus_id, TEST_PREFIX + "%"])
        conn.execute("DELETE FROM patent_corpora WHERE corpus_id = %s AND corpus_id LIKE %s",
                     [corpus_id, TEST_PREFIX + "%"])
        conn.autocommit = False
        left = conn.execute(
            "SELECT (SELECT count(*) FROM patent_corpora WHERE corpus_id = %(c)s)"
            " + (SELECT count(*) FROM patent_documents WHERE corpus_id = %(c)s)"
            " + (SELECT count(*) FROM patent_embeddings WHERE corpus_id = %(c)s)", {"c": corpus_id}).fetchone()[0]
        after = _counts(conn)
        conn.close()
        print(f"patent table counts before={before} after={after}")
        assert left == 0, f"{left} rows of {corpus_id} left behind"
