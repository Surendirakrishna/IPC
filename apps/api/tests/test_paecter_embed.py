"""TASK-0006: PaECTER encoder and the resumable patent embedding runner.

Unit tests use a fake model / encoder and an in-memory store: no model, no database.

Integration tests (DbEmbeddingStore) write to sakthi only with an explicit opt-in, on it-test-* corpora:
    IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1 uv run pytest -m integration tests/test_paecter_embed.py

The ``ml`` test loads the real model from the local copy (PAECTER_LOCAL_PATH, default D:/ipc_knn/models/paecter):
    uv run pytest -m ml tests/test_paecter_embed.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from app.db.patents import Corpus, CorpusStateError
from app.services.embeddings import paecter
from app.services.embeddings.paecter import (
    PAECTER_MODEL_ID,
    PAECTER_REVISION,
    EncoderMismatch,
    PaecterEncoder,
    PaecterError,
    PatentEncodingPolicy,
    patent_text,
)
from app.services.patents import embed
from app.services.patents.embed import (
    EmbedReport,
    PatentEmbedError,
    PendingDocument,
    PrecomputedSource,
    compute_input_hash,
    embed_corpus,
    import_precomputed,
    read_id_list,
    stored_input_digest,
    text_digest,
)

LOCAL_MODEL = Path(os.environ.get("PAECTER_LOCAL_PATH", "D:/ipc_knn/models/paecter"))
OFFLINE_EMB = Path("D:/ipc_knn/emb/paecter")
OFFLINE_CORPUS = Path("D:/ipc_knn/corpus.jsonl")


def _vec(text: str) -> np.ndarray:
    """Deterministic unit vector for a text (what the fake model/encoder 'embeds')."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


# --- encoder with a fake model -------------------------------------------------------------------------------


class FakeModel:
    """SentenceTransformer-shaped. Fails with a CUDA-style OOM when a batch is larger than oom_above."""

    def __init__(self, oom_above: int | None = None, error: BaseException | None = None) -> None:
        self.oom_above = oom_above
        self.error = error
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):  # noqa: ANN001
        assert normalize_embeddings and convert_to_numpy and not show_progress_bar
        assert batch_size == len(texts)
        if self.error is not None:
            raise self.error
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.calls.append(list(texts))
        return np.stack([_vec(t) * 1.001 for t in texts]).astype(np.float16)  # fp16, slightly off unit norm


def _texts(n: int) -> list[str]:
    return [f"Title {i}. " + "word " * ((i * 7) % 23 + 1) for i in range(n)]


def test_patent_text_is_title_dot_abstract() -> None:
    assert patent_text("Drill string", "Fatigue is managed.") == "Drill string. Fatigue is managed."
    assert patent_text(None, " Fatigue is managed. ") == "Fatigue is managed."
    assert patent_text("  ", "Abstract.") == "Abstract."


def test_encode_returns_normalised_float32_in_input_order() -> None:
    model = FakeModel()
    enc = PaecterEncoder(model, batch_size=8)
    texts = _texts(20)
    out = enc.encode(texts)
    assert out.dtype == np.float32 and out.shape == (20, 1024)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    for i, t in enumerate(texts):
        assert float(out[i] @ _vec(t)) > 0.999  # row i belongs to text i despite the length sort
    seen = [t for call in model.calls for t in call]
    assert [len(t) for t in seen] == sorted((len(t) for t in texts), reverse=True)  # longest first
    assert [len(c) for c in model.calls] == [8, 8, 4]


def test_encode_halves_the_batch_on_out_of_memory_and_keeps_it() -> None:
    model = FakeModel(oom_above=5)
    enc = PaecterEncoder(model, batch_size=32)
    texts = _texts(23)
    out = enc.encode(texts)
    assert [len(c) for c in model.calls] == [4, 4, 4, 4, 4, 3]  # 32 -> 16 -> 8 -> 4
    assert enc.batch_size == 4
    for i, t in enumerate(texts):
        assert float(out[i] @ _vec(t)) > 0.999


def test_encode_gives_up_at_batch_one_and_re_raises_other_errors() -> None:
    with pytest.raises(RuntimeError, match="out of memory"):
        PaecterEncoder(FakeModel(oom_above=0), batch_size=4).encode(_texts(3))
    model = FakeModel(error=ValueError("tokenizer exploded"))
    with pytest.raises(ValueError, match="tokenizer exploded"):
        PaecterEncoder(model, batch_size=4).encode(_texts(3))
    with pytest.raises(MemoryError):  # host OOM is retried too, then re-raised at batch 1
        PaecterEncoder(FakeModel(error=MemoryError()), batch_size=2).encode(_texts(3))


def test_encode_edge_cases() -> None:
    enc = PaecterEncoder(FakeModel())
    assert enc.encode([]).shape == (0, 1024)
    with pytest.raises(ValueError, match="text 1 is empty"):
        enc.encode(["fine", "  "])
    with pytest.raises(ValueError):
        PaecterEncoder(FakeModel(), batch_size=0)


def test_encoder_exposes_identity_and_policy() -> None:
    enc = PaecterEncoder(FakeModel(), dtype="float16")
    assert (enc.model_id, enc.revision, enc.dimension) == (PAECTER_MODEL_ID, PAECTER_REVISION, 1024)
    assert enc.policy == PatentEncodingPolicy(PAECTER_MODEL_ID, PAECTER_REVISION, 1024, 512, "mean", True,
                                              "title. abstract", "float16")  # fmt: skip
    assert "dtype" not in enc.policy.identity()  # precision does not change what a vector means


def _corpus(state: str = "loading", revision: str = PAECTER_REVISION, corpus_id: str = "c1", **kw) -> Corpus:
    now = datetime(2026, 10, 13, tzinfo=UTC)
    fields = dict(corpus_id=corpus_id, source="hupd", description=None, label_scheme="compact", label_ipc_edition=None,
                  labels_taxonomy_version=None, document_count=0, encoder_model_id=PAECTER_MODEL_ID,
                  encoder_model_revision=revision, input_hash=None, state=state, is_serving=False, created_at=now,
                  updated_at=now, served_at=None)  # fmt: skip
    fields.update(kw)
    return Corpus(**fields)  # type: ignore[arg-type]


def test_encoder_refuses_a_corpus_with_another_revision() -> None:
    enc = PaecterEncoder(FakeModel())
    enc.require_compatible(_corpus())
    with pytest.raises(EncoderMismatch, match="not comparable"):
        enc.require_compatible(_corpus(revision="0" * 40))
    with pytest.raises(EncoderMismatch):
        enc.require_compatible(_corpus(encoder_model_id="BAAI/bge-m3"))


def test_local_copy_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(PaecterError, match="modules.json"):
        paecter.check_local_copy(tmp_path, PAECTER_REVISION)
    (tmp_path / "modules.json").write_text("[]")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setitem(paecter.KNOWN_WEIGHTS_SHA256, "rev-a", hashlib.sha256(b"weights").hexdigest())
    paecter.check_local_copy(tmp_path, "rev-a")
    monkeypatch.setitem(paecter.KNOWN_WEIGHTS_SHA256, "rev-b", "0" * 64)
    with pytest.raises(EncoderMismatch, match="SHA-256"):
        paecter.check_local_copy(tmp_path, "rev-b")
    # A revision without a known hash is refused unless verification is skipped explicitly, which is logged.
    with pytest.raises(EncoderMismatch, match="no known weights SHA-256 for revision rev-unknown"):
        paecter.check_local_copy(tmp_path, "rev-unknown")
    caplog.set_level(logging.WARNING, logger="app.services.embeddings.paecter")
    assert "NOT verified" not in caplog.text
    paecter.check_local_copy(tmp_path, "rev-unknown", verify_weights=False)
    paecter.check_local_copy(tmp_path, "rev-b", verify_weights=False)  # also skips a known (wrong) hash
    assert caplog.text.count("weights NOT verified") == 2 and "rev-unknown" in caplog.text
    snap = tmp_path / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (snap / "modules.json").write_text("[]")
    with pytest.raises(EncoderMismatch, match="snapshot"):
        paecter.check_local_copy(snap, "b" * 40)


# --- runner with an in-memory store --------------------------------------------------------------------------


class MemoryStore:
    def __init__(self, corpus: Corpus, docs: dict[str, tuple[str | None, str]]) -> None:
        self._corpus = corpus
        self.docs = dict(docs)
        self.vectors: dict[str, np.ndarray] = {}
        self.write_counts: dict[str, int] = {}
        self.write_batches: list[int] = []

    def corpus(self, corpus_id: str) -> Corpus | None:
        return self._corpus if corpus_id == self._corpus.corpus_id else None

    def pending(self, corpus_id: str, after: str | None, limit: int) -> list[PendingDocument]:
        ids = sorted(d for d in self.docs if d not in self.vectors and (after is None or d > after))
        return [PendingDocument(d, *self.docs[d]) for d in ids[:limit]]

    def embedded_ids(self, corpus_id: str) -> set[str]:
        return set(self.vectors)

    def documents(self, corpus_id: str, document_ids):  # noqa: ANN001
        return {d: PendingDocument(d, *self.docs[d]) for d in document_ids if d in self.docs}

    def write(self, corpus_id: str, rows) -> int:  # noqa: ANN001
        if self._corpus.state == "validated":
            raise CorpusStateError("validated")
        for doc_id, v in rows:
            assert doc_id in self.docs
            self.vectors[doc_id] = np.asarray(v, dtype=np.float16)
            self.write_counts[doc_id] = self.write_counts.get(doc_id, 0) + 1
        self.write_batches.append(len(rows))
        return len(rows)

    def counts(self, corpus_id: str) -> tuple[int, int]:
        return len(self.docs), len(self.vectors)

    def input_rows(self, corpus_id: str):  # noqa: ANN201
        for d in sorted(self.docs):
            yield d, stored_input_digest(*self.docs[d])

    def set_embedded(self, corpus_id: str, document_count: int, input_hash: str) -> Corpus:
        self._corpus = dataclasses.replace(self._corpus, state="embedded", document_count=document_count,
                                           input_hash=input_hash)  # fmt: skip
        return self._corpus


class Killed(BaseException):
    """Stands in for a process kill / Ctrl-C between two batches."""


class FakeEncoder:
    def __init__(self, revision: str = PAECTER_REVISION, fail_after_calls: int | None = None) -> None:
        self.model_id = PAECTER_MODEL_ID
        self.revision = revision
        self.policy = PatentEncodingPolicy(PAECTER_MODEL_ID, revision)
        self.fail_after_calls = fail_after_calls
        self.seen: list[str] = []
        self.calls = 0

    def encode(self, texts):  # noqa: ANN001
        if self.fail_after_calls is not None and self.calls >= self.fail_after_calls:
            raise Killed("killed")  # what a killed / interrupted run looks like mid-way
        self.calls += 1
        self.seen.extend(texts)
        return np.stack([_vec(t) for t in texts])


def _docs(n: int, prefix: str = "d") -> dict[str, tuple[str | None, str]]:
    return {f"{prefix}{i:03d}": (None if i % 5 == 0 else f"Title {i}", f"Abstract number {i} about a device.")
            for i in range(n)}  # fmt: skip


def test_runner_embeds_every_document_in_batches_and_marks_embedded() -> None:
    store = MemoryStore(_corpus(), _docs(45))
    enc = FakeEncoder()
    progress: list[int] = []
    report = embed_corpus(store, "c1", enc, page_size=20, progress=lambda r: progress.append(r.embedded_now))
    assert report.batches == [20, 20, 5] and store.write_batches == [20, 20, 5] and progress == [20, 40, 45]
    assert set(store.vectors) == set(store.docs) and all(c == 1 for c in store.write_counts.values())
    assert report.state == "embedded" and store.corpus("c1").state == "embedded"
    want, n = compute_input_hash(enc.policy, ((d, stored_input_digest(*store.docs[d])) for d in sorted(store.docs)))
    assert report.input_hash == want and n == 45 and store.corpus("c1").document_count == 45
    # the text embedded is "title. abstract" (abstract alone without a title)
    assert patent_text(*store.docs["d001"]) in enc.seen and "Abstract number 0 about a device." in enc.seen
    np.testing.assert_allclose(store.vectors["d001"].astype(np.float32),
                               _vec(patent_text(*store.docs["d001"])), atol=1e-3)  # fmt: skip


def test_a_killed_run_resumes_without_duplicates() -> None:
    store = MemoryStore(_corpus(), _docs(50))
    first = FakeEncoder(fail_after_calls=2)
    with pytest.raises(Killed):
        embed_corpus(store, "c1", first, page_size=15)
    assert len(store.vectors) == 30 and store.corpus("c1").state == "loading"  # two committed pages survive
    second = FakeEncoder()
    report = embed_corpus(store, "c1", second, page_size=15)
    assert report.embedded_before == 30 and report.embedded_now == 20 and report.batches == [15, 5]
    assert not set(first.seen) & set(second.seen)  # nothing embedded twice
    assert all(c == 1 for c in store.write_counts.values()) and len(store.vectors) == 50
    assert report.state == "embedded"
    again = embed_corpus(store, "c1", FakeEncoder(), page_size=15)  # a completed corpus: nothing to do
    assert again.embedded_now == 0 and again.state == "embedded" and again.input_hash == report.input_hash


def test_new_documents_after_embedding_are_embedded_and_change_the_input_hash() -> None:
    store = MemoryStore(_corpus(), _docs(10))
    first = embed_corpus(store, "c1", FakeEncoder(), page_size=4)
    store.docs.update(_docs(3, prefix="e"))
    second = embed_corpus(store, "c1", FakeEncoder(), page_size=4)
    assert second.embedded_now == 3 and second.state == "embedded" and second.input_hash != first.input_hash


def test_trial_run_with_max_documents_does_not_finalize() -> None:
    store = MemoryStore(_corpus(), _docs(30))
    report = embed_corpus(store, "c1", FakeEncoder(), page_size=8, max_documents=10)
    assert report.embedded_now == 10 and report.batches == [8, 2]
    assert report.state == "loading" and report.input_hash is None


def test_runner_refuses_a_mismatched_encoder_or_a_validated_corpus() -> None:
    store = MemoryStore(_corpus(revision="0" * 40), _docs(5))
    enc = FakeEncoder()
    with pytest.raises(EncoderMismatch):
        embed_corpus(store, "c1", enc)
    assert enc.calls == 0 and store.vectors == {}
    with pytest.raises(CorpusStateError, match="validated"):
        embed_corpus(MemoryStore(_corpus(state="validated"), _docs(5)), "c1", FakeEncoder())
    with pytest.raises(embed.CorpusNotFound):
        embed_corpus(MemoryStore(_corpus(), _docs(5)), "nope", FakeEncoder())


def test_runner_rejects_unnormalised_encoder_output() -> None:
    class Loud(FakeEncoder):
        def encode(self, texts):  # noqa: ANN001
            return super().encode(texts) * 3

    store = MemoryStore(_corpus(), _docs(5))
    with pytest.raises(PaecterError, match="not L2-normalised"):
        embed_corpus(store, "c1", Loud())
    assert store.vectors == {}


def test_empty_corpus_is_not_marked_embedded() -> None:
    report = embed_corpus(MemoryStore(_corpus(), {}), "c1", FakeEncoder())
    assert report.state == "loading" and report.documents == 0


# --- precomputed vectors -------------------------------------------------------------------------------------


def _write_offline(tmp_path: Path, rows: list[tuple[str, str | None, str]], shard: int = 4, **meta_over) -> Path:
    d = tmp_path / "emb"
    d.mkdir()
    vecs = np.stack([_vec(patent_text(t, a)) for _, t, a in rows]).astype(np.float16)
    for i in range(0, len(rows), shard):
        np.save(d / f"corpus_{i // shard:04d}.npy", vecs[i:i + shard])
    meta = {"model_id": PAECTER_MODEL_ID, "revision": PAECTER_REVISION, "max_seq_length": 512, "dtype": "float16",
            "text": "title. abstract"} | meta_over  # fmt: skip
    (d / "meta.json").write_text(json.dumps(meta))
    with (tmp_path / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for doc_id, t, a in rows:
            fh.write(json.dumps({"id": doc_id, "title": t, "abstract": a, "labels": ["H04L908"]}) + "\n")
    return d


def test_import_precomputed_reuses_matching_vectors_and_skips_the_rest(tmp_path: Path) -> None:
    docs = _docs(10)
    offline = [(d, t, a) for d, (t, a) in docs.items()]
    offline[3] = (offline[3][0], offline[3][1], "A different abstract than the one stored.")  # text changed since
    offline.append(("x-not-in-corpus", "T", "Held out."))
    del docs["d009"]  # corpus is smaller than the offline id list ...
    docs["d100"] = ("New", "A document the offline run never saw.")  # ... and has one the offline run lacks
    store = MemoryStore(_corpus(), docs)
    src = PrecomputedSource.from_directory(_write_offline(tmp_path, offline), tmp_path / "corpus.jsonl")
    assert len(src.shards) == 3 and src.text_digests is not None

    report = import_precomputed(store, "c1", src, batch_size=4)
    assert report.imported == 8  # 9 offline rows in the corpus, minus d003 whose text changed
    assert report.text_checked is True
    assert report.skipped_text_mismatch == 1 and report.skipped_unknown_id == 2  # d009 and x-not-in-corpus
    assert report.state == "loading"  # d003 and d100 still need vectors
    np.testing.assert_array_equal(store.vectors["d001"], _vec(patent_text(*docs["d001"])).astype(np.float16))

    resumed = import_precomputed(store, "c1", src, batch_size=4)  # an interrupted import resumes
    assert resumed.imported == 0 and resumed.skipped_existing == 8
    enc = FakeEncoder()
    final = embed_corpus(store, "c1", enc, page_size=10)
    assert sorted(enc.seen) == sorted([patent_text(*docs["d003"]), patent_text(*docs["d100"])])
    assert final.state == "embedded" and len(store.vectors) == len(docs)


@pytest.mark.parametrize(("meta", "message"), [
    ({"revision": "0" * 40}, "revision"), ({"model_id": "BAAI/bge-m3"}, "model_id"),
    ({"text": "abstract"}, "text template"), ({"max_seq_length": 256}, "max_seq_length"),
])  # fmt: skip
def test_import_refuses_vectors_from_another_model_or_template(tmp_path: Path, meta: dict, message: str) -> None:
    docs = _docs(4)
    d = _write_offline(tmp_path, [(k, t, a) for k, (t, a) in docs.items()], **meta)
    store = MemoryStore(_corpus(), docs)
    with pytest.raises(PatentEmbedError, match=message):
        import_precomputed(store, "c1", PrecomputedSource.from_directory(d, tmp_path / "corpus.jsonl"))
    assert store.vectors == {}


def test_import_checks_shapes_and_ids(tmp_path: Path) -> None:
    docs = _docs(4)
    d = _write_offline(tmp_path, [(k, t, a) for k, (t, a) in docs.items()])
    store = MemoryStore(_corpus(), docs)
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps(list(docs)[:3]))
    with pytest.raises(PatentEmbedError, match="4 rows but the id list has 3"):
        import_precomputed(store, "c1", PrecomputedSource.from_directory(d, ids_file))
    ids_file.write_text(json.dumps(["d000", "d000", "d001", "d002"]))
    with pytest.raises(PatentEmbedError, match="duplicate"):
        import_precomputed(store, "c1", PrecomputedSource.from_directory(d, ids_file))
    (d / "meta.json").unlink()
    with pytest.raises(PatentEmbedError, match="meta.json"):
        PrecomputedSource.from_directory(d, ids_file)


def test_import_without_texts_and_id_list_formats(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "a.json").write_text('["1", 2]')
    assert read_id_list(tmp_path / "a.json") == (["1", "2"], None)
    (tmp_path / "b.txt").write_text("10\n 11 \n\n")
    assert read_id_list(tmp_path / "b.txt") == (["10", "11"], None)
    (tmp_path / "c.jsonl").write_text('{"id": "5", "title": null, "abstract": "x"}\n{"id": 6, "abstract": "y"}\n')
    assert read_id_list(tmp_path / "c.jsonl") == (["5", "6"], [text_digest(None, "x"), text_digest("", "y")])
    assert read_id_list(tmp_path / "c.jsonl", with_texts=False) == (["5", "6"], None)

    docs = _docs(4)
    d = _write_offline(tmp_path, [(k, t, a) for k, (t, a) in docs.items()])
    (tmp_path / "ids.txt").write_text("\n".join(docs))
    store = MemoryStore(_corpus(), docs)
    caplog.set_level(logging.WARNING, logger="app.services.patents.embed")
    report = import_precomputed(store, "c1", PrecomputedSource.from_directory(d, tmp_path / "ids.txt"))
    assert report.imported == 4 and report.state == "embedded"
    assert report.text_checked is False and report.as_dict()["text_checked"] is False  # a plain id list: no texts
    assert "WITHOUT a text check" in caplog.text


def test_import_rejects_unnormalised_shards(tmp_path: Path) -> None:
    docs = _docs(2)
    d = _write_offline(tmp_path, [(k, t, a) for k, (t, a) in docs.items()], shard=2)
    np.save(d / "corpus_0000.npy", np.ones((2, 1024), dtype=np.float16))
    with pytest.raises(PatentEmbedError, match="not L2-normalised"):
        import_precomputed(MemoryStore(_corpus(), docs), "c1", PrecomputedSource.from_directory(d, tmp_path / "corpus.jsonl"))


def test_report_as_dict() -> None:
    r = EmbedReport("c1", batches=[1, 2])
    assert r.as_dict()["batch_count"] == 2 and "batches" not in r.as_dict()


# --- ml: the real model ----------------------------------------------------------------------------------------


@pytest.mark.ml
def test_paecter_encodes_deterministically_at_1024_dims_and_matches_the_offline_vectors() -> None:
    if not (LOCAL_MODEL / "modules.json").is_file():
        pytest.skip(f"PaECTER local copy not found at {LOCAL_MODEL}")
    enc = PaecterEncoder.load(local_path=LOCAL_MODEL)  # verifies the weights' SHA-256 for the pinned revision
    try:
        assert enc.dimension == 1024 and enc.model.max_seq_length == 512
        assert enc.policy.model_revision == PAECTER_REVISION and enc.source.startswith("local:")
        text = patent_text("Quantum key distribution for smart grids",
                           "Quantum keys are distributed between grid nodes over authenticated channels.")
        a = enc.encode([text])
        b = enc.encode([text])
        assert a.shape == (1, 1024) and a.dtype == np.float32
        np.testing.assert_array_equal(a, b)
        assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5
        # truncation at 512 tokens: words past the limit do not change the vector
        long = "semiconductor wafer " * 400
        c = enc.encode([long, long + " completely different trailing words about agriculture"])
        assert float(c[0] @ c[1]) > 0.9999
        # same model, template and precision as the offline run that produced D:/ipc_knn/emb/paecter
        if OFFLINE_CORPUS.is_file() and (OFFLINE_EMB / "corpus_0000.npy").is_file():
            with OFFLINE_CORPUS.open(encoding="utf-8") as fh:
                first = json.loads(fh.readline())
            offline = np.load(OFFLINE_EMB / "corpus_0000.npy", mmap_mode="r")[0].astype(np.float32)
            mine = enc.encode([patent_text(first["title"], first["abstract"])])[0]
            assert float(mine @ offline) > 0.999
        with pytest.raises(EncoderMismatch):
            enc.require_compatible(_corpus(revision="0" * 40))
    finally:
        del enc
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- integration: DbEmbeddingStore on sakthi (opt-in) or a local database --------------------------------------

TEST_PREFIX = "it-test-"


def _integration_target() -> str | None:
    target = os.environ.get("IPC_DB_TARGET")
    if target == "local":
        return target
    if target == "sakthi" and os.environ.get("IPC_INTEGRATION_ALLOW_SAKTHI") == "1":
        return target
    return None


BUSY_STATES = ("loading", "embedded")


def _refusal(corpora: list[tuple[str, str, bool]]) -> str | None:
    """Why the integration tests must not run against this database, or None.

    They create/drop an index and ANALYZE the shared patent_embeddings table, which would block or disturb a
    live load. So they refuse when any real (non-it-test) corpus is serving, or is loading or embedding.
    """
    for corpus_id, state, is_serving in corpora:
        if corpus_id.startswith(TEST_PREFIX):
            continue
        if is_serving:
            return f"refusing to run: corpus {corpus_id!r} is serving"
        if state in BUSY_STATES:
            return f"refusing to run: corpus {corpus_id!r} is {state!r} (a load may be running)"
    return None


def test_integration_refuses_when_a_real_corpus_is_serving_or_loading() -> None:
    assert _refusal([]) is None
    assert _refusal([("it-test-a", "loading", False), ("it-test-b", "validated", True)]) is None
    assert _refusal([("hupd-2017", "validated", False), ("old", "failed", False)]) is None
    assert "serving" in _refusal([("hupd-2017", "validated", True)])  # type: ignore[operator]
    assert "'loading'" in _refusal([("uspto-2021", "loading", False)])  # type: ignore[operator]
    assert "'embedded'" in _refusal([("uspto-2021", "embedded", False)])  # type: ignore[operator]


@pytest.fixture
def db_conn():
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
    created: list[str] = []
    try:
        yield conn, created
    finally:
        assert all(c.startswith(TEST_PREFIX) for c in created), created
        conn.rollback()
        conn.execute("DELETE FROM patent_corpora WHERE corpus_id = ANY(%s) AND corpus_id LIKE %s AND NOT is_serving",
                     [created, TEST_PREFIX + "%"])
        conn.commit()
        conn.close()


def _db_corpus(conn, created: list[str], docs: dict[str, tuple[str | None, str]]) -> str:  # noqa: ANN001
    from app.db import patents
    from app.db.patents import CorpusSpec, PatentDocument

    corpus_id = f"{TEST_PREFIX}{uuid.uuid4().hex[:10]}"
    created.append(corpus_id)
    patents.create_or_get_corpus(conn, CorpusSpec(corpus_id, "hupd", "compact", PAECTER_MODEL_ID, PAECTER_REVISION))
    patents.upsert_documents(conn, corpus_id, [
        PatentDocument(d, t, a, date(2017, 1, 1), None, ["H04L908"], [("H04L", "H04L 9/00", "H04L 9/08")])
        for d, (t, a) in docs.items()])  # fmt: skip
    return corpus_id


@pytest.mark.integration
def test_db_runner_resumes_and_marks_embedded(db_conn) -> None:
    from app.db import patents
    from app.services.patents.embed import DbEmbeddingStore

    conn, created = db_conn
    docs = _docs(40)
    docs["d001"] = ("Übertragung — 量子鍵", "Ünïcödé abstract with an em dash — and CJK 量子.")  # md5 over UTF-8
    corpus_id = _db_corpus(conn, created, docs)
    store = DbEmbeddingStore(conn)
    first = FakeEncoder(fail_after_calls=1)
    with pytest.raises(Killed):
        embed_corpus(store, corpus_id, first, page_size=16)
    assert store.counts(corpus_id) == (40, 16)
    assert patents.get_corpus(conn, corpus_id).state == "loading"  # type: ignore[union-attr]
    conn.rollback()
    second = FakeEncoder()
    report = embed_corpus(store, corpus_id, second, page_size=16)
    assert report.embedded_before == 16 and report.batches == [16, 8] and not set(first.seen) & set(second.seen)
    corpus = patents.get_corpus(conn, corpus_id)
    conn.rollback()
    assert corpus.state == "embedded" and corpus.document_count == 40  # type: ignore[union-attr]
    # the server-side md5 digests equal the Python ones, so the hash does not depend on the store
    memory = MemoryStore(_corpus(corpus_id=corpus_id), docs)
    want, _ = compute_input_hash(second.policy, memory.input_rows(corpus_id))
    assert corpus.input_hash == report.input_hash == want  # type: ignore[union-attr]
    rows = conn.execute("SELECT count(*), count(DISTINCT document_id) FROM patent_embeddings WHERE corpus_id = %s",
                        [corpus_id]).fetchone()  # fmt: skip
    conn.rollback()
    assert rows == (40, 40)
    stored = conn.execute("SELECT embedding::text FROM patent_embeddings WHERE corpus_id = %s AND document_id = 'd002'",
                          [corpus_id]).fetchone()[0]  # fmt: skip
    conn.rollback()
    np.testing.assert_allclose(np.array(json.loads(stored), dtype=np.float32), _vec(patent_text(*docs["d002"])), atol=1e-3)


@pytest.mark.integration
def test_db_import_precomputed(db_conn, tmp_path: Path) -> None:
    from app.services.patents.embed import DbEmbeddingStore

    conn, created = db_conn
    docs = _docs(9)
    corpus_id = _db_corpus(conn, created, docs)
    d = _write_offline(tmp_path, [(k, t, a) for k, (t, a) in docs.items()])
    report = import_precomputed(DbEmbeddingStore(conn), corpus_id,
                                PrecomputedSource.from_directory(d, tmp_path / "corpus.jsonl"), batch_size=4)  # fmt: skip
    assert report.imported == 9 and report.batches == [4, 4, 1] and report.state == "embedded"
    with pytest.raises(PatentEmbedError, match="autocommit"):
        conn.autocommit = True
        try:
            DbEmbeddingStore(conn)
        finally:
            conn.autocommit = False
