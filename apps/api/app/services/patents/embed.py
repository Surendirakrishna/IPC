"""Fill a patent corpus's vectors with PaECTER (ADR-0005), resumably.

``embed_corpus`` embeds every document of a corpus that has no vector yet, in pages ordered by
document id. Each page is encoded and written as one committed batch (``db.patents.insert_embeddings``,
halfvec(1024)), so a killed run loses at most one page and the next run continues with the documents
still missing a vector: nothing is embedded twice and nothing is duplicated (the table's key is
(corpus_id, document_id) anyway).

``import_precomputed`` reuses vectors computed offline (``.npy`` shards plus an id list, e.g. the 294k
PaECTER vectors in ``D:/ipc_knn/emb/paecter``) instead of re-embedding. It refuses unless the shards'
recorded model, revision, token limit and text template match the corpus, and it skips any document
whose text differs from the text the vector was computed from (when the id list carries the texts).

Both end with ``finalize_corpus``: when every document has a vector, it records the input hash (a hash
of the encoding policy and of every document's id, title and abstract) and sets the state to
``embedded``. The model id and revision are part of the corpus row from creation
(``create_or_get_corpus``); the runner refuses an encoder or shards that do not match them.

Storage goes through ``EmbeddingStore``: ``DbEmbeddingStore`` on a psycopg connection in production, an
in-memory store in unit tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from psycopg import Connection

from app.db import patents
from app.db.patents import Corpus, CorpusNotFound, CorpusStateError
from app.services.embeddings.paecter import (
    DIMENSION,
    MAX_TOKENS,
    TEXT_TEMPLATE,
    PaecterError,
    PatentEncodingPolicy,
    check_normalised,
    patent_text,
    require_same_encoder,
)

log = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 1_000
IMPORT_NORM_TOLERANCE = 1e-2  # float16 storage of a unit vector


class PatentEmbedError(RuntimeError):
    pass


class PatentEncoder(Protocol):
    model_id: str
    revision: str
    policy: PatentEncodingPolicy

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class PendingDocument:
    document_id: str
    title: str | None
    abstract: str


class EmbeddingStore(Protocol):
    def corpus(self, corpus_id: str) -> Corpus | None: ...

    def pending(self, corpus_id: str, after: str | None, limit: int) -> list[PendingDocument]:
        """Documents without a vector, document_id > after, ordered by document_id."""
        ...

    def embedded_ids(self, corpus_id: str) -> set[str]: ...

    def documents(self, corpus_id: str, document_ids: Sequence[str]) -> dict[str, PendingDocument]: ...

    def write(self, corpus_id: str, rows: Sequence[tuple[str, np.ndarray]]) -> int:
        """Store vectors as one committed batch; returns the number written."""
        ...

    def counts(self, corpus_id: str) -> tuple[int, int]:
        """(documents, documents with a vector)."""
        ...

    def input_rows(self, corpus_id: str) -> Iterator[tuple[str, str]]:
        """(document_id, ``stored_input_digest(title, abstract)``) for every document, by document_id in
        code point order."""
        ...

    def set_embedded(self, corpus_id: str, document_count: int, input_hash: str) -> Corpus: ...


@dataclass
class EmbedReport:
    corpus_id: str
    documents: int = 0
    embedded_before: int = 0
    embedded_now: int = 0
    imported: int = 0
    skipped_existing: int = 0
    skipped_unknown_id: int = 0
    skipped_text_mismatch: int = 0
    # False only for an import whose id list carried no texts: its vectors were stored without checking that
    # the stored documents still have the text they were computed from.
    text_checked: bool = True
    state: str | None = None
    input_hash: str | None = None
    batches: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "batches"} | {"batch_count": len(self.batches)}


def _writable_corpus(store: EmbeddingStore, corpus_id: str) -> Corpus:
    corpus = store.corpus(corpus_id)
    if corpus is None:
        raise CorpusNotFound(f"corpus {corpus_id!r} not found")
    if corpus.state == "validated":
        raise CorpusStateError(f"corpus {corpus_id!r} is validated; its vectors are read-only")
    return corpus


def stored_input_digest(title: str | None, abstract: str) -> str:
    """MD5 of a document's stored inputs, ``coalesce(title, '') || U+001F || abstract``. The database store
    computes the same value server-side (``md5()`` over UTF-8 text), so only digests cross the network."""
    return hashlib.md5(f"{title or ''}\x1f{abstract}".encode(), usedforsecurity=False).hexdigest()


def text_digest(title: str | None, abstract: str) -> str:
    """MD5 of the text PaECTER embeds for a document (``patent_text``): equal digests, same input."""
    return hashlib.md5(patent_text(title, abstract).encode("utf-8"), usedforsecurity=False).hexdigest()


def compute_input_hash(policy: PatentEncodingPolicy, rows: Iterable[tuple[str, str]]) -> tuple[str, int]:
    """SHA-256 over the encoding policy and every (document_id, input digest), in document_id order."""
    h = hashlib.sha256(policy.identity_json().encode("utf-8"))
    n = 0
    for doc_id, digest in rows:
        h.update(f"\n{doc_id}\t{digest}".encode())
        n += 1
    return h.hexdigest(), n


def finalize_corpus(store: EmbeddingStore, corpus_id: str, policy: PatentEncodingPolicy,
                    report: EmbedReport | None = None) -> EmbedReport:  # fmt: skip
    """Record the input hash and set ``embedded`` when every document has a vector; otherwise leave the state."""
    report = report or EmbedReport(corpus_id)
    corpus = _writable_corpus(store, corpus_id)
    documents, embedded = store.counts(corpus_id)
    report.documents = documents
    if documents == 0 or embedded != documents:
        report.state = corpus.state
        log.info("corpus %s: %d of %d documents have a vector; state stays %s", corpus_id, embedded, documents,
                 corpus.state)  # fmt: skip
        return report
    input_hash, n = compute_input_hash(policy, store.input_rows(corpus_id))
    if n != documents:  # documents were added while hashing; the next run will finalize
        report.state = corpus.state
        return report
    updated = store.set_embedded(corpus_id, documents, input_hash)
    report.state, report.input_hash = updated.state, updated.input_hash
    return report


def embed_corpus(
    store: EmbeddingStore,
    corpus_id: str,
    encoder: PatentEncoder,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_documents: int | None = None,
    progress: Callable[[EmbedReport], None] | None = None,
) -> EmbedReport:
    """Embed every document of ``corpus_id`` that has no vector yet, then finalize the corpus.

    ``page_size`` documents are read, encoded and written per committed batch. ``max_documents`` stops
    early (for a trial run); the corpus is then finalized only if nothing is missing.
    """
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    corpus = _writable_corpus(store, corpus_id)
    require_same_encoder(encoder.model_id, encoder.revision, corpus_id=corpus_id,
                         corpus_model_id=corpus.encoder_model_id, corpus_revision=corpus.encoder_model_revision)  # fmt: skip
    if encoder.policy.dimension != DIMENSION:
        raise PatentEmbedError(f"encoder dimension {encoder.policy.dimension} != {DIMENSION}")
    report = EmbedReport(corpus_id)
    report.documents, report.embedded_before = store.counts(corpus_id)
    after: str | None = None
    while max_documents is None or report.embedded_now < max_documents:
        limit = page_size if max_documents is None else min(page_size, max_documents - report.embedded_now)
        page = store.pending(corpus_id, after, limit)
        if not page:
            break
        vectors = encoder.encode([patent_text(d.title, d.abstract) for d in page])
        if vectors.shape != (len(page), DIMENSION):
            raise PatentEmbedError(f"encoder returned {vectors.shape} for {len(page)} documents")
        check_normalised(vectors)
        store.write(corpus_id, [(d.document_id, v) for d, v in zip(page, vectors, strict=True)])
        report.embedded_now += len(page)
        report.batches.append(len(page))
        after = page[-1].document_id
        if progress is not None:
            progress(report)
    return finalize_corpus(store, corpus_id, encoder.policy, report)


# --- precomputed vectors -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecomputedSource:
    """Offline vectors: ``.npy`` shards (row order = id order) plus their ids and recorded metadata.

    ``text_digests`` (optional, same order as ``ids``) are ``text_digest`` values of the texts the vectors
    were computed from (digests, not texts: 294k abstracts would cost several hundred MB of memory); when
    given, a document whose stored text differs is skipped rather than given a stale vector.
    """

    shards: tuple[Path, ...]
    ids: tuple[str, ...]
    meta: dict[str, Any]
    text_digests: tuple[str, ...] | None = None

    @classmethod
    def from_directory(cls, directory: str | Path, id_list: str | Path, *, pattern: str = "corpus_*.npy",
                       with_texts: bool = True) -> PrecomputedSource:  # fmt: skip
        """Shards ``directory/pattern`` (sorted by name) and ``directory/meta.json``.

        ``id_list`` is a JSON array of ids, a text file with one id per line, or a JSONL file whose rows
        have ``id`` (and ``title`` / ``abstract``, kept as text digests when ``with_texts``), e.g. the
        offline ``corpus.jsonl``.
        """
        d = Path(directory)
        shards = tuple(sorted(d.glob(pattern)))
        if not shards:
            raise PatentEmbedError(f"no shards matching {pattern!r} in {d}")
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            raise PatentEmbedError(f"{meta_path} not found: precomputed vectors need their recorded model and template")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids, digests = read_id_list(Path(id_list), with_texts=with_texts)
        return cls(shards, tuple(ids), meta, tuple(digests) if digests is not None else None)


def read_id_list(path: Path, *, with_texts: bool = True) -> tuple[list[str], list[str] | None]:
    """Ids, and ``text_digest`` per id when the file is JSONL with title / abstract (and ``with_texts``)."""
    with path.open(encoding="utf-8") as fh:  # streamed: corpus.jsonl is 260 MB
        first = ""
        for first in fh:
            if first.strip():
                break
        if first.lstrip().startswith("["):
            return [str(x) for x in json.loads(first + fh.read())], None
        if not first.lstrip().startswith("{"):
            return [x.strip() for x in [first, *fh] if x.strip()], None
        ids: list[str] = []
        digests: list[str] = []
        has_text = with_texts
        for line in _chain(first, fh):
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row["id"]))
            if has_text and "abstract" in row:
                digests.append(text_digest(row.get("title"), row["abstract"]))
            else:
                has_text = False
        return ids, (digests if has_text else None)


def _chain(first: str, rest: Iterable[str]) -> Iterator[str]:
    yield first
    yield from rest


def check_precomputed_meta(meta: dict[str, Any], corpus: Corpus) -> None:
    """Refuse shards whose recorded model, revision, token limit or text template differ from the corpus."""
    problems = []
    if meta.get("model_id") != corpus.encoder_model_id:
        problems.append(f"model_id {meta.get('model_id')!r} != corpus {corpus.encoder_model_id!r}")
    if meta.get("revision") != corpus.encoder_model_revision:
        problems.append(f"revision {meta.get('revision')!r} != corpus {corpus.encoder_model_revision!r}")
    if meta.get("text") != TEXT_TEMPLATE:
        problems.append(f"text template {meta.get('text')!r} != {TEXT_TEMPLATE!r}")
    if int(meta.get("max_seq_length", -1)) != MAX_TOKENS:
        problems.append(f"max_seq_length {meta.get('max_seq_length')!r} != {MAX_TOKENS}")
    if problems:
        raise PatentEmbedError("precomputed vectors do not match the corpus: " + "; ".join(problems))


def _shard_rows(shards: Sequence[Path]) -> Iterator[np.ndarray]:
    for path in shards:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != DIMENSION:
            raise PatentEmbedError(f"{path} has shape {arr.shape}, expected (n, {DIMENSION})")
        yield arr


def import_precomputed(
    store: EmbeddingStore,
    corpus_id: str,
    source: PrecomputedSource,
    *,
    batch_size: int = 2_000,
    policy: PatentEncodingPolicy | None = None,
    progress: Callable[[EmbedReport], None] | None = None,
) -> EmbedReport:
    """Store offline vectors for the corpus's documents that have none yet, then finalize the corpus.

    Ids not in the corpus are skipped (the offline corpus may be larger or held out documents); ids that
    already have a vector are skipped (so an interrupted import resumes). Shard rows must add up to the
    number of ids.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    corpus = _writable_corpus(store, corpus_id)
    check_precomputed_meta(source.meta, corpus)
    policy = policy or PatentEncodingPolicy(model_id=corpus.encoder_model_id, model_revision=corpus.encoder_model_revision,
                                            dtype=str(source.meta.get("dtype", "float16")))  # fmt: skip
    if (policy.model_id, policy.model_revision) != (corpus.encoder_model_id, corpus.encoder_model_revision):
        raise PatentEmbedError("policy does not match the corpus encoder")
    arrays = list(_shard_rows(source.shards))
    total = sum(a.shape[0] for a in arrays)
    if total != len(source.ids):
        raise PatentEmbedError(f"shards hold {total} rows but the id list has {len(source.ids)} ids")
    if source.text_digests is not None and len(source.text_digests) != len(source.ids):
        raise PatentEmbedError("text digests and ids differ in length")
    if len(set(source.ids)) != len(source.ids):
        raise PatentEmbedError("the id list has duplicate ids")

    report = EmbedReport(corpus_id, text_checked=source.text_digests is not None)
    if not report.text_checked:
        log.warning("corpus %s: importing precomputed vectors WITHOUT a text check (the id list has no titles or "
                    "abstracts); a document whose text changed since the vectors were computed keeps a stale "
                    "vector. Pass a JSONL id list with title/abstract to check.", corpus_id)  # fmt: skip
    report.documents, report.embedded_before = store.counts(corpus_id)
    done = store.embedded_ids(corpus_id)
    offsets = np.cumsum([0] + [a.shape[0] for a in arrays])

    def row(i: int) -> np.ndarray:
        s = int(np.searchsorted(offsets, i, side="right") - 1)
        return np.asarray(arrays[s][i - offsets[s]], dtype=np.float32)

    pending = [i for i, doc_id in enumerate(source.ids) if doc_id not in done]
    report.skipped_existing = len(source.ids) - len(pending)
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        docs = store.documents(corpus_id, [source.ids[i] for i in chunk])
        rows: list[tuple[str, np.ndarray]] = []
        for i in chunk:
            doc_id = source.ids[i]
            doc = docs.get(doc_id)
            if doc is None:
                report.skipped_unknown_id += 1
                continue
            if source.text_digests is not None and source.text_digests[i] != text_digest(doc.title, doc.abstract):
                report.skipped_text_mismatch += 1
                continue
            rows.append((doc_id, row(i)))
        if rows:
            vectors = np.stack([v for _, v in rows])
            try:
                check_normalised(vectors, tolerance=IMPORT_NORM_TOLERANCE)
            except PaecterError as exc:
                raise PatentEmbedError(f"precomputed vectors: {exc}") from exc
            store.write(corpus_id, rows)
            report.imported += len(rows)
            report.batches.append(len(rows))
        if progress is not None:
            progress(report)
    return finalize_corpus(store, corpus_id, policy, report)


# --- database store ------------------------------------------------------------------------------------------


class DbEmbeddingStore:
    """``EmbeddingStore`` on a transactional psycopg connection. Every method ends its transaction."""

    def __init__(self, conn: Connection) -> None:
        if conn.autocommit:
            raise PatentEmbedError("DbEmbeddingStore needs a transactional connection; autocommit is on")
        self.conn = conn

    def _read(self, query: str, params: Sequence[Any]) -> list[tuple[Any, ...]]:
        try:
            rows = self.conn.execute(query, params).fetchall()
        finally:
            self.conn.rollback()  # end the read transaction; nothing was written
        return rows

    def corpus(self, corpus_id: str) -> Corpus | None:
        try:
            return patents.get_corpus(self.conn, corpus_id)
        finally:
            self.conn.rollback()

    def pending(self, corpus_id: str, after: str | None, limit: int) -> list[PendingDocument]:
        # Keyset pagination in the column's own collation, so the primary key index serves the order.
        # ``after`` only ever comes from the previous page, so the comparison and the order agree.
        keyset = "" if after is None else " AND d.document_id > %s"
        params: list[Any] = [corpus_id] + ([] if after is None else [after]) + [limit]
        rows = self._read(
            "SELECT d.document_id, d.title, d.abstract FROM patent_documents d"
            f" WHERE d.corpus_id = %s{keyset}"
            " AND NOT EXISTS (SELECT 1 FROM patent_embeddings e"
            "                 WHERE e.corpus_id = d.corpus_id AND e.document_id = d.document_id)"
            " ORDER BY d.document_id LIMIT %s",
            params,
        )
        return [PendingDocument(*r) for r in rows]

    def embedded_ids(self, corpus_id: str) -> set[str]:
        return {r[0] for r in self._read("SELECT document_id FROM patent_embeddings WHERE corpus_id = %s", [corpus_id])}

    def documents(self, corpus_id: str, document_ids: Sequence[str]) -> dict[str, PendingDocument]:
        rows = self._read(
            "SELECT document_id, title, abstract FROM patent_documents WHERE corpus_id = %s AND document_id = ANY(%s)",
            [corpus_id, list(document_ids)],
        )
        return {r[0]: PendingDocument(*r) for r in rows}

    def write(self, corpus_id: str, rows: Sequence[tuple[str, np.ndarray]]) -> int:
        return patents.insert_embeddings(self.conn, corpus_id, rows, batch_size=max(1, len(rows)))

    def counts(self, corpus_id: str) -> tuple[int, int]:
        (docs, embedded), = self._read(
            "SELECT (SELECT count(*) FROM patent_documents WHERE corpus_id = %s),"
            " (SELECT count(*) FROM patent_embeddings WHERE corpus_id = %s)",
            [corpus_id, corpus_id],
        )  # fmt: skip
        return int(docs), int(embedded)

    def input_rows(self, corpus_id: str) -> Iterator[tuple[str, str]]:
        # Digests are computed server-side (about 40 bytes per document cross the network, not the
        # abstracts). COLLATE "C" is code point order for UTF-8, so the hash does not depend on the
        # server's locale. Streamed with a server-side cursor.
        try:
            with self.conn.cursor(name="patent_input_rows") as cur:
                cur.itersize = 10_000
                cur.execute(
                    "SELECT document_id, md5(coalesce(title, '') || chr(31) || abstract) FROM patent_documents"
                    " WHERE corpus_id = %s ORDER BY document_id COLLATE \"C\"",
                    [corpus_id],
                )
                yield from cur
        finally:
            self.conn.rollback()

    def set_embedded(self, corpus_id: str, document_count: int, input_hash: str) -> Corpus:
        return patents.set_corpus_state(self.conn, corpus_id, "embedded", document_count=document_count,
                                        input_hash=input_hash)  # fmt: skip
