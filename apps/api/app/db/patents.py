"""Storage helpers for labelled patent corpora (migration 0004, ADR-0005).

Write helpers take a psycopg ``Connection`` and commit per batch or state change, like the embedding
runner and the importer, so a long load can resume. The read helpers (``get_serving_corpus``,
``fetch_documents``) also accept a SQLAlchemy ``Session`` (they run on its driver connection) so the
request path can use the pooled engine.

``resolved_labels`` are stored as a JSON array of ``[subclass, main_group | null, exact | null]``
arrays, main label first: the shape of ``services.patents.labels.ResolvedLabel``.

All values are bound parameters; identifiers are fixed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, TypeVar

import numpy as np
from pgvector import HalfVector
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.types.json import Jsonb
from sqlalchemy.orm import Session

CorpusState = Literal["loading", "embedded", "validated", "failed"]
LabelScheme = Literal["compact", "structured"]
CORPUS_STATES: tuple[str, ...] = ("loading", "embedded", "validated", "failed")
LABEL_SCHEMES: tuple[str, ...] = ("compact", "structured")
EMBEDDING_DIM = 1024
HALF_MAX = 65504.0  # largest finite float16
DEFAULT_BATCH_SIZE = 2_000

CORPUS_FIELDS = (
    "corpus_id, source, description, label_scheme, label_ipc_edition, labels_taxonomy_version, document_count,"
    " encoder_model_id, encoder_model_revision, input_hash, state, is_serving, created_at, updated_at, served_at"
)
DOCUMENT_FIELDS = (
    "document_id, title, abstract, filing_date, publication_date, raw_labels, resolved_labels"
)


class PatentCorpusError(RuntimeError):
    pass


class CorpusNotFound(PatentCorpusError):
    pass


class CorpusConflict(PatentCorpusError):
    """An existing corpus with this id was created with a different source, scheme or encoder."""


class CorpusStateError(PatentCorpusError):
    """The corpus is not in a state that allows the operation (e.g. serving an unvalidated corpus)."""


@dataclass(frozen=True)
class CorpusSpec:
    """What identifies a corpus. Re-running a load with the same spec finds the same corpus."""

    corpus_id: str
    source: str
    label_scheme: LabelScheme
    encoder_model_id: str
    encoder_model_revision: str
    description: str | None = None
    label_ipc_edition: str | None = None  # IPC edition the source labels were assigned under
    labels_taxonomy_version: str | None = None  # catalogue release the labels were resolved against


@dataclass(frozen=True)
class Corpus:
    corpus_id: str
    source: str
    description: str | None
    label_scheme: str
    label_ipc_edition: str | None
    labels_taxonomy_version: str | None
    document_count: int
    encoder_model_id: str
    encoder_model_revision: str
    input_hash: str | None
    state: str
    is_serving: bool
    created_at: datetime
    updated_at: datetime
    served_at: datetime | None


@dataclass(frozen=True)
class PatentDocument:
    document_id: str
    title: str | None
    abstract: str
    filing_date: date | None
    publication_date: date | None
    raw_labels: list[Any]  # as the source wrote them, main label first
    resolved_labels: list[tuple[str, str | None, str | None]]  # main label first


DbConn = Connection | Session

T = TypeVar("T")


def _driver(conn: DbConn) -> Connection:
    if isinstance(conn, Session):
        return conn.connection().connection.driver_connection  # type: ignore[return-value]
    return conn


def _batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _get_corpus(conn: Connection, corpus_id: str, *, lock: str = "") -> Corpus | None:
    row = conn.execute(
        f"SELECT {CORPUS_FIELDS} FROM patent_corpora WHERE corpus_id = %s {lock}", [corpus_id]
    ).fetchone()
    return Corpus(*row) if row else None


def get_corpus(conn: DbConn, corpus_id: str) -> Corpus | None:
    return _get_corpus(_driver(conn), corpus_id)


def create_or_get_corpus(conn: Connection, spec: CorpusSpec) -> Corpus:
    """Create the corpus in state ``loading``, or return the existing one with the same id.

    An existing corpus must match the spec's source, label scheme and encoder, otherwise
    ``CorpusConflict``: mixing encoders in one corpus would make its vectors incomparable.
    """
    if spec.label_scheme not in LABEL_SCHEMES:
        raise ValueError(f"label_scheme must be one of {LABEL_SCHEMES}, got {spec.label_scheme!r}")
    conn.execute(
        "INSERT INTO patent_corpora (corpus_id, source, description, label_scheme, label_ipc_edition,"
        " labels_taxonomy_version, encoder_model_id, encoder_model_revision)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (corpus_id) DO NOTHING",
        [spec.corpus_id, spec.source, spec.description, spec.label_scheme, spec.label_ipc_edition,
         spec.labels_taxonomy_version, spec.encoder_model_id, spec.encoder_model_revision],
    )  # fmt: skip
    corpus = _get_corpus(conn, spec.corpus_id)
    conn.commit()
    assert corpus is not None
    wanted = (spec.source, spec.label_scheme, spec.encoder_model_id, spec.encoder_model_revision)
    have = (corpus.source, corpus.label_scheme, corpus.encoder_model_id, corpus.encoder_model_revision)
    if wanted != have:
        raise CorpusConflict(
            f"corpus {spec.corpus_id!r} exists with (source, label_scheme, encoder, revision) = {have}, not {wanted}"
        )
    return corpus


def _require_writable(conn: Connection, corpus_id: str) -> None:
    corpus = _get_corpus(conn, corpus_id, lock="FOR SHARE")
    if corpus is None:
        conn.rollback()
        raise CorpusNotFound(f"corpus {corpus_id!r} not found")
    if corpus.state == "validated":
        conn.rollback()
        raise CorpusStateError(
            f"corpus {corpus_id!r} is validated{' and serving' if corpus.is_serving else ''}; "
            "set it back to 'loading' (and stop serving it) before changing its documents or vectors"
        )


def _labels_json(labels: Sequence[Sequence[str | None]]) -> list[list[str | None]]:
    out = []
    for label in labels:
        if len(label) != 3:
            raise ValueError(f"a resolved label is (subclass, main_group, exact), got {label!r}")
        out.append(list(label))
    return out


def upsert_documents(
    conn: Connection, corpus_id: str, documents: Iterable[PatentDocument], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> int:
    """Insert or update documents in batches (one commit per batch). Returns the number written.

    A document whose title or abstract changed loses its embedding, so it is embedded again. Within a
    batch the last occurrence of a document id wins.
    """
    written = 0
    for batch in _batches(documents, batch_size):
        _require_writable(conn, corpus_id)
        unique = list({d.document_id: d for d in batch}.values())
        rows = [
            (corpus_id, d.document_id, d.title, d.abstract, d.filing_date, d.publication_date,
             Jsonb(list(d.raw_labels)), Jsonb(_labels_json(d.resolved_labels)))
            for d in unique
        ]  # fmt: skip
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS stage_patent_documents"
                " (LIKE patent_documents INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
            )
            with cur.copy(
                "COPY stage_patent_documents (corpus_id, document_id, title, abstract, filing_date,"
                " publication_date, raw_labels, resolved_labels) FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["text", "text", "text", "text", "date", "date", "jsonb", "jsonb"])
                for row in rows:
                    copy.write_row(row)
            cur.execute(
                "DELETE FROM patent_embeddings e USING stage_patent_documents s, patent_documents d"
                " WHERE e.corpus_id = s.corpus_id AND e.document_id = s.document_id"
                " AND d.corpus_id = s.corpus_id AND d.document_id = s.document_id"
                " AND (d.title, d.abstract) IS DISTINCT FROM (s.title, s.abstract)"
            )
            cur.execute(
                "INSERT INTO patent_documents (corpus_id, document_id, title, abstract, filing_date,"
                " publication_date, raw_labels, resolved_labels)"
                " SELECT corpus_id, document_id, title, abstract, filing_date, publication_date, raw_labels,"
                " resolved_labels FROM stage_patent_documents"
                " ON CONFLICT (corpus_id, document_id) DO UPDATE SET title = EXCLUDED.title,"
                " abstract = EXCLUDED.abstract, filing_date = EXCLUDED.filing_date,"
                " publication_date = EXCLUDED.publication_date, raw_labels = EXCLUDED.raw_labels,"
                " resolved_labels = EXCLUDED.resolved_labels, updated_at = now()"
            )
        conn.commit()
        written += len(rows)
    return written


def _halfvec(document_id: str, vector: Any) -> HalfVector:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    if v.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"embedding for {document_id!r} has {v.shape[0]} dimensions, expected {EMBEDDING_DIM}")
    if not np.isfinite(v).all():
        raise ValueError(f"embedding for {document_id!r} contains non-finite values")
    if float(np.abs(v).max()) > HALF_MAX:
        raise ValueError(f"embedding for {document_id!r} overflows halfvec")
    return HalfVector(v)


def insert_embeddings(
    conn: Connection,
    corpus_id: str,
    embeddings: Iterable[tuple[str, Any]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Store ``(document_id, vector)`` pairs as halfvec(1024), in batches (one commit per batch).

    Re-inserting a document's vector replaces it. Every document must already exist in the corpus
    (foreign key). Returns the number written.
    """
    register_vector(conn)
    written = 0
    for batch in _batches(embeddings, batch_size):
        _require_writable(conn, corpus_id)
        unique = {doc_id: vec for doc_id, vec in batch}
        rows = [(corpus_id, doc_id, _halfvec(doc_id, vec)) for doc_id, vec in unique.items()]
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS stage_patent_embeddings"
                " (LIKE patent_embeddings INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
            )
            with cur.copy(
                "COPY stage_patent_embeddings (corpus_id, document_id, embedding) FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["text", "text", "halfvec"])
                for row in rows:
                    copy.write_row(row)
            cur.execute(
                "INSERT INTO patent_embeddings (corpus_id, document_id, embedding)"
                " SELECT corpus_id, document_id, embedding FROM stage_patent_embeddings"
                " ON CONFLICT (corpus_id, document_id) DO UPDATE SET embedding = EXCLUDED.embedding,"
                " created_at = now()"
            )
        conn.commit()
        written += len(rows)
    return written


def fetch_documents(conn: DbConn, corpus_id: str, document_ids: Sequence[str]) -> dict[str, PatentDocument]:
    """Documents of one corpus by id. Ids that do not exist are absent from the result."""
    if not document_ids:
        return {}
    rows = _driver(conn).execute(
        f"SELECT {DOCUMENT_FIELDS} FROM patent_documents WHERE corpus_id = %s AND document_id = ANY(%s)",
        [corpus_id, list(dict.fromkeys(document_ids))],
    ).fetchall()
    out: dict[str, PatentDocument] = {}
    for doc_id, title, abstract, filed, published, raw, resolved in rows:
        out[doc_id] = PatentDocument(doc_id, title, abstract, filed, published, list(raw),
                                     [(lab[0], lab[1], lab[2]) for lab in resolved])  # fmt: skip
    return out


def get_serving_corpus(conn: DbConn) -> Corpus | None:
    row = _driver(conn).execute(f"SELECT {CORPUS_FIELDS} FROM patent_corpora WHERE is_serving").fetchone()
    return Corpus(*row) if row else None


def set_corpus_state(
    conn: Connection,
    corpus_id: str,
    state: CorpusState,
    *,
    document_count: int | None = None,
    input_hash: str | None = None,
) -> Corpus:
    """Move a corpus to ``state``; optionally record its document count and input hash.

    A serving corpus can only stay ``validated``: stop serving it first (``unset_serving_corpus``).
    """
    if state not in CORPUS_STATES:
        raise ValueError(f"state must be one of {CORPUS_STATES}, got {state!r}")
    if document_count is not None and document_count < 0:
        raise ValueError("document_count must be >= 0")
    corpus = _get_corpus(conn, corpus_id, lock="FOR UPDATE")
    if corpus is None:
        conn.rollback()
        raise CorpusNotFound(f"corpus {corpus_id!r} not found")
    if corpus.is_serving and state != "validated":
        conn.rollback()
        raise CorpusStateError(f"corpus {corpus_id!r} is serving; stop serving it before setting it to {state!r}")
    conn.execute(
        "UPDATE patent_corpora SET state = %s, document_count = COALESCE(%s, document_count),"
        " input_hash = COALESCE(%s, input_hash), updated_at = now() WHERE corpus_id = %s",
        [state, document_count, input_hash, corpus_id],
    )
    updated = _get_corpus(conn, corpus_id)
    conn.commit()
    assert updated is not None
    return updated


def set_serving_corpus(conn: Connection, corpus_id: str) -> Corpus:
    """Serve this corpus instead of any other. Refused unless the corpus is ``validated``."""
    corpus = _get_corpus(conn, corpus_id, lock="FOR UPDATE")
    if corpus is None:
        conn.rollback()
        raise CorpusNotFound(f"corpus {corpus_id!r} not found")
    if corpus.state != "validated":
        conn.rollback()
        raise CorpusStateError(f"corpus {corpus_id!r} is {corpus.state!r}; only a validated corpus can serve")
    conn.execute(
        "UPDATE patent_corpora SET is_serving = false, updated_at = now() WHERE is_serving AND corpus_id <> %s",
        [corpus_id],
    )
    conn.execute(
        "UPDATE patent_corpora SET is_serving = true, served_at = now(), updated_at = now()"
        " WHERE corpus_id = %s AND NOT is_serving",
        [corpus_id],
    )
    updated = _get_corpus(conn, corpus_id)
    conn.commit()
    assert updated is not None
    return updated


def unset_serving_corpus(conn: Connection) -> str | None:
    """Stop serving any corpus (the classifier falls back to definitions). Returns the corpus id that was serving."""
    row = conn.execute(
        "UPDATE patent_corpora SET is_serving = false, updated_at = now() WHERE is_serving RETURNING corpus_id"
    ).fetchone()
    conn.commit()
    return row[0] if row else None
