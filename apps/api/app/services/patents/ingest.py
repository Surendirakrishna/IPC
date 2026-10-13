"""Load ``PatentRecord``s from any source reader into a patent corpus (ADR-0005, TASK-0005).

    records (HupdReader, UsptoOdpReader, ...) -> eligible pool -> optional seeded hold-out + sample
      -> labels resolved against the catalogue (+ WIPO concordance) -> documents upserted in batches
      -> corpus left in state ``loading`` (ready for ``embed.embed_corpus`` / ``import_precomputed``)

Pool and sampling reproduce the offline corpus build (data/manifests/benchmarks/knn_hupd/prepare_corpus.py)
exactly, so the precomputed PaECTER vectors (D:/ipc_knn/emb/paecter, keyed by the ids of corpus.jsonl) can
be imported for the same documents:

- the pool is every record with a usable label, in source order. For compact (HUPD) labels "usable" is the
  offline rule, ``^[A-H]\\d{2}[A-Z]\\d{3,}$`` (a subclass plus at least three digits); for structured labels,
  any label;
- ``random.Random(seed).shuffle`` permutes the pool (the permutation depends only on its length), the first
  ``holdout_size`` records are the hold-out and the next ``sample_size`` (default: all the rest) the corpus;
- hold-out ids go to the manifest only, never into the corpus.

Then each corpus record's labels are resolved; a record none of whose labels resolves can't vote and is
skipped (counted as ``unresolvable``). Re-running with the same input and parameters is idempotent: documents
are upserted on (corpus_id, document_id).

With a re-iterable source (a reader object or a list) sampling takes two passes and keeps only document ids
in memory; a one-shot iterator is held in memory instead.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from psycopg import Connection

from app.db import patents
from app.db.patents import Corpus, CorpusSpec, PatentDocument
from app.services.embeddings.paecter import PAECTER_MODEL_ID, PAECTER_REVISION
from app.services.patents.labels import Concordance, ResolutionStats, resolve_labels
from app.services.patents.sources.record import PatentRecord

COMPACT_LABEL_RE = re.compile(r"^[A-H]\d{2}[A-Z]\d{3,}$")  # prepare_corpus.COMPACT
DEFAULT_BATCH_SIZE = 2_000
SAMPLING_METHOD = ("eligible pool in source order; random.Random(seed).shuffle; hold-out = first holdout_size, "
                   "corpus = next sample_size (prepare_corpus.py)")


class IngestError(RuntimeError):
    pass


class HoldoutLeak(IngestError):
    """Hold-out ids are in the corpus after the write (an earlier run with another seed or hold-out, or duplicate
    ids in the source). ``manifest`` is the full ingestion manifest, with ``holdout_leak`` listing them."""

    def __init__(self, message: str, manifest: dict[str, Any]) -> None:
        super().__init__(message)
        self.manifest = manifest


class IngestRepository(Protocol):
    """Corpus storage used by ``ingest``. ``DbIngestRepository`` is the database implementation."""

    def create_or_get_corpus(self, spec: CorpusSpec) -> Corpus: ...

    def set_loading(self, corpus_id: str, document_count: int | None = None) -> Corpus: ...

    def upsert_documents(self, corpus_id: str, documents: Iterable[PatentDocument], batch_size: int) -> int: ...

    def document_count(self, corpus_id: str) -> int: ...

    def existing_ids(self, corpus_id: str, document_ids: Sequence[str]) -> set[str]: ...


class DbIngestRepository:
    """``db.patents`` writers on a transactional psycopg connection (they refuse autocommit and commit per batch)."""

    def __init__(self, conn: Connection) -> None:
        if conn.autocommit:
            raise IngestError("ingestion needs a transactional connection; autocommit is on")
        self.conn = conn

    def create_or_get_corpus(self, spec: CorpusSpec) -> Corpus:
        return patents.create_or_get_corpus(self.conn, spec)

    def set_loading(self, corpus_id: str, document_count: int | None = None) -> Corpus:
        return patents.set_corpus_state(self.conn, corpus_id, "loading", document_count=document_count)

    def upsert_documents(self, corpus_id: str, documents: Iterable[PatentDocument], batch_size: int) -> int:
        return patents.upsert_documents(self.conn, corpus_id, documents, batch_size=batch_size)

    def document_count(self, corpus_id: str) -> int:
        n = self.conn.execute("SELECT count(*) FROM patent_documents WHERE corpus_id = %s", [corpus_id]).fetchone()[0]
        self.conn.rollback()
        return int(n)

    def existing_ids(self, corpus_id: str, document_ids: Sequence[str]) -> set[str]:
        if not document_ids:
            return set()
        rows = self.conn.execute(
            "SELECT document_id FROM patent_documents WHERE corpus_id = %s AND document_id = ANY(%s)",
            [corpus_id, list(document_ids)],
        ).fetchall()
        self.conn.rollback()
        return {r[0] for r in rows}


def has_usable_label(record: PatentRecord) -> bool:
    """Pool membership (prepare_corpus.py): a compact label needs a subclass and at least three digits."""
    if record.label_scheme == "compact":
        return any(COMPACT_LABEL_RE.match(str(label).strip().upper()) for label in record.raw_labels)
    return bool(record.raw_labels)


def split_sample(pool_size: int, *, seed: int, holdout_size: int = 0,
                 sample_size: int | None = None) -> tuple[list[int], list[int]]:  # fmt: skip
    """(hold-out positions, corpus positions) into a pool of ``pool_size`` records, in shuffled order.

    ``random.shuffle`` draws depend only on the list's length, so shuffling positions gives exactly the
    permutation prepare_corpus.py applied to its pool of records.
    """
    if holdout_size < 0 or (sample_size is not None and sample_size < 0):
        raise ValueError("holdout_size and sample_size must be >= 0")
    order = list(range(pool_size))
    random.Random(seed).shuffle(order)
    holdout = order[:holdout_size]
    rest = order[holdout_size:]
    return holdout, rest if sample_size is None else rest[:sample_size]


@dataclass
class _Counts:
    read: int = 0
    no_usable_label: int = 0
    pool: int = 0
    held_out: int = 0
    not_sampled: int = 0
    selected: int = 0
    holdout_duplicate: int = 0  # selected, but its id is also a hold-out id (duplicate ids in the source): skipped
    unresolvable: int = 0
    stored: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _is_reiterable(source: Iterable[PatentRecord]) -> bool:
    return iter(source) is not source


def _reader_stats(source: Any) -> dict[str, Any] | None:
    stats = getattr(source, "stats", None)
    return stats.as_dict() if hasattr(stats, "as_dict") else None


def _check_scheme(record: PatentRecord, scheme: str | None, source: str | None) -> tuple[str, str]:
    if scheme is not None and (record.label_scheme, record.source) != (scheme, source):
        raise IngestError(f"record {record.document_id!r} is ({record.source}, {record.label_scheme}); the load is "
                          f"({source}, {scheme}): one corpus holds one source and one label scheme")  # fmt: skip
    return record.label_scheme, record.source


def ingest(
    source_records: Iterable[PatentRecord],
    corpus_id: str | CorpusSpec,
    catalogue: Container[str],
    *,
    seed: int,
    sample_size: int | None = None,
    holdout_size: int = 0,
    concordance: Concordance | None = None,
    repository: IngestRepository | Connection,
    encoder_model_id: str = PAECTER_MODEL_ID,
    encoder_model_revision: str = PAECTER_REVISION,
    description: str | None = None,
    label_ipc_edition: str | None = None,
    labels_taxonomy_version: str | None = None,
    source_parameters: Mapping[str, Any] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    allow_holdout_leak: bool = False,
) -> dict[str, Any]:
    """Write ``source_records`` into the corpus and return the ingestion manifest.

    ``corpus_id`` is the id (the source and label scheme are taken from the records; the encoder defaults to
    the pinned PaECTER) or a full ``CorpusSpec``. ``catalogue`` answers ``display_code in catalogue`` for the
    labels' taxonomy release; ``concordance`` (``concordance.load_concordance(...)``) re-homes abolished
    symbols. ``repository`` is an ``IngestRepository`` or a transactional psycopg connection.

    Without ``sample_size`` and ``holdout_size`` every eligible record is loaded, in source order. The corpus
    is set to ``loading`` before writing (a serving corpus is refused by the state machine) and stays
    ``loading`` with its document count recorded: ready for embedding.

    The hold-out and the corpus are disjoint by position; to keep them disjoint by id, a selected record whose
    id is also a hold-out id is skipped (``holdout_duplicate``), and after writing the corpus is checked for every
    hold-out id. Any found (e.g. left by an earlier run with another seed) raise ``HoldoutLeak`` (listed in
    ``manifest["holdout_leak"]``) unless ``allow_holdout_leak``.

    A ``concordance`` must have been loaded with a catalogue (``load_concordance(path, catalogue)``): without one
    its ``resolve`` cannot tell which targets exist.
    """
    t0 = time.perf_counter()
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if concordance is not None and getattr(concordance, "catalogue", None) is None:
        raise IngestError("the concordance has no catalogue; load it with load_concordance(path, catalogue) for the "
                          "same taxonomy release as the labels")
    repo: IngestRepository = DbIngestRepository(repository) if isinstance(repository, Connection) else repository
    sampling = sample_size is not None or holdout_size > 0
    counts = _Counts()
    stats = ResolutionStats()
    scheme: str | None = None
    source: str | None = None
    holdout_ids: list[str] = []
    reader_stats: dict[str, Any] | None = None
    two_pass = False

    def eligible(records: Iterable[PatentRecord], *, count: bool) -> Iterator[PatentRecord]:
        nonlocal scheme, source
        for record in records:
            scheme, source = _check_scheme(record, scheme, source)
            if count:
                counts.read += 1
            if has_usable_label(record):
                yield record
            elif count:
                counts.no_usable_label += 1

    if sampling:
        two_pass = _is_reiterable(source_records)
        pool_ids: list[str] = []
        pool: list[PatentRecord] = []
        for record in eligible(source_records, count=True):
            pool_ids.append(record.document_id)
            if not two_pass:
                pool.append(record)
        counts.pool = len(pool_ids)
        # A reader's own counters would double on the second pass: keep the first pass's.
        reader_stats = _reader_stats(source_records)
        holdout_pos, corpus_pos = split_sample(counts.pool, seed=seed, holdout_size=holdout_size,
                                               sample_size=sample_size)  # fmt: skip
        holdout_ids = [pool_ids[p] for p in holdout_pos]
        counts.held_out = len(holdout_pos)
        counts.selected = len(corpus_pos)
        counts.not_sampled = counts.pool - counts.held_out - counts.selected
        wanted = set(corpus_pos)

        def selected() -> Iterator[PatentRecord]:
            if not two_pass:
                for p in sorted(wanted):
                    yield pool[p]
                return
            position = -1
            for record in eligible(source_records, count=False):
                position += 1
                if position >= counts.pool or record.document_id != pool_ids[position]:
                    raise IngestError("the source yielded different records on the second pass; it must be "
                                      "deterministic (or pass a one-shot iterator to hold it in memory)")
                if position in wanted:
                    yield record
            if position + 1 != counts.pool:
                raise IngestError(f"the source yielded {position + 1} eligible records on the second pass, "
                                  f"{counts.pool} on the first")

        records: Iterable[PatentRecord] = selected()
    else:

        def every() -> Iterator[PatentRecord]:
            for record in eligible(source_records, count=True):
                counts.pool += 1
                counts.selected += 1
                yield record

        records = every()

    # The corpus needs the source and label scheme, which come from the first record.
    records = iter(records)
    first = next(records, None)
    if first is None and scheme is None:
        raise IngestError("the source yielded no records")
    spec = corpus_id if isinstance(corpus_id, CorpusSpec) else CorpusSpec(
        corpus_id=corpus_id, source=source or "", label_scheme=scheme or "compact",  # type: ignore[arg-type]
        encoder_model_id=encoder_model_id, encoder_model_revision=encoder_model_revision, description=description,
        label_ipc_edition=label_ipc_edition, labels_taxonomy_version=labels_taxonomy_version)  # fmt: skip
    if (spec.source, spec.label_scheme) != (source, scheme):
        raise IngestError(f"corpus spec is ({spec.source}, {spec.label_scheme}) but the records are ({source}, {scheme})")
    corpus = repo.create_or_get_corpus(spec)
    if corpus.state != "loading":
        corpus = repo.set_loading(spec.corpus_id)

    def documents() -> Iterator[PatentDocument]:
        if first is not None:
            yield from _document(first)
        for record in records:
            yield from _document(record)

    holdout_set = set(holdout_ids)

    def _document(record: PatentRecord) -> Iterator[PatentDocument]:
        if record.document_id in holdout_set:
            counts.holdout_duplicate += 1
            return
        resolution = resolve_labels(record.raw_labels, record.label_scheme, catalogue, concordance)
        stats.add(resolution.stats)
        if not resolution.labels:
            counts.unresolvable += 1
            return
        yield PatentDocument(
            document_id=record.document_id, title=record.title, abstract=record.abstract,
            filing_date=record.filing_date, publication_date=record.publication_date,
            raw_labels=list(record.raw_labels), resolved_labels=[tuple(label) for label in resolution.labels],
        )  # fmt: skip

    counts.stored = repo.upsert_documents(spec.corpus_id, documents(), batch_size)
    document_count = repo.document_count(spec.corpus_id)
    corpus = repo.set_loading(spec.corpus_id, document_count=document_count)
    if not sampling:
        reader_stats = _reader_stats(source_records)
    manifest: dict[str, Any] = {
        "corpus_id": spec.corpus_id,
        "source": spec.source,
        "label_scheme": spec.label_scheme,
        "encoder": {"model_id": spec.encoder_model_id, "model_revision": spec.encoder_model_revision},
        "labels_taxonomy_version": spec.labels_taxonomy_version,
        "label_ipc_edition": spec.label_ipc_edition,
        "source_parameters": dict(source_parameters or {}),
        "reader": type(source_records).__name__,
        "reader_stats": reader_stats,
        "sampling": {"seed": seed, "sample_size": sample_size, "holdout_size": holdout_size,
                     "method": SAMPLING_METHOD if sampling else "none: every eligible record in source order",
                     "two_pass": two_pass},  # fmt: skip
        "concordance": None if concordance is None else {"sources": list(getattr(concordance, "sources", ()))},
        "counts": counts.as_dict(),
        "label_resolution": stats.as_dict(),
        "corpus_document_count": document_count,
        "state": corpus.state,
        "holdout_ids": holdout_ids,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if document_count != counts.stored:
        manifest["note"] = (f"the corpus holds {document_count} documents, this run wrote {counts.stored}: documents "
                            "from earlier runs with other parameters (or duplicate ids in the source) are counted too")
    leaked = sorted(repo.existing_ids(spec.corpus_id, holdout_ids)) if holdout_ids else []
    manifest["holdout_leak"] = leaked
    if leaked and not allow_holdout_leak:
        raise HoldoutLeak(f"{len(leaked)} hold-out id(s) are in corpus {spec.corpus_id!r} (e.g. {leaked[:5]}): the "
                          "hold-out would not be held out; delete those documents or load into a fresh corpus",
                          manifest)  # fmt: skip
    return manifest
