"""TASK-0005: patent ingestion (source records -> corpus documents with raw and resolved labels).

Unit tests use in-memory fakes for the reader, the catalogue and the repository. The integration test
ingests the HUPD fixture tarball end to end; it writes to sakthi only with an explicit opt-in (there is no
local docker database on this machine):

    IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1 uv run pytest -m integration tests/test_patent_ingest.py

It touches only the patent tables and only corpus ids starting with "it-test-", deletes them afterwards, and
refuses to run when a real (non-it-test) corpus is serving, loading or embedded.
"""

from __future__ import annotations

import os
import random
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from app.db.patents import Corpus, CorpusSpec, CorpusStateError, PatentDocument
from app.services.patents import ingest as ing
from app.services.patents.ingest import IngestError, has_usable_label, ingest, split_sample
from app.services.patents.labels import ResolvedLabel
from app.services.patents.sources import HupdReader, PatentRecord

CATALOGUE = frozenset({"H04L", "H04L 9/00", "H04L 9/08", "H04L 12/00", "H04L 12/18", "G06F", "G06F 16/00",
                       "G06F 16/10", "F16H", "F16H 25/00", "F16H 25/22", "B62D", "B62D 5/00", "B62D 5/04"})  # fmt: skip
LABELS = ["H04L908", "H04L1218", "G06F1610", "F16H2522", "B62D504"]


def _record(i: int, labels: tuple[Any, ...] | None = None, *, scheme: str = "compact", source: str = "hupd"
            ) -> PatentRecord:  # fmt: skip
    return PatentRecord(source=source, document_id=f"{15_000_000 + i}", title=f"Title {i}",
                        abstract=f"Abstract number {i} about something long enough to keep.", filing_date=date(2017, 1, 2),
                        publication_date=None, raw_labels=labels if labels is not None else (LABELS[i % 5], "H04L908"),
                        label_scheme=scheme)  # type: ignore[arg-type]  # fmt: skip


class FakeReader:
    """Re-iterable like HupdReader (each ``iter`` starts over), with reader stats."""

    def __init__(self, records: list[PatentRecord]) -> None:
        self.records = records
        self.passes = 0
        self.stats = _Stats()

    def __iter__(self) -> Iterator[PatentRecord]:
        self.passes += 1
        for r in self.records:
            self.stats.read += 1
            yield r


class _Stats:
    def __init__(self) -> None:
        self.read = 0

    def as_dict(self) -> dict[str, int]:
        return {"read": self.read}


class FakeRepo:
    """In-memory corpus storage with the db.patents semantics ingestion relies on."""

    def __init__(self) -> None:
        self.corpora: dict[str, Corpus] = {}
        self.documents: dict[str, dict[str, PatentDocument]] = {}
        self.batches: list[int] = []

    def create_or_get_corpus(self, spec: CorpusSpec) -> Corpus:
        if spec.corpus_id not in self.corpora:
            now = datetime(2026, 10, 13, tzinfo=UTC)
            self.corpora[spec.corpus_id] = Corpus(
                spec.corpus_id, spec.source, spec.description, spec.label_scheme, spec.label_ipc_edition,
                spec.labels_taxonomy_version, 0, spec.encoder_model_id, spec.encoder_model_revision, None, "loading",
                False, now, now, None)  # fmt: skip
            self.documents[spec.corpus_id] = {}
        return self.corpora[spec.corpus_id]

    def set_loading(self, corpus_id: str, document_count: int | None = None) -> Corpus:
        c = self.corpora[corpus_id]
        if c.is_serving:
            raise CorpusStateError(f"corpus {corpus_id!r} is serving")
        c = replace(c, state="loading", document_count=c.document_count if document_count is None else document_count)
        self.corpora[corpus_id] = c
        return c

    def upsert_documents(self, corpus_id: str, documents: Iterable[PatentDocument], batch_size: int) -> int:
        written, batch = 0, 0
        for d in documents:
            self.documents[corpus_id][d.document_id] = d
            written += 1
            batch += 1
            if batch == batch_size:
                self.batches.append(batch)
                batch = 0
        if batch:
            self.batches.append(batch)
        return written

    def document_count(self, corpus_id: str) -> int:
        return len(self.documents[corpus_id])


def _ids(repo: FakeRepo, corpus_id: str = "c1") -> set[str]:
    return set(repo.documents[corpus_id])


# --- sampling ------------------------------------------------------------------------------------------------


def test_split_sample_is_prepare_corpus_shuffle() -> None:
    # prepare_corpus.py shuffles its pool of records; shuffling positions gives the same permutation
    pool = [{"id": f"x{i}"} for i in range(1000)]
    random.Random(20261013).shuffle(pool)
    holdout_pos, corpus_pos = split_sample(1000, seed=20261013, holdout_size=20, sample_size=300)
    assert [f"x{p}" for p in holdout_pos] == [r["id"] for r in pool[:20]]
    assert [f"x{p}" for p in corpus_pos] == [r["id"] for r in pool[20:320]]
    _, rest = split_sample(1000, seed=20261013, holdout_size=20)
    assert len(rest) == 980  # sample_size None: everything after the hold-out
    with pytest.raises(ValueError):
        split_sample(10, seed=1, holdout_size=-1)


def test_usable_label_is_the_offline_pool_rule() -> None:
    assert has_usable_label(_record(1, ("H04L908",)))
    assert has_usable_label(_record(1, ("H04L", " h04l908 ")))  # stripped and upper-cased like prepare_corpus
    assert not has_usable_label(_record(1, ("H04L", "H04L90")))  # fewer than three digits
    assert not has_usable_label(_record(1, ()))
    assert has_usable_label(_record(1, ({"section": "H"},), scheme="structured"))
    assert not has_usable_label(_record(1, (), scheme="structured"))


def test_sampling_is_deterministic_and_the_holdout_is_disjoint() -> None:
    records = [_record(i) for i in range(200)]
    runs = []
    for seed in (7, 7, 8):
        repo = FakeRepo()
        m = ingest(FakeReader(records), "c1", CATALOGUE, seed=seed, sample_size=50, holdout_size=20, repository=repo)
        runs.append((m, _ids(repo)))
    (m1, ids1), (m2, ids2), (m3, ids3) = runs
    assert ids1 == ids2 and m1["holdout_ids"] == m2["holdout_ids"]  # same seed, same documents
    assert ids1 != ids3  # another seed, another sample
    assert len(ids1) == 50 and len(m1["holdout_ids"]) == 20
    assert not ids1 & set(m1["holdout_ids"])  # hold-out never enters the corpus
    assert m1["counts"] == {"read": 200, "no_usable_label": 0, "pool": 200, "held_out": 20, "not_sampled": 130,
                            "selected": 50, "unresolvable": 0, "stored": 50}  # fmt: skip
    # exactly the documents prepare_corpus.py would pick
    pool = list(records)
    random.Random(7).shuffle(pool)
    assert m1["holdout_ids"] == [r.document_id for r in pool[:20]]
    assert ids1 == {r.document_id for r in pool[20:70]}


def test_one_shot_iterator_gives_the_same_sample_as_two_passes() -> None:
    records = [_record(i) for i in range(120)]
    reader = FakeReader(records)
    repo_a, repo_b = FakeRepo(), FakeRepo()
    a = ingest(reader, "c1", CATALOGUE, seed=3, sample_size=40, holdout_size=10, repository=repo_a)
    b = ingest(iter(records), "c1", CATALOGUE, seed=3, sample_size=40, holdout_size=10, repository=repo_b)
    assert reader.passes == 2 and a["sampling"]["two_pass"] and not b["sampling"]["two_pass"]
    assert _ids(repo_a) == _ids(repo_b) and a["holdout_ids"] == b["holdout_ids"]
    assert a["reader_stats"] == {"read": 120}  # first pass only, not doubled by the second


def test_a_changing_source_is_refused_on_the_second_pass() -> None:
    class Flaky(FakeReader):
        def __iter__(self) -> Iterator[PatentRecord]:
            self.passes += 1
            yield from (self.records if self.passes == 1 else self.records[::-1])

    with pytest.raises(IngestError, match="second pass"):
        ingest(Flaky([_record(i) for i in range(30)]), "c1", CATALOGUE, seed=1, holdout_size=5, repository=FakeRepo())


# --- writing ---------------------------------------------------------------------------------------------------


def test_ingest_is_idempotent_and_leaves_the_corpus_loading() -> None:
    records = [_record(i) for i in range(30)]
    repo = FakeRepo()
    first = ingest(FakeReader(records), "c1", CATALOGUE, seed=1, repository=repo, batch_size=8,
                   source_parameters={"years": [2017]})  # fmt: skip
    snapshot = dict(repo.documents["c1"])
    second = ingest(FakeReader(records), "c1", CATALOGUE, seed=1, repository=repo, batch_size=8)
    assert repo.documents["c1"] == snapshot and len(snapshot) == 30  # no duplicates, same rows
    assert first["counts"] == second["counts"]
    assert first["counts"]["stored"] == first["corpus_document_count"] == 30
    assert repo.batches == [8, 8, 8, 6] * 2  # batches of batch_size
    assert repo.corpora["c1"].state == "loading" and repo.corpora["c1"].document_count == 30
    assert first["state"] == "loading" and first["holdout_ids"] == []
    assert first["sampling"]["method"].startswith("none") and first["source_parameters"] == {"years": [2017]}
    doc = repo.documents["c1"]["15000001"]
    assert doc.raw_labels == ["H04L1218", "H04L908"]  # as the source wrote them, main first
    assert doc.resolved_labels == [("H04L", "H04L 12/00", "H04L 12/18"), ("H04L", "H04L 9/00", "H04L 9/08")]
    # i % 5 == 0 repeats "H04L908" (counted once per document): 6 * 1 + 24 * 2 distinct labels
    assert first["label_resolution"]["exact"] == 54 and first["label_resolution"]["missing"] == 0


def test_unresolvable_and_unusable_documents_are_skipped_and_counted() -> None:
    records = [_record(0), _record(1, ("H01L21027",)), _record(2, ("H04L",)), _record(3, ("Z99Z1234", "H01L2100"))]
    repo = FakeRepo()
    m = ingest(FakeReader(records), "c1", CATALOGUE, seed=1, repository=repo)
    assert _ids(repo) == {"15000000"}
    assert m["counts"] == {"read": 4, "no_usable_label": 1, "pool": 3, "held_out": 0, "not_sampled": 0,
                           "selected": 3, "unresolvable": 2, "stored": 1}  # fmt: skip
    assert m["label_resolution"]["missing"] == 3


def test_concordance_is_applied_and_counted() -> None:
    class Conc:
        sources = ("wipo_ipc_concordance_20260101.zip",)

        def resolve(self, code: str) -> ResolvedLabel | None:
            return ResolvedLabel("G06F", "G06F 16/00", None) if code == "H01L21027" else None

        def apply(self, code: str) -> list[str] | None:
            raise AssertionError("not used")

    repo = FakeRepo()
    m = ingest(FakeReader([_record(1, ("H01L21027",))]), "c1", CATALOGUE, seed=1, concordance=Conc(), repository=repo)
    assert repo.documents["c1"]["15000001"].resolved_labels == [("G06F", "G06F 16/00", None)]
    assert m["label_resolution"]["concordance"] == 1 and m["counts"]["unresolvable"] == 0
    assert m["concordance"] == {"sources": ["wipo_ipc_concordance_20260101.zip"]}


def test_corpus_spec_and_state_rules() -> None:
    repo = FakeRepo()
    m = ingest(FakeReader([_record(1)]), "c1", CATALOGUE, seed=1, repository=repo, labels_taxonomy_version="20260101",
               label_ipc_edition="2017.01")  # fmt: skip
    c = repo.corpora["c1"]
    assert (c.source, c.label_scheme, c.labels_taxonomy_version, c.label_ipc_edition) == ("hupd", "compact", "20260101",
                                                                                          "2017.01")  # fmt: skip
    assert m["encoder"]["model_id"] == "mpi-inno-comp/paecter"
    # an embedded (or validated, not serving) corpus goes back to loading on a re-run
    repo.corpora["c1"] = replace(c, state="embedded")
    ingest(FakeReader([_record(1)]), "c1", CATALOGUE, seed=1, repository=repo)
    assert repo.corpora["c1"].state == "loading"
    # a serving corpus is refused before anything is written
    repo.corpora["c1"] = replace(c, state="validated", is_serving=True)
    before = dict(repo.documents["c1"])
    with pytest.raises(CorpusStateError):
        ingest(FakeReader([_record(2)]), "c1", CATALOGUE, seed=1, repository=repo)
    assert repo.documents["c1"] == before


def test_mixed_sources_or_schemes_are_refused() -> None:
    records = [_record(1), _record(2, ({"section": "H"},), scheme="structured", source="uspto_odp")]
    with pytest.raises(IngestError, match="one source and one label scheme"):
        ingest(FakeReader(records), "c1", CATALOGUE, seed=1, repository=FakeRepo())
    with pytest.raises(IngestError, match="no records"):
        ingest(FakeReader([]), "c1", CATALOGUE, seed=1, repository=FakeRepo())


def test_structured_labels_are_resolved_with_the_structured_scheme() -> None:
    labels = ({"section": "H", "class": "04", "subclass": "L", "main_group": "12", "subgroup": "18"},
              {"section": "H", "class": "04", "subclass": "L", "main_group": "0009", "subgroup": "080000"})
    repo = FakeRepo()
    ingest(FakeReader([_record(1, labels, scheme="structured", source="uspto_odp")]), "c1", CATALOGUE, seed=1,
           repository=repo)  # fmt: skip
    doc = repo.documents["c1"]["15000001"]
    assert doc.resolved_labels == [("H04L", "H04L 12/00", "H04L 12/18"), ("H04L", "H04L 9/00", "H04L 9/08")]
    assert doc.raw_labels == list(labels)


def test_db_repository_refuses_autocommit() -> None:
    from psycopg import Connection

    class Auto(Connection):  # type: ignore[misc]
        def __init__(self) -> None:  # no real connection
            pass

        @property
        def autocommit(self) -> bool:  # type: ignore[override]
            return True

    with pytest.raises(IngestError, match="autocommit"):
        ing.DbIngestRepository(Auto())


# --- integration: sakthi with an explicit opt-in (or a local database) ---------------------------------------

TEST_PREFIX = "it-test-"
BUSY_STATES = ("loading", "embedded")
HUPD_SAMPLE = Path(__file__).resolve().parent / "fixtures" / "patents" / "hupd_2017_sample.tar.gz"


def _integration_target() -> str | None:
    target = os.environ.get("IPC_DB_TARGET")
    if target == "local":
        return target
    if target == "sakthi" and os.environ.get("IPC_INTEGRATION_ALLOW_SAKTHI") == "1":
        return target
    return None


def _refusal(corpora: list[tuple[str, str, bool]]) -> str | None:
    """Refuse when a real (non-it-test) corpus is serving, loading or embedded (a live load may be running)."""
    for corpus_id, state, is_serving in corpora:
        if corpus_id.startswith(TEST_PREFIX):
            continue
        if is_serving:
            return f"refusing to run: corpus {corpus_id!r} is serving"
        if state in BUSY_STATES:
            return f"refusing to run: corpus {corpus_id!r} is {state!r} (a load may be running)"
    return None


def test_integration_refuses_when_a_real_corpus_is_serving_or_busy() -> None:
    assert _refusal([("it-test-a", "loading", False)]) is None
    assert _refusal([("hupd-2017", "validated", False)]) is None
    assert "serving" in _refusal([("hupd-2017", "validated", True)])  # type: ignore[operator]
    assert "'loading'" in _refusal([("uspto-2021", "loading", False)])  # type: ignore[operator]


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
        left = conn.execute("SELECT count(*) FROM patent_documents WHERE corpus_id = ANY(%s)", [created]).fetchone()[0]
        conn.rollback()
        conn.close()
        assert left == 0


@pytest.mark.integration
def test_ingest_the_hupd_fixture_end_to_end(db_conn) -> None:
    from app.db import patents

    conn, created = db_conn
    corpus_id = f"{TEST_PREFIX}{uuid.uuid4().hex[:10]}"
    created.append(corpus_id)
    catalogue = {"H04L", "H04L 9/00", "H04L 9/08", "H04L 29/00", "F16H", "F16H 25/00", "F16H 25/22", "B62D",
                 "B62D 5/00", "B62D 5/04"}  # fmt: skip
    reader = HupdReader([HUPD_SAMPLE])
    m = ingest(reader, corpus_id, catalogue, seed=20261013, repository=conn, labels_taxonomy_version="20260101",
               label_ipc_edition="2017.01", source_parameters={"paths": [HUPD_SAMPLE.name]})  # fmt: skip
    print("\nmanifest:", m)
    assert m["counts"] == {"read": 2, "no_usable_label": 0, "pool": 2, "held_out": 0, "not_sampled": 0,
                           "selected": 2, "unresolvable": 0, "stored": 2}  # fmt: skip
    assert m["reader_stats"] == {"read": 3, "yielded": 2, "skipped": 1, "skipped_reasons": {"short_abstract": 1}}
    corpus = patents.get_corpus(conn, corpus_id)
    docs = patents.fetch_documents(conn, corpus_id, ["15900001", "15900002"])
    conn.rollback()
    assert corpus.state == "loading" and corpus.document_count == 2 and corpus.label_scheme == "compact"
    assert corpus.labels_taxonomy_version == "20260101" and corpus.label_ipc_edition == "2017.01"
    assert docs["15900001"].raw_labels == ["H04L908", "H04L2906"]
    assert docs["15900001"].resolved_labels == [("H04L", "H04L 9/00", "H04L 9/08"), ("H04L", "H04L 29/00", None)]
    assert docs["15900002"].resolved_labels == [("F16H", "F16H 25/00", "F16H 25/22"), ("B62D", "B62D 5/00", "B62D 5/04")]
    assert docs["15900001"].filing_date == date(2017, 10, 5)

    again = ingest(HupdReader([HUPD_SAMPLE]), corpus_id, catalogue, seed=20261013, repository=conn)
    assert again["counts"]["stored"] == 2 and again["corpus_document_count"] == 2  # idempotent: no duplicates

    # hold-out: 1 of the 2 documents, recorded in the manifest only
    held_id = f"{TEST_PREFIX}{uuid.uuid4().hex[:10]}"
    created.append(held_id)
    h = ingest(HupdReader([HUPD_SAMPLE]), held_id, catalogue, seed=20261013, holdout_size=1, repository=conn)
    stored = set(patents.fetch_documents(conn, held_id, ["15900001", "15900002"]))
    conn.rollback()
    assert len(h["holdout_ids"]) == 1 and stored == {"15900001", "15900002"} - set(h["holdout_ids"])
