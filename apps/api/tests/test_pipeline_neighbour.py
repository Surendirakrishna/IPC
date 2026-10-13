"""TASK-0010: the neighbour vote as the primary classifier in ClassificationPipeline.

Drives the real ``ClassificationPipeline.classify`` end to end with fakes, no database and no models:

- ``FakeSession``: answers the catalogue embedding-run lookup (the only SQL the pipeline issues itself);
- ``FakeRepo``: the taxonomy repository (releases, records, references, WIPO texts), patched in as
  ``pipeline.repo``; keyword/vector search and expansion are patched with catalogue-backed fakes;
- ``FakeEncoder`` / ``FakeReranker``: the BGE query encoder and the cross-encoder (word-overlap scores);
- ``FakePatentEncoder`` / ``FakePatentStore``: PaECTER and the patent corpus (serving corpus + k nearest
  neighbours with resolved labels); the real ``neighbour_vote.vote`` turns those into the vote.

Every response is built by ``ClassifyResponse`` (so its contract validators ran) and is re-validated from
JSON. ``classifier`` is asserted directly on every path, because its default would also validate.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import psycopg
import pytest
from sqlalchemy import exc as sa_exc

from app.config import Settings
from app.db.patents import Corpus
from app.db.repository import Record, Release
from app.schemas.classify import ClassifyRequest, ClassifyResponse, own_level_share
from app.services.classification import pipeline as pl
from app.services.classification.pipeline import (
    NO_PASSAGE_NOTE,
    VOTE_CONFIDENCE_WARNING,
    ClassificationPipeline,
    DbPatentVoteStore,
    PipelineUnavailable,
    internal_code,
)
from app.services.classification.reranker import PairScore
from app.services.confidence.calibrator import ConfidenceService
from app.services.embeddings.paecter import PAECTER_MODEL_ID, PAECTER_REVISION
from app.services.embeddings.providers import EncodingPolicy
from app.services.fields.mapping import load_mapping
from app.services.retrieval.keyword import KeywordHit
from app.services.retrieval.patent_search import NeighbourShortfall, PatentNeighbour
from app.services.retrieval.vector import VectorHit

VERSION = "20260101"
ABSTRACT = (
    "Quantum keys are distributed among smart grid nodes for key distribution. "
    "Each node verifying the identity of a user before encrypted communication. "
    "The system comprises smart meters, substations, inverters, chargers, and control centers."
)
TITLE = "Quantum key distribution in a smart grid"
NOT_CALIBRATED_WARNING = "numeric confidence is null"

# --- catalogue -----------------------------------------------------------------------------------------

_TITLES = {
    "H": ("section", "ELECTRICITY"),
    "G": ("section", "PHYSICS"),
    "H04": ("class", "ELECTRIC COMMUNICATION TECHNIQUE"),
    "H02": ("class", "GENERATION, CONVERSION, OR DISTRIBUTION OF ELECTRIC POWER"),
    "G06": ("class", "COMPUTING OR CALCULATING; COUNTING"),
    "H04L": ("subclass", "TRANSMISSION OF DIGITAL INFORMATION"),
    "H02J": ("subclass", "CIRCUIT ARRANGEMENTS OR SYSTEMS FOR SUPPLYING OR DISTRIBUTING ELECTRIC POWER"),
    "G06Q": ("subclass", "INFORMATION AND COMMUNICATION TECHNOLOGY SPECIALLY ADAPTED FOR ADMINISTRATIVE PURPOSES"),
    "H04L 9/00": ("maingroup", "Arrangements for secret or secure communications"),
    "H04L 9/08": ("subgroup", "Key distribution"),
    "H04L 9/32": ("subgroup", "including means for verifying the identity or authority of a user"),
    "H04L 67/00": ("maingroup", "Network arrangements or protocols for supporting network services"),
    "H04L 67/12": ("subgroup", "adapted for proprietary or special-purpose networking environments"),
    "H02J 13/00": ("maingroup", "Circuit arrangements for providing remote indication of network conditions"),
    "G06Q 50/00": ("maingroup", "Information and communication technology specially adapted for business sectors"),
    "G06Q 50/06": ("subgroup", "Energy or water supply"),
}


def _record(display: str) -> Record:
    rtype, title = _TITLES[display]
    code = internal_code(display)
    section, cls, sub = display[0], display[:3], display[:4]
    chain = [section, cls, sub][: {"section": 0, "class": 1, "subclass": 2}.get(rtype, 3)]
    mg_code = internal_code(display.split("/")[0] + "/00") if "/" in display else None
    if rtype == "subgroup":
        chain.append(mg_code)
    return Record(
        code=code, display_code=display, record_type=rtype, title=title, title_raw=title, semantic_text=title,
        section_code=section, section_title=_TITLES[section][1],
        class_code=cls if rtype != "section" else None, class_title=_TITLES.get(cls, ("", None))[1],
        subclass_code=sub if rtype in ("subclass", "maingroup", "subgroup") else None,
        subclass_title=_TITLES.get(sub, ("", None))[1] if rtype in ("subclass", "maingroup", "subgroup") else None,
        maingroup_code=mg_code, maingroup_title=_TITLES[display.split("/")[0] + "/00"][1] if mg_code else None,
        parent_code=chain[-1] if chain else None, ancestor_codes=chain, child_codes=[], depth=len(chain),
        context_heading=None, titles=None, canonical_title=None, is_duplicate_symbol=False, original_title=None,
        original_title_source=None, official_title=None,
    )  # fmt: skip


CATALOGUE = {internal_code(d): d for d in _TITLES}


def _precedence(owner: str, preferred: str, dominated: str, *, conditional: bool = False) -> dict[str, Any]:
    text = f"{preferred} takes precedence over {dominated}."
    return {"owner_symbol": internal_code(dominated), "source_kind": "scheme_note", "part_type": "note", "ordinal": 1,
            "note_type": "note", "text": text, "related_symbols": [internal_code(preferred)], "container_symbol": None,
            "scope_end_symbol": None, "scope_symbols": None, "editions": "2026.01", "rule_kind": "precedence",
            "rules": [{"kind": "precedence", "preferred": [internal_code(preferred)],
                       "dominated": [internal_code(dominated)], "conditional": conditional, "sentence": text}],
            "applies_to": internal_code(owner)}  # fmt: skip


class FakeRepo:
    """Stands in for ``app.db.repository`` (patched as ``pipeline.repo``)."""

    Release = Release

    def __init__(self, *, official: dict[str, list[dict[str, Any]]] | None = None, active: bool = True,
                 drop: tuple[str, ...] = ()) -> None:  # fmt: skip
        self.official = official or {}
        self.active = active
        self.drop = {internal_code(d) for d in drop}
        self.record_requests: list[list[str]] = []

    def active_release(self, session: Any) -> Release | None:
        return Release(VERSION, {"ipc_scheme": "sha256:test"}, {}, None, "complete", True) if self.active else None

    def get_release(self, session: Any, version: str) -> Release | None:
        return None

    def get_records(self, session: Any, version: str, codes: list[str]) -> dict[str, Record]:
        assert version == VERSION
        self.record_requests.append(list(codes))
        return {c: _record(CATALOGUE[c]) for c in codes if c in CATALOGUE and c not in self.drop}

    def get_references(self, session: Any, version: str, codes: list[str]) -> dict[str, list[dict[str, Any]]]:
        return {}

    def get_official_texts(self, session: Any, version: str, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
        return {s: self.official[s] for s in symbols if s in self.official}


class _Result:
    def __init__(self, row: tuple | None) -> None:
        self.row = row

    def first(self) -> tuple | None:
        return self.row


class FakeSession:
    """Answers the catalogue embedding-run lookup; everything else goes through the patched fakes."""

    def __init__(self, *, embeddings: bool = True) -> None:
        self.embeddings = embeddings
        self.savepoints = 0

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _Result:
        assert "FROM embedding_runs" in str(stmt), str(stmt)
        if not self.embeddings:
            return _Result(None)
        return _Result(("run-test", "ipc_embeddings", {}, "fake/bge", "rev-1"))

    def begin_nested(self) -> contextlib.AbstractContextManager[None]:
        self.savepoints += 1
        return contextlib.nullcontext()


class FakeEncoder:
    """BGE-M3 stand-in (catalogue query encoder)."""

    policy = EncodingPolicy(provider="fake", model_id="fake/bge", model_revision="rev-1", dimension=8,
                            storage_table="ipc_embeddings", normalized=True, document_template="{text}",
                            query_template="{text}", taxonomy_max_tokens=512, query_chunk_tokens=1024,
                            dtype="float32")  # fmt: skip

    def __init__(self) -> None:
        self.calls = 0

    def encode_queries(self, texts: list[str], max_tokens: int | None = None) -> np.ndarray:
        self.calls += 1
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)


def _words(text: str) -> set[str]:
    return {w.strip(".,;:()").lower() for w in text.split() if len(w.strip(".,;:()")) >= 4}


class FakeReranker:
    """Score = shared words (>= 4 letters) between query and passage; a scope listed in ``veto`` scores -10
    for every evidence passage (an input with no support for that code)."""

    model_id = "fake/reranker"
    revision = "rr-1"

    def __init__(self, veto: tuple[str, ...] = ()) -> None:
        self.veto = veto

    def score(self, pairs: list[tuple[str, str]]) -> list[PairScore]:
        out = []
        for query, passage in pairs:
            vetoed = any(passage.startswith(v + ":") for v in self.veto)
            out.append(PairScore(-10.0 if vetoed else float(len(_words(query) & _words(passage))), False, False))
        return out


@dataclass
class FakePatentEncoder:
    model_id: str = PAECTER_MODEL_ID
    revision: str = PAECTER_REVISION
    texts: list[str] = field(default_factory=list)
    error: BaseException | None = None  # raised by encode (a CUDA error, a bug, ...)

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts += list(texts)
        if self.error is not None:
            raise self.error
        return np.full((len(texts), 1024), 1 / 32, dtype=np.float32)


def _corpus(**overrides: Any) -> Corpus:
    now = datetime(2026, 10, 13, tzinfo=UTC)
    values = dict(corpus_id="it-test-fake", source="hupd", description=None, label_scheme="compact",
                  label_ipc_edition="2017.01", labels_taxonomy_version=VERSION, document_count=0,
                  encoder_model_id=PAECTER_MODEL_ID, encoder_model_revision=PAECTER_REVISION, input_hash=None,
                  state="validated", is_serving=True, created_at=now, updated_at=now, served_at=now)  # fmt: skip
    values.update(overrides)
    return Corpus(**values)


class FakePatentStore:
    def __init__(self, corpus: Corpus | None, neighbours: list[PatentNeighbour] | None = None, *,
                 error: Exception | None = None, count: int = 293_711) -> None:  # fmt: skip
        self.corpus = corpus
        self.neighbours = neighbours or []
        self.error = error
        self.count = count
        self.searches: list[tuple[str, int, int]] = []

    def serving_corpus(self, session: Any) -> Corpus | None:
        return self.corpus

    def document_count(self, session: Any, corpus: Corpus) -> int:
        return self.count

    def search(self, session: Any, corpus_id: str, query_vector: Any, k: int, ef_search: int) -> list[PatentNeighbour]:
        self.searches.append((corpus_id, k, ef_search))
        assert np.asarray(query_vector).shape == (1024,)
        if self.error is not None:
            raise self.error
        return list(self.neighbours)


# --- neighbours ------------------------------------------------------------------------------------------


def _label(display: str) -> tuple[str, str | None, str | None]:
    """A resolved label for a display code: (subclass, main group | None, exact | None)."""
    if "/" not in display:
        return (display, None, None)
    return (display[:4], display.split("/")[0] + "/00", display)


def _mg_only(display: str) -> tuple[str, str, None]:
    return (display[:4], display, None)


def _nb(i: int, *labels: tuple, similarity: float = 0.97) -> PatentNeighbour:
    return PatentNeighbour(f"doc-{i:02d}", f"Patent {i}", similarity, tuple(labels))  # type: ignore[arg-type]


K = 10
FILLER = ("A01B", "B65D", "C07D", "E04B", "F16K", "G01N", "H01M")


def strong_neighbours() -> list[PatentNeighbour]:
    """Equal similarities, so each neighbour weighs 1/10. Shares: H04L 0.8 (H04L 9/00 0.8, H04L 9/08 0.8,
    H04L 9/32 0.5), H02J 13/00 0.1 (labelled with the main group itself), G06Q 50/06 0.1; H04 0.8."""
    out = [_nb(i, _label("H04L 9/08"), *([_label("H04L 9/32")] if i < 5 else [])) for i in range(8)]
    out.append(_nb(8, _label("H02J 13/00")))
    out.append(_nb(9, _label("G06Q 50/06")))
    return out


def weak_neighbours() -> list[PatentNeighbour]:
    """Ten neighbours in ten different subclasses (0.1 each). The tie goes to the nearest carrier, so the
    first code is H04L 9/08 with a subclass share of 0.1, below the 0.2 minimum."""
    out = [_nb(0, _label("H04L 9/08"))]
    out += [_nb(1 + j, _label(sub)) for j, sub in enumerate(FILLER + ("G06Q", "H02J"))]
    return out[:K]


def gate_neighbours(beta_secondary: bool = True) -> list[PatentNeighbour]:
    """H04L carried by 2 main labels + 1 secondary label (weighs beta) out of 10 equal neighbours: the raw
    subclass share is (2 + beta) / 10."""
    out = [_nb(0, _label("H04L 9/08")), _nb(1, _label("H04L 9/08")), _nb(2, _label("A01B"), _label("H04L 9/08"))]
    out += [_nb(3 + j, _label(sub)) for j, sub in enumerate(FILLER)]
    return out


# --- harness ---------------------------------------------------------------------------------------------


def patch_catalogue(monkeypatch: pytest.MonkeyPatch, module: Any, repo: FakeRepo,
                    on_keyword: Any = None) -> None:  # fmt: skip
    """Patch a pipeline module's repository, keyword/vector search and expansion with catalogue-backed fakes.
    ``module`` is ``pipeline`` here, or the recorded 1.x pipeline when the golden snapshot is regenerated."""
    monkeypatch.setattr(module, "repo", repo)

    def keyword_search(session, *, version, query, limit, record_types, section=None, subclass=None,
                       subclasses=None):  # fmt: skip
        if on_keyword is not None:
            on_keyword((record_types, subclasses))
        codes = [c for c, d in CATALOGUE.items() if _TITLES[d][0] in record_types
                 and (not subclasses or d[:4] in subclasses) and (not subclass or d[:4] == subclass)]
        return [KeywordHit(c, i + 1, 1.0 / (i + 1)) for i, c in enumerate(sorted(codes))][:limit]

    def vector_search(session, *, table, version, query_vector, limit, ef_search, record_types, section=None,
                      subclass=None, subclasses=None, exact=False):  # fmt: skip
        codes = [c for c, d in CATALOGUE.items() if _TITLES[d][0] in record_types
                 and (not subclasses or d[:4] in subclasses) and (not subclass or d[:4] == subclass)]
        return [VectorHit(c, i + 1, 0.1 + 0.01 * i) for i, c in enumerate(sorted(codes))][:limit]

    def expand(session, *, version, table, fused, abstract_vector, top_n, max_siblings, pool_cap, allowed_types):
        return [c for c in fused if _TITLES[CATALOGUE[c.code]][0] in allowed_types][:pool_cap]

    monkeypatch.setattr(module, "keyword_search", keyword_search)
    monkeypatch.setattr(module, "vector_search", vector_search)
    monkeypatch.setattr(module, "expand", expand)


@dataclass
class Harness:
    pipeline: ClassificationPipeline
    repo: FakeRepo
    session: FakeSession
    encoder: FakeEncoder
    store: FakePatentStore | None
    patent_encoder: FakePatentEncoder | None
    keyword_calls: list[tuple] = field(default_factory=list)

    def classify(self, **req: Any) -> ClassifyResponse:
        body = {"abstract": ABSTRACT, "title": TITLE, **req}
        resp = self.pipeline.classify(self.session, ClassifyRequest(**body))  # type: ignore[arg-type]
        # The contract validators ran when the pipeline built the response; check the wire form too.
        again = ClassifyResponse.model_validate_json(resp.model_dump_json())
        assert again.classifier == resp.classifier
        return resp


@pytest.fixture
def make(monkeypatch: pytest.MonkeyPatch, tmp_path):
    def build(*, enabled: bool = True, store: FakePatentStore | None = None, patent_encoder: Any = "default",
              repo: FakeRepo | None = None, session: FakeSession | None = None, veto: tuple[str, ...] = (),
              **settings: Any) -> Harness:  # fmt: skip
        values = dict(neighbour_vote_enabled=enabled, neighbour_k=K, neighbour_temperature=0.005, neighbour_beta=1.0,
                      neighbour_min_vote_share=0.2, patent_hnsw_ef_search=200)  # fmt: skip
        values.update(settings)
        s = Settings(_env_file=None, **values)
        fake_repo = repo or FakeRepo()
        h = Harness(
            pipeline=None,  # type: ignore[arg-type]
            repo=fake_repo, session=session or FakeSession(), encoder=FakeEncoder(), store=store,
            patent_encoder=FakePatentEncoder() if patent_encoder == "default" else patent_encoder,
        )  # fmt: skip
        patch_catalogue(monkeypatch, pl, fake_repo, on_keyword=h.keyword_calls.append)
        h.pipeline = ClassificationPipeline(
            s, h.encoder, FakeReranker(veto), ConfidenceService(tmp_path, "fp-test"), load_mapping(s.field_mapping_path),
            patent_encoder=h.patent_encoder, patent_store=store,
        )  # fmt: skip
        return h

    return build


def _strong_store(**kw: Any) -> FakePatentStore:
    return FakePatentStore(_corpus(**kw), strong_neighbours())


def assert_vote_primary(resp: ClassifyResponse, h: Harness) -> None:
    assert resp.classifier == "neighbour_vote"
    assert resp.fallback_reason is None
    assert resp.vote is not None
    assert resp.provenance.vote_settings is not None
    assert resp.vote.first_code_share.subclass >= resp.provenance.vote_settings.min_vote_share  # the gate
    assert resp.vote.neighbour_count == resp.provenance.vote_settings.k == K
    assert h.keyword_calls == [] and h.encoder.calls == 0  # no definition retrieval ran
    for c in resp.candidates:
        assert c.origin == ["neighbour_vote"]
        assert c.confidence.status == "vote_share_uncalibrated" and c.confidence.calibrator_id is None
        if c.level != "class":
            assert c.confidence.score == own_level_share(c.level, c.vote_share)
    assert not any(NOT_CALIBRATED_WARNING in w for w in resp.warnings)
    assert VOTE_CONFIDENCE_WARNING in resp.warnings


def assert_fallback(resp: ClassifyResponse, h: Harness, reason: str) -> None:
    assert resp.classifier == "definition_fallback"
    assert resp.fallback_reason == reason
    assert h.keyword_calls, "the definition pipeline did not run"
    for c in [*resp.candidates, *resp.alternatives]:
        assert c.vote_share.is_empty()
        assert c.confidence.status != "vote_share_uncalibrated"
    for c in resp.candidates:
        assert "neighbour_vote" not in c.origin


# --- path 1: vote primary --------------------------------------------------------------------------------


def test_vote_primary_codes_order_and_shares(make) -> None:
    h = make(store=_strong_store())
    resp = h.classify(include_ranked_debug=True)
    assert_vote_primary(resp, h)
    v = resp.vote
    assert (v.first_code, v.first_code_level) == ("H04L 9/08", "subgroup")
    assert v.first_code_share.model_dump() == {"subclass": 0.8, "maingroup": 0.8, "exact": 0.8}
    # vote order; first candidate == first code; further codes only at share >= min (H02J 13/00 is 0.1)
    assert [c.display_code for c in resp.candidates] == ["H04L 9/08", "H04L 9/32"]
    assert resp.candidates[0].display_code == v.first_code
    assert resp.candidates[1].vote_share.model_dump() == {"subclass": 0.8, "maingroup": 0.8, "exact": 0.5}
    assert resp.candidates[1].confidence.score == 0.5
    assert [c.display_code for c in v.top_codes.exact] == ["H04L 9/08", "H04L 9/32", "H02J 13/00", "G06Q 50/06"]
    assert len(v.neighbours) == K and v.neighbours[0].document_id == "doc-00"
    assert v.neighbours[0].ipc_codes == ["H04L 9/08", "H04L 9/32"]
    assert sum(n.vote_weight for n in v.neighbours) == pytest.approx(1.0, abs=1e-3)
    # the patent encoder saw "title. abstract"; the search asked for exactly k with the configured ef_search
    assert h.patent_encoder.texts == [f"{TITLE}. {ABSTRACT}"]
    assert h.store.searches == [("it-test-fake", K, 200)]
    assert h.session.savepoints == 1
    # summary wording (docs/api.md): "vote share: N%"
    assert [ln.confidence_text for ln in resp.summary.classification_lines] == ["vote share: 80.0%", "vote share: 50.0%"]
    assert resp.summary.classification_cell_text == "H04L 9/08 — vote share: 80.0%; H04L 9/32 — vote share: 50.0%"
    assert resp.status == "needs_review"
    assert resp.field_of_invention.mapping_key == "H04"
    assert resp.candidates[0].rationale.startswith(
        "Neighbour vote: 80.0% of the similarity-weighted 10 nearest patents carry H04L, 80.0% H04L 9/00 and "
        "80.0% exactly H04L 9/08.")
    assert {"patent_encode", "patent_search", "vote", "total"} <= set(resp.timings_ms)


def test_vote_primary_provenance(make) -> None:
    h = make(store=_strong_store(label_ipc_edition="2017.01", document_count=0))
    p = h.classify().provenance
    assert p.patent_encoder.model_dump() == {"model_id": PAECTER_MODEL_ID, "model_revision": PAECTER_REVISION}
    # label_ipc_edition (DB) -> labels_ipc_edition (contract); the real count, never the stored 0
    assert p.patent_corpus.model_dump() == {"corpus_id": "it-test-fake", "source": "hupd", "document_count": 293_711,
                                            "labels_ipc_edition": "2017.01"}  # fmt: skip
    assert p.vote_settings.model_dump() == {"k": K, "temperature": 0.005, "beta": 1.0, "min_vote_share": 0.2,
                                            "hnsw_ef_search": 200}  # fmt: skip
    assert p.calibrator_id is None and p.embedding_run_id == "run-test"


def test_vote_primary_debug_has_vote_ranking_and_neighbour_ids(make) -> None:
    h = make(store=_strong_store())
    resp = h.classify(include_ranked_debug=True)
    rd = resp.retrieval_debug
    assert rd["vote_subclass"] == ["H04L", "H02J", "G06Q"]
    assert rd["vote_maingroup"] == ["H04L 9/00", "H02J 13/00", "G06Q 50/00"]
    assert rd["vote_exact"] == ["H04L 9/08", "H04L 9/32", "H02J 13/00", "G06Q 50/06"]
    assert rd["vote_neighbour_ids"] == [f"doc-{i:02d}" for i in range(K)]
    assert [r.display_code for r in resp.ranked_debug] == ["H04L 9/08", "H04L 9/32"]
    assert all(r.outcome == "supported" for r in resp.ranked_debug)
    assert h.classify().retrieval_debug is None  # only on request


def test_evidence_and_wipo_notes_explain_vote_codes_and_never_veto(make) -> None:
    official = {internal_code("H04L 9/32"): [_precedence("H04L 9/32", "H04L 67/12", "H04L 9/32")]}
    # H04L 9/32's scope gets no evidence at all: it stays (vote primary), with a review note
    h = make(store=_strong_store(), repo=FakeRepo(official=official), veto=("H04L 9/32",))
    resp = h.classify()
    assert_vote_primary(resp, h)
    first, second = resp.candidates
    assert first.supporting_passages, "evidence passages are computed for vote codes"
    assert first.supporting_passages[0].quote in ABSTRACT + TITLE
    assert second.display_code == "H04L 9/32" and second.supporting_passages == []
    assert NO_PASSAGE_NOTE in second.unresolved_ambiguity
    assert "No input passage supports it" in second.rationale
    # WIPO notes are attached to the vote code they apply to; the cited preferred place is NOT added
    assert any(a.startswith("WIPO precedence rule for H04L 9/32") for a in second.unresolved_ambiguity)
    codes = {c.display_code for c in [*resp.candidates, *resp.alternatives]}
    assert codes <= {c.display_code for c in resp.vote.top_codes.exact}
    assert "H04L 67/12" not in codes


def test_precedence_demotion_moves_first_code_to_alternatives(make) -> None:
    official = {internal_code("H04L 9/08"): [_precedence("H04L 9/08", "H04L 9/32", "H04L 9/08")]}
    h = make(store=_strong_store(), repo=FakeRepo(official=official))
    resp = h.classify()
    assert_vote_primary(resp, h)
    assert [c.display_code for c in resp.candidates] == ["H04L 9/32"]
    demoted = [a for a in resp.alternatives if a.display_code == "H04L 9/08"]
    assert len(demoted) == 1 and "takes precedence over H04L 9/08" in demoted[0].reason
    assert demoted[0].vote_share.exact == 0.8 and demoted[0].confidence.status == "vote_share_uncalibrated"


def test_conditional_precedence_only_adds_a_note(make) -> None:
    official = {internal_code("H04L 9/08"): [_precedence("H04L 9/08", "H04L 9/32", "H04L 9/08", conditional=True)]}
    resp = make(store=_strong_store(), repo=FakeRepo(official=official)).classify()
    assert [c.display_code for c in resp.candidates] == ["H04L 9/08", "H04L 9/32"]
    assert any("conditional rule" in a for a in resp.candidates[0].unresolved_ambiguity)


def test_max_results_overflow_becomes_alternatives(make) -> None:
    resp = make(store=_strong_store()).classify(max_results=1)
    assert [c.display_code for c in resp.candidates] == ["H04L 9/08"]
    assert [(a.display_code, a.reason) for a in resp.alternatives] == [("H04L 9/32", "Beyond max_results ceiling.")]
    assert resp.alternatives[0].confidence.score == 0.5


@pytest.mark.parametrize(
    ("target", "display", "level", "score", "share"),
    [
        ("subgroup", "H04L 9/08", "subgroup", 0.8, {"subclass": 0.8, "maingroup": 0.8, "exact": 0.8}),
        ("maingroup", "H04L 9/00", "maingroup", 0.8, {"subclass": 0.8, "maingroup": 0.8, "exact": 0.0}),
        ("subclass", "H04L", "subclass", 0.8, {"subclass": 0.8, "maingroup": None, "exact": None}),
        ("class", "H04", "class", 0.8, {"subclass": None, "maingroup": None, "exact": None}),
    ],
)
def test_target_level_reports_the_first_codes_ancestor(make, target, display, level, score, share) -> None:
    h = make(store=_strong_store())
    resp = h.classify(target_level=target)
    assert_vote_primary(resp, h)
    assert resp.vote.first_code == "H04L 9/08"  # the vote always runs at the deepest level
    lead = resp.candidates[0]
    assert (lead.display_code, lead.level) == (display, level)
    assert lead.confidence.score == score
    assert lead.vote_share.model_dump() == share


def test_class_level_further_codes_use_class_shares(make) -> None:
    resp = make(store=_strong_store(), neighbour_min_vote_share=0.1).classify(target_level="class")
    # H04 0.8, then H02 and G06 at 0.1 each (computed from the neighbours' labels, not TASK-0008's tables)
    assert [(c.display_code, c.confidence.score) for c in resp.candidates] == [("H04", 0.8), ("H02", 0.1),
                                                                               ("G06", 0.1)]  # fmt: skip
    assert all(c.vote_share.is_empty() for c in resp.candidates)


def test_maingroup_first_code_exact_share_is_its_own_label_share(make) -> None:
    # nobody carries an exact code in H04L 9/00: the vote's first code is the main group, and its exact share
    # (neighbours labelled with H04L 9/00 itself) is 0.0, not None
    neighbours = [_nb(i, _mg_only("H04L 9/00")) for i in range(6)] + [_nb(6 + j, _label(s)) for j, s in
                                                                         enumerate(FILLER[:4])]  # fmt: skip
    h = make(store=FakePatentStore(_corpus(), neighbours))
    resp = h.classify()
    assert_vote_primary(resp, h)
    assert (resp.vote.first_code, resp.vote.first_code_level) == ("H04L 9/00", "maingroup")
    assert resp.vote.first_code_share.model_dump() == {"subclass": 0.6, "maingroup": 0.6, "exact": 0.0}
    assert resp.candidates[0].display_code == "H04L 9/00" and resp.candidates[0].confidence.score == 0.6


@pytest.mark.parametrize(("beta", "classifier", "emitted"), [(0.9996, "neighbour_vote", 0.3),
                                                             (0.9994, "definition_fallback", 0.2999)])
def test_gate_compares_the_rounded_share(make, beta, classifier, emitted) -> None:
    # raw subclass share (2 + beta) / 10: 0.29996 is emitted as 0.3 and passes a 0.3 minimum; 0.29994 -> 0.2999
    h = make(store=FakePatentStore(_corpus(), gate_neighbours()), neighbour_min_vote_share=0.3, neighbour_beta=beta)
    resp = h.classify()
    assert resp.classifier == classifier
    assert resp.vote.first_code_share.subclass == emitted
    if classifier == "neighbour_vote":
        assert resp.vote.first_code_share.subclass >= resp.provenance.vote_settings.min_vote_share
    else:
        assert resp.fallback_reason == "vote_below_minimum"
        assert resp.vote.first_code_share.subclass < resp.provenance.vote_settings.min_vote_share


def test_vote_primary_without_catalogue_embeddings_still_serves(make) -> None:
    h = make(store=_strong_store(), session=FakeSession(embeddings=False))
    resp = h.classify()
    assert_vote_primary(resp, h)
    assert resp.provenance.embedding_run_id is None
    assert any("Catalogue embeddings are unavailable (embeddings_unavailable)" in w for w in resp.warnings)


def test_vote_primary_ignores_use_llm_and_filters_with_a_warning(make) -> None:
    resp = make(store=_strong_store()).classify(use_llm=True, subclass="G06Q")
    assert resp.classifier == "neighbour_vote"
    assert any(w.startswith("use_llm is not applied") for w in resp.warnings)
    assert any(w.startswith("The section/subclass filters apply to definition matching only") for w in resp.warnings)


# --- path 2: weak vote -> fallback ----------------------------------------------------------------------


def test_weak_vote_falls_back_and_keeps_the_vote(make) -> None:
    h = make(store=FakePatentStore(_corpus(), weak_neighbours()))
    resp = h.classify(include_ranked_debug=True)
    assert_fallback(resp, h, "vote_below_minimum")
    assert resp.vote is not None and resp.vote.first_code == "H04L 9/08"
    assert resp.vote.first_code_share.subclass == 0.1 < resp.provenance.vote_settings.min_vote_share
    assert any(w.startswith("Neighbour vote too weak (10.0% for H04L, minimum 20.0%)") for w in resp.warnings)
    assert resp.provenance.patent_corpus is not None and resp.provenance.patent_encoder is not None
    assert resp.candidates, "the definition pipeline produced codes"
    # debug keeps both the definition ranking and the vote ranking
    assert "reranked_candidates" in resp.retrieval_debug and resp.retrieval_debug["vote_exact"][0] == "H04L 9/08"
    assert any(NOT_CALIBRATED_WARNING in w for w in resp.warnings)  # the 1.x confidence warning stays true here


def test_weak_vote_fallback_matches_the_definition_pipeline(make) -> None:
    disabled = make(enabled=False).classify()
    weak = make(store=FakePatentStore(_corpus(), weak_neighbours())).classify()
    assert [c.display_code for c in weak.candidates] == [c.display_code for c in disabled.candidates]
    assert [c.confidence for c in weak.candidates] == [c.confidence for c in disabled.candidates]


# --- path 3: no corpus -> fallback ----------------------------------------------------------------------


def test_no_serving_corpus_falls_back(make) -> None:
    h = make(store=FakePatentStore(None))
    resp = h.classify()
    assert_fallback(resp, h, "no_serving_corpus")
    assert resp.vote is None and resp.provenance.patent_corpus is None
    assert resp.provenance.vote_settings is not None and resp.provenance.patent_encoder is not None
    assert any(w.startswith("No validated patent corpus is serving") for w in resp.warnings)
    assert h.store.searches == []


def test_vote_disabled_is_the_1x_pipeline(make) -> None:
    h = make(enabled=False, store=_strong_store())
    resp = h.classify()
    assert_fallback(resp, h, "neighbour_vote_disabled")
    assert resp.vote is None
    p = resp.provenance
    assert p.patent_encoder is None and p.patent_corpus is None and p.vote_settings is None
    assert h.store.searches == [] and h.patent_encoder.texts == []
    assert h.session.savepoints == 0
    assert any("NEIGHBOUR_VOTE_ENABLED=false" in w for w in resp.warnings)
    assert [ln.confidence_text for ln in resp.summary.classification_lines] == (
        ["confidence: not calibrated"] * len(resp.candidates))


@pytest.mark.parametrize(
    ("kwargs", "reason", "warning"),
    [
        ({"patent_encoder": None}, "patent_encoder_unavailable", "The patent encoder (PaECTER) is not loaded"),
        ({"patent_encoder": FakePatentEncoder(revision="other")}, "patent_encoder_unavailable", "was embedded with"),
        ({"store": FakePatentStore(_corpus(), [_nb(i) for i in range(K)])}, "vote_empty",
         "None of the 10 nearest corpus patents has a resolvable IPC label"),
        ({"store": FakePatentStore(_corpus(), error=NeighbourShortfall("it-test-fake", K, []))}, "vote_failed",
         "Neighbour vote failed: patent search on corpus 'it-test-fake' returned 0 of k=10"),
        ({"store": FakePatentStore(_corpus(), strong_neighbours()[:7])}, "vote_failed", "returned 7 of k=10"),
        ({"store": FakePatentStore(_corpus(), error=psycopg.OperationalError("connection reset"))}, "vote_failed",
         "Neighbour vote failed (OperationalError: connection reset)"),
        ({"store": FakePatentStore(_corpus(), error=sa_exc.OperationalError("SELECT", {}, Exception("server gone")))},
         "vote_failed", "Neighbour vote failed (OperationalError: "),
        ({"patent_encoder": FakePatentEncoder(error=RuntimeError("CUDA error: out of memory"))}, "vote_failed",
         "Neighbour vote failed (_EncoderFailed: RuntimeError: CUDA error: out of memory)"),
        ({"patent_encoder": FakePatentEncoder(error=MemoryError())}, "vote_failed",
         "Neighbour vote failed (_EncoderFailed: MemoryError: "),
        ({"store": _strong_store(), "repo": FakeRepo(drop=("H04L 9/08",))}, "vote_failed",
         "its first code H04L 9/08 is not in taxonomy release 20260101"),
    ],
    ids=["no-encoder", "encoder-revision", "empty", "shortfall", "short-store", "psycopg-error", "sqlalchemy-error",
         "encoder-cuda-error", "encoder-oom", "not-in-catalogue"],
)
def test_other_fallback_reasons(make, kwargs, reason, warning) -> None:
    kwargs.setdefault("store", _strong_store())
    h = make(**kwargs)
    resp = h.classify()
    assert_fallback(resp, h, reason)
    assert resp.vote is None  # only vote_below_minimum keeps the vote
    assert any(warning in w for w in resp.warnings), resp.warnings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"store": FakePatentStore(_corpus(), error=TypeError("search() got an unexpected keyword argument"))},
        {"store": FakePatentStore(_corpus(), error=KeyError("resolved_labels"))},
        {"store": FakePatentStore(_corpus(), error=RuntimeError("a bug, not an outage"))},  # only the encoder's
        {"patent_encoder": FakePatentEncoder(error=TypeError("encode() takes 1 positional argument"))},
    ],
    ids=["store-typeerror", "store-keyerror", "store-runtimeerror", "encoder-typeerror"],
)
def test_programming_errors_in_the_vote_are_not_swallowed(make, kwargs) -> None:
    # B2: a bug must surface (a 500, as on the definition path), never a quiet 200 vote_failed.
    kwargs.setdefault("store", _strong_store())
    h = make(**kwargs)
    error = kwargs.get("patent_encoder", kwargs["store"]).error
    with pytest.raises(type(error)):
        h.classify()
    assert h.keyword_calls == []  # the fallback did not run


# --- 503 patent_corpus_unavailable ----------------------------------------------------------------------


def test_503_only_when_the_vote_cannot_run_and_the_fallback_fails(make) -> None:
    h = make(store=FakePatentStore(None), session=FakeSession(embeddings=False))
    with pytest.raises(PipelineUnavailable) as exc:
        h.classify()
    assert exc.value.code == "patent_corpus_unavailable"
    assert exc.value.details == {"reason": "no_serving_corpus", "fallback_error": "embeddings_unavailable"}


@pytest.mark.parametrize("case", ["disabled", "weak"])
def test_fallback_error_is_returned_as_is_when_the_vote_was_off_or_weak(make, case) -> None:
    store = FakePatentStore(_corpus(), weak_neighbours())
    h = make(enabled=case != "disabled", store=store, session=FakeSession(embeddings=False))
    with pytest.raises(PipelineUnavailable) as exc:
        h.classify()
    assert exc.value.code == "embeddings_unavailable" and exc.value.details is None


def test_missing_taxonomy_release(make) -> None:
    # vote ran: the taxonomy error is returned as-is; vote could not run: wrapped
    with pytest.raises(PipelineUnavailable) as exc:
        make(store=_strong_store(), repo=FakeRepo(active=False)).classify()
    assert exc.value.code == "corpus_unavailable"
    with pytest.raises(PipelineUnavailable) as exc:
        make(store=FakePatentStore(None), repo=FakeRepo(active=False)).classify()
    assert exc.value.code == "patent_corpus_unavailable"
    assert exc.value.details == {"reason": "no_serving_corpus", "fallback_error": "corpus_unavailable"}


# --- units -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("display", "internal"), [("H04L 9/08", "H04L0009080000"), ("H04L 9/00", "H04L0009000000"),
                                                   ("A01B 33/123", "A01B0033123000"), ("G06F 16/10", "G06F0016100000"),
                                                   ("H04L", "H04L"), ("H04", "H04")])  # fmt: skip
def test_internal_code_inverts_display_symbol(display, internal) -> None:
    from app.services.ingestion.wipo import display_symbol

    assert internal_code(display) == internal
    assert display_symbol(internal) == display


def test_db_store_counts_documents_when_the_stored_count_is_zero() -> None:
    class CountSession:
        calls = 0

        def execute(self, stmt, params):
            CountSession.calls += 1
            assert "FROM patent_documents" in str(stmt) and params == {"c": "it-test-fake"}

            class R:
                def scalar_one(self_inner):
                    return 1234

            return R()

    store = DbPatentVoteStore()
    assert store.document_count(CountSession(), _corpus(document_count=0)) == 1234
    assert store.document_count(CountSession(), _corpus(document_count=0)) == 1234
    assert CountSession.calls == 1  # cached per corpus version
    assert store.document_count(CountSession(), _corpus(document_count=99)) == 99


# --- integration: the vote's savepoint on a real session (sakthi with an explicit opt-in) ---------------------
# IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1 uv run pytest -m integration tests/test_pipeline_neighbour.py
# Touches only the patent tables and one it-test-* corpus (deleted afterwards); refuses to run when a real corpus is
# serving, loading or embedded.

IT_PREFIX = "it-test-"


def _it_target() -> str | None:
    import os

    target = os.environ.get("IPC_DB_TARGET")
    if target == "local" or (target == "sakthi" and os.environ.get("IPC_INTEGRATION_ALLOW_SAKTHI") == "1"):
        return target
    return None


def _it_refusal(corpora: list[tuple[str, str, bool]]) -> str | None:
    for corpus_id, state, is_serving in corpora:
        if corpus_id.startswith(IT_PREFIX):
            continue
        if is_serving:
            return f"refusing to run: corpus {corpus_id!r} is serving"
        if state in ("loading", "embedded"):
            return f"refusing to run: corpus {corpus_id!r} is {state!r} (a load may be running)"
    return None


def test_it_refusal_rule() -> None:
    assert _it_refusal([("it-test-a", "loading", True)]) is None
    assert "serving" in (_it_refusal([("hupd-2017", "validated", True)]) or "")
    assert "'embedded'" in (_it_refusal([("uspto", "embedded", False)]) or "")


@pytest.fixture
def it_corpus():
    """An it-test corpus with 3 documents on the configured database; deleted in teardown."""
    target = _it_target()
    if target is None:
        pytest.skip("set IPC_DB_TARGET=local (or IPC_DB_TARGET=sakthi with IPC_INTEGRATION_ALLOW_SAKTHI=1)")
    import uuid
    from datetime import date

    from app.config import get_settings
    from app.db import patents
    from app.db.patents import CorpusSpec, PatentDocument
    from app.db.session import connect_direct

    get_settings.cache_clear()
    assert get_settings().ipc_db_target == target
    conn = connect_direct()
    reason = _it_refusal(conn.execute("SELECT corpus_id, state, is_serving FROM patent_corpora").fetchall())
    conn.commit()
    if reason:
        conn.close()
        pytest.fail(reason)
    corpus_id = f"{IT_PREFIX}{uuid.uuid4().hex[:10]}"
    try:
        patents.create_or_get_corpus(conn, CorpusSpec(corpus_id, "hupd", "compact", PAECTER_MODEL_ID, PAECTER_REVISION))
        patents.upsert_documents(conn, corpus_id, [
            PatentDocument(f"d{i}", f"T{i}", "A synthetic abstract for the savepoint test.", date(2017, 1, 1), None,
                           ["H04L908"], [("H04L", "H04L 9/00", "H04L 9/08")]) for i in range(3)])  # fmt: skip
        yield corpus_id
    finally:
        conn.rollback()
        conn.execute("DELETE FROM patent_corpora WHERE corpus_id = %s AND corpus_id LIKE %s", [corpus_id, IT_PREFIX + "%"])
        conn.commit()
        left = conn.execute("SELECT count(*) FROM patent_documents WHERE corpus_id = %s", [corpus_id]).fetchone()[0]
        conn.close()
        assert left == 0


@pytest.mark.integration
def test_a_database_error_inside_the_vote_leaves_the_request_transaction_usable(it_corpus, tmp_path) -> None:
    from sqlalchemy import text

    from app.db import patents
    from app.db.session import job_session

    corpus_id = it_corpus

    class FailingSearchStore(DbPatentVoteStore):
        """The real store (corpus row and document count read through the request session), whose search then
        hits a real database error on that session, aborting its transaction unless a savepoint protects it."""

        def serving_corpus(self, session):  # noqa: ANN001
            return patents.get_corpus(session, corpus_id)

        def search(self, session, corpus_id_, query_vector, k, ef_search):  # noqa: ANN001
            session.execute(text("SELECT 1 / 0"))  # division_by_zero -> sqlalchemy DataError

    s = Settings(_env_file=None, neighbour_vote_enabled=True, neighbour_k=K, patent_hnsw_ef_search=200)
    p = ClassificationPipeline(s, FakeEncoder(), FakeReranker(), ConfidenceService(tmp_path, "fp-test"),
                               load_mapping(s.field_mapping_path), patent_encoder=FakePatentEncoder(),
                               patent_store=FailingSearchStore())  # fmt: skip
    req = ClassifyRequest(abstract=ABSTRACT, title=TITLE)
    count_sql = text("SELECT count(*) FROM patent_documents WHERE corpus_id = :c")

    with job_session() as session, session.begin():  # the route's transaction shape
        attempt = p._attempt_vote(session, req, {})
        assert attempt.reason == "vote_failed" and attempt.vote is None
        assert any("DataError" in w and "division by zero" in w for w in attempt.warnings), attempt.warnings
        assert attempt.corpus_info is not None and attempt.corpus_info.document_count == 3  # read before the error
        # The definition fallback runs its queries on this same session: it still works.
        assert session.execute(count_sql, {"c": corpus_id}).scalar_one() == 3

    # Control: the same error without the vote's savepoint aborts the transaction for every later query.
    with job_session() as session, session.begin():
        with pytest.raises(sa_exc.DataError):
            session.execute(text("SELECT 1 / 0"))
        with pytest.raises((sa_exc.InternalError, sa_exc.PendingRollbackError)):  # aborted, whichever layer says so
            session.execute(count_sql, {"c": corpus_id})


# --- golden: the vote-disabled path is the 1.x pipeline -------------------------------------------------------
# GOLDEN_1X was recorded by running the 1.x pipeline (pipeline.py at 2aa60b8, before TASK-0010) on these fakes for
# every case below (the review's parity matrix: 5 levels x 3 document sets x 2 max_results x 3 policies x rules on
# and off = 180 cases). Each value is the SHA-256 of the canonical JSON of the response without the fields the 2.0
# contract adds (``contract_free``); GOLDEN_1X_FULL is one complete recorded response, for a readable diff.
# Regenerate only from the 1.x code (scratch script: patch_catalogue() + golden_digest() on that module).

GOLDEN_SPEC = ("The smart meter verifies the identity of a user. Key distribution uses quantum channels. "
               "Encrypted communication follows between substations and control centers. ") * 3
GOLDEN_CLAIMS = "1. A method of key distribution comprising quantum channels.\n2. The method of claim 1, verifying identity."
GOLDEN_DOCS: dict[str, dict[str, str]] = {
    "abstract": {}, "spec": {"specification": GOLDEN_SPEC},
    "spec+claims": {"claims": GOLDEN_CLAIMS, "specification": GOLDEN_SPEC},
}  # fmt: skip
GOLDEN_POLICIES: dict[str, dict[str, Any]] = {
    "default": {}, "lexical": {"enforce_lexical_checks": True},
    "floors": {"min_reranker_score": 1.0, "max_reranker_gap": 2.0},
}  # fmt: skip
GOLDEN_CASES = [(level, docs, n, policy, rules) for level in ("group", "maingroup", "subgroup", "subclass", "class")
                for docs in GOLDEN_DOCS for n in (1, 3) for policy in GOLDEN_POLICIES for rules in (False, True)]
GOLDEN_FULL_CASE = "group|abstract|3|default|rules"


def golden_id(case: tuple) -> str:
    level, docs, n, policy, rules = case
    return f"{level}|{docs}|{n}|{policy}|{'rules' if rules else 'norules'}"


def golden_official(rules: bool) -> dict[str, list[dict[str, Any]]] | None:
    """An unconditional precedence rule (H04L 9/08 over H04L 9/32), so demotions are part of the snapshot."""
    return {internal_code("H04L 9/32"): [_precedence("H04L 9/32", "H04L 9/08", "H04L 9/32")]} if rules else None


def golden_request(case: tuple) -> ClassifyRequest:
    level, docs, n, _, _ = case
    return ClassifyRequest(abstract=ABSTRACT, title=TITLE, target_level=level, max_results=n, include_ranked_debug=True,
                           **GOLDEN_DOCS[docs])  # fmt: skip


def contract_free(d: dict[str, Any]) -> dict[str, Any]:
    """A response dump without what contract 2.0.0 adds (classifier, fallback_reason, vote, the provenance patent
    fields, vote_share), without timings, and without the vote-disabled notice the 2.0 fallback adds."""
    d = {k: v for k, v in d.items() if k not in ("timings_ms", "classifier", "fallback_reason", "vote")}
    d["provenance"] = {k: v for k, v in d["provenance"].items()
                       if k not in ("patent_encoder", "patent_corpus", "vote_settings")}  # fmt: skip
    for key in ("candidates", "alternatives"):
        d[key] = [{k: v for k, v in c.items() if k != "vote_share"} for c in d[key]]
    d["warnings"] = [w for w in d["warnings"] if not w.startswith("Neighbour vote is disabled by the operator")]
    return d


def golden_digest(d: dict[str, Any]) -> str:
    import hashlib
    import json

    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=golden_id)
def test_vote_disabled_path_matches_recorded_1x_output(make, case) -> None:
    level, docs, n, policy, rules = case
    h = make(enabled=False, repo=FakeRepo(official=golden_official(rules)), **GOLDEN_POLICIES[policy])
    resp = h.pipeline.classify(h.session, golden_request(case))
    assert resp.classifier == "definition_fallback" and resp.fallback_reason == "neighbour_vote_disabled"
    got = contract_free(resp.model_dump(mode="json"))
    if golden_id(case) == GOLDEN_FULL_CASE:
        assert got == GOLDEN_1X_FULL
    assert golden_digest(got) == GOLDEN_1X[golden_id(case)]


def test_golden_snapshot_is_not_vacuous() -> None:
    assert len(GOLDEN_1X) == len(GOLDEN_CASES) == 180
    assert len(set(GOLDEN_1X.values())) > 20  # the cases really differ
    assert GOLDEN_1X_FULL["candidates"] and GOLDEN_1X_FULL["ranked_debug"] and GOLDEN_1X_FULL["alternatives"]
    # the precedence rule changes the recorded output in some cases (a demotion is part of the snapshot)
    assert any(GOLDEN_1X[golden_id((*c[:4], True))] != GOLDEN_1X[golden_id((*c[:4], False))] for c in GOLDEN_CASES)


# fmt: off
GOLDEN_1X: dict[str, str] = {'group|abstract|1|default|norules': '0e0b91274500a3bf2eea4eed32f9402d2f0dda30563eb81d7707bf67c2fe2dc0',
 'group|abstract|1|default|rules': 'b6345ca85c0789ee0abd5e21fd08cee54601a786274706051634ffa2d8e163b2',
 'group|abstract|1|lexical|norules': '6f639f1ab7ee630e7ca4c2c14cd6cf10c6351e389721c25fbc514aaeea674d2e',
 'group|abstract|1|lexical|rules': '6f639f1ab7ee630e7ca4c2c14cd6cf10c6351e389721c25fbc514aaeea674d2e',
 'group|abstract|1|floors|norules': '0732823bdd1928e4b50ba3f424deff1f9efeefd6f6ecd1ef763988895a717931',
 'group|abstract|1|floors|rules': 'e7c0e94548231f7fade1b4395482104653773f1eae11f9d2a3232ee89324ec1a',
 'group|abstract|3|default|norules': '0e0b91274500a3bf2eea4eed32f9402d2f0dda30563eb81d7707bf67c2fe2dc0',
 'group|abstract|3|default|rules': 'b6345ca85c0789ee0abd5e21fd08cee54601a786274706051634ffa2d8e163b2',
 'group|abstract|3|lexical|norules': '6f639f1ab7ee630e7ca4c2c14cd6cf10c6351e389721c25fbc514aaeea674d2e',
 'group|abstract|3|lexical|rules': '6f639f1ab7ee630e7ca4c2c14cd6cf10c6351e389721c25fbc514aaeea674d2e',
 'group|abstract|3|floors|norules': 'eea4574d52613c7af846024c7bc2c40b17908df707b5943b09851efc77fa4029',
 'group|abstract|3|floors|rules': 'e7c0e94548231f7fade1b4395482104653773f1eae11f9d2a3232ee89324ec1a',
 'group|spec|1|default|norules': 'd5f3a44347d5d0ee2d220e1248cbb6b70763f0c4d6a9aca341e0de1b4e1d2fda',
 'group|spec|1|default|rules': 'a0ad728f2c26001f8a150df8e98db6571684e569d5e28760577530c7e0307502',
 'group|spec|1|lexical|norules': 'f83678d1c23699c2f22cd8b6116b3d668fc448aa9991a7829ad49134603a83bf',
 'group|spec|1|lexical|rules': 'f83678d1c23699c2f22cd8b6116b3d668fc448aa9991a7829ad49134603a83bf',
 'group|spec|1|floors|norules': 'b302870b3bf74bdc9bdc02b523d6eebc077970860c6ba1358c38337b38d258ac',
 'group|spec|1|floors|rules': '903330086043ddad20ef5388cf9021b95755ca6ff732ec7ce2f705e2005c551a',
 'group|spec|3|default|norules': 'd5f3a44347d5d0ee2d220e1248cbb6b70763f0c4d6a9aca341e0de1b4e1d2fda',
 'group|spec|3|default|rules': 'a0ad728f2c26001f8a150df8e98db6571684e569d5e28760577530c7e0307502',
 'group|spec|3|lexical|norules': 'f83678d1c23699c2f22cd8b6116b3d668fc448aa9991a7829ad49134603a83bf',
 'group|spec|3|lexical|rules': 'f83678d1c23699c2f22cd8b6116b3d668fc448aa9991a7829ad49134603a83bf',
 'group|spec|3|floors|norules': '127472a071b61d70c9da4e151344d3fb893b6a4f15e5805391878a20723cf20b',
 'group|spec|3|floors|rules': '903330086043ddad20ef5388cf9021b95755ca6ff732ec7ce2f705e2005c551a',
 'group|spec+claims|1|default|norules': 'f9a63688cc110f7cbeae775dcd90f47b22f72a088eb91e15310e2376b0d327ec',
 'group|spec+claims|1|default|rules': '014a777af67803612983112ef125eccbc9f1d6e00e3fd86c5e5d87a44d262b18',
 'group|spec+claims|1|lexical|norules': 'd40db988185fe6cec28b4cb0de72ec4b14d960a40f3c16134d7cb484b40669ea',
 'group|spec+claims|1|lexical|rules': 'd40db988185fe6cec28b4cb0de72ec4b14d960a40f3c16134d7cb484b40669ea',
 'group|spec+claims|1|floors|norules': '61b252f0927012327d7bc2691a507060e3c8c90fc002451c36855b108ba97313',
 'group|spec+claims|1|floors|rules': 'df53325bcfabfebf0902492c39da10d605e5f47faf633296fe7a22ab8b20fb6a',
 'group|spec+claims|3|default|norules': 'f9a63688cc110f7cbeae775dcd90f47b22f72a088eb91e15310e2376b0d327ec',
 'group|spec+claims|3|default|rules': '014a777af67803612983112ef125eccbc9f1d6e00e3fd86c5e5d87a44d262b18',
 'group|spec+claims|3|lexical|norules': 'd40db988185fe6cec28b4cb0de72ec4b14d960a40f3c16134d7cb484b40669ea',
 'group|spec+claims|3|lexical|rules': 'd40db988185fe6cec28b4cb0de72ec4b14d960a40f3c16134d7cb484b40669ea',
 'group|spec+claims|3|floors|norules': '535ac97cb62601ddee3fc281a7150d5cb6cc7ea4f2828649182005b28dec85b9',
 'group|spec+claims|3|floors|rules': 'df53325bcfabfebf0902492c39da10d605e5f47faf633296fe7a22ab8b20fb6a',
 'maingroup|abstract|1|default|norules': 'b997ae2e7e54624214264d20d4b486e96b4beff826afe17646b19af47abfc0cb',
 'maingroup|abstract|1|default|rules': 'b997ae2e7e54624214264d20d4b486e96b4beff826afe17646b19af47abfc0cb',
 'maingroup|abstract|1|lexical|norules': '82e992b1f63e53d2eded7cd35636bb0b884b78b38540ec0fda5114b750e3747a',
 'maingroup|abstract|1|lexical|rules': '82e992b1f63e53d2eded7cd35636bb0b884b78b38540ec0fda5114b750e3747a',
 'maingroup|abstract|1|floors|norules': 'dd0582b58513a60375dea5a30a02556dd0ef4968b8c565bb1ad4d502398fc987',
 'maingroup|abstract|1|floors|rules': 'dd0582b58513a60375dea5a30a02556dd0ef4968b8c565bb1ad4d502398fc987',
 'maingroup|abstract|3|default|norules': 'bd6b802f7b3bbf792bca6eb1ee7e2a65049fd589d40e2c208a5a4b0d294a2f03',
 'maingroup|abstract|3|default|rules': 'bd6b802f7b3bbf792bca6eb1ee7e2a65049fd589d40e2c208a5a4b0d294a2f03',
 'maingroup|abstract|3|lexical|norules': '82e992b1f63e53d2eded7cd35636bb0b884b78b38540ec0fda5114b750e3747a',
 'maingroup|abstract|3|lexical|rules': '82e992b1f63e53d2eded7cd35636bb0b884b78b38540ec0fda5114b750e3747a',
 'maingroup|abstract|3|floors|norules': 'eac5164a3a197dbd082fbcb82fd27a2052322d9b0c08a07209e9e86d32b270b1',
 'maingroup|abstract|3|floors|rules': 'eac5164a3a197dbd082fbcb82fd27a2052322d9b0c08a07209e9e86d32b270b1',
 'maingroup|spec|1|default|norules': '07b79188154e093cff592035de571dc80ff5c6410b32a37b4c42aa97ffb9f409',
 'maingroup|spec|1|default|rules': '07b79188154e093cff592035de571dc80ff5c6410b32a37b4c42aa97ffb9f409',
 'maingroup|spec|1|lexical|norules': '2861793316c0567239c6b9c7cbea3d2f77ea9447b90b3800dca000eb8d4f3014',
 'maingroup|spec|1|lexical|rules': '2861793316c0567239c6b9c7cbea3d2f77ea9447b90b3800dca000eb8d4f3014',
 'maingroup|spec|1|floors|norules': '3035d459adebe224354814df2e7adaa9b61acfb18dc84ca9174671b3c3dff52b',
 'maingroup|spec|1|floors|rules': '3035d459adebe224354814df2e7adaa9b61acfb18dc84ca9174671b3c3dff52b',
 'maingroup|spec|3|default|norules': 'ab6b15899453b1612f1ddfe62a431e5e1f6135aabce00fbb4da71f82e7009cc7',
 'maingroup|spec|3|default|rules': 'ab6b15899453b1612f1ddfe62a431e5e1f6135aabce00fbb4da71f82e7009cc7',
 'maingroup|spec|3|lexical|norules': '2861793316c0567239c6b9c7cbea3d2f77ea9447b90b3800dca000eb8d4f3014',
 'maingroup|spec|3|lexical|rules': '2861793316c0567239c6b9c7cbea3d2f77ea9447b90b3800dca000eb8d4f3014',
 'maingroup|spec|3|floors|norules': '373688cffa14a1d67f71143d5b3d40e384cf13905ed115cb5564d051e10ee069',
 'maingroup|spec|3|floors|rules': '373688cffa14a1d67f71143d5b3d40e384cf13905ed115cb5564d051e10ee069',
 'maingroup|spec+claims|1|default|norules': 'fb1def3756cdca43c055626a8cbe3b613cb0db6890933c363b6da9bf49adb6cd',
 'maingroup|spec+claims|1|default|rules': 'fb1def3756cdca43c055626a8cbe3b613cb0db6890933c363b6da9bf49adb6cd',
 'maingroup|spec+claims|1|lexical|norules': '1987def45f0f5d0c8539b9f95a186a949f0fe9c0a8dd8e5d289106e3e043f965',
 'maingroup|spec+claims|1|lexical|rules': '1987def45f0f5d0c8539b9f95a186a949f0fe9c0a8dd8e5d289106e3e043f965',
 'maingroup|spec+claims|1|floors|norules': '8274d09e237d068c3517459300feb774344caa8fe5fe38dea5d5fa8fd2fc40cc',
 'maingroup|spec+claims|1|floors|rules': '8274d09e237d068c3517459300feb774344caa8fe5fe38dea5d5fa8fd2fc40cc',
 'maingroup|spec+claims|3|default|norules': '22ed758b46bc8abfdeea5f8274cf7744ecb57541048790162b4d6b2bd23ee4fe',
 'maingroup|spec+claims|3|default|rules': '22ed758b46bc8abfdeea5f8274cf7744ecb57541048790162b4d6b2bd23ee4fe',
 'maingroup|spec+claims|3|lexical|norules': '1987def45f0f5d0c8539b9f95a186a949f0fe9c0a8dd8e5d289106e3e043f965',
 'maingroup|spec+claims|3|lexical|rules': '1987def45f0f5d0c8539b9f95a186a949f0fe9c0a8dd8e5d289106e3e043f965',
 'maingroup|spec+claims|3|floors|norules': 'de0db86de66a32c1b1b5be97ca8bd1da2a778837a696e7aaabf20276769226ef',
 'maingroup|spec+claims|3|floors|rules': 'de0db86de66a32c1b1b5be97ca8bd1da2a778837a696e7aaabf20276769226ef',
 'subgroup|abstract|1|default|norules': 'ec735cbf0358caedce5f39e5fa9fe65efd5ed6df469dcc166002e832b2368673',
 'subgroup|abstract|1|default|rules': '8a0f4ebdcfc95a97a25b056c4c67b93f04d6aa007987f7c9eac497283cec58a1',
 'subgroup|abstract|1|lexical|norules': '3528b3720d0fa12ab0985060a49f5e485b7c2a6adb1d5fd3ff7ca9a233cae0e0',
 'subgroup|abstract|1|lexical|rules': '3528b3720d0fa12ab0985060a49f5e485b7c2a6adb1d5fd3ff7ca9a233cae0e0',
 'subgroup|abstract|1|floors|norules': 'f1d0f8d98848aab9ff694e67ae57bf36aee20d8b34f01a081e8c7540cd349dd4',
 'subgroup|abstract|1|floors|rules': '6bb20e139336e9690af1a678868ecda79f23657477ffc285e699887d70d20267',
 'subgroup|abstract|3|default|norules': 'ec735cbf0358caedce5f39e5fa9fe65efd5ed6df469dcc166002e832b2368673',
 'subgroup|abstract|3|default|rules': '8a0f4ebdcfc95a97a25b056c4c67b93f04d6aa007987f7c9eac497283cec58a1',
 'subgroup|abstract|3|lexical|norules': '3528b3720d0fa12ab0985060a49f5e485b7c2a6adb1d5fd3ff7ca9a233cae0e0',
 'subgroup|abstract|3|lexical|rules': '3528b3720d0fa12ab0985060a49f5e485b7c2a6adb1d5fd3ff7ca9a233cae0e0',
 'subgroup|abstract|3|floors|norules': 'bcee0535c0f020cdad3a147ce92a3a3ec7e370b0a121d7aa8cec112e29993b21',
 'subgroup|abstract|3|floors|rules': '6bb20e139336e9690af1a678868ecda79f23657477ffc285e699887d70d20267',
 'subgroup|spec|1|default|norules': '2c5441e73e06f130b810098d7884e9de0b6e47f8c8795d7a39b105f96aca6630',
 'subgroup|spec|1|default|rules': '659e55f34c8a145d43ab88c904d467bf46a6856813623872adc7e17d17b3994b',
 'subgroup|spec|1|lexical|norules': '865a455f721029833274441ccf9cc4266fe555d01617494625ca12c488fc5e1e',
 'subgroup|spec|1|lexical|rules': '865a455f721029833274441ccf9cc4266fe555d01617494625ca12c488fc5e1e',
 'subgroup|spec|1|floors|norules': '021b251dda38b9ce406b2bc6c7664bb638bd21be38ede809183c14b630afe0ea',
 'subgroup|spec|1|floors|rules': 'd44bb97aa1013c456fa1e8864cdb30a641713cc57921b26eb13fa288f25da9f6',
 'subgroup|spec|3|default|norules': '2c5441e73e06f130b810098d7884e9de0b6e47f8c8795d7a39b105f96aca6630',
 'subgroup|spec|3|default|rules': '659e55f34c8a145d43ab88c904d467bf46a6856813623872adc7e17d17b3994b',
 'subgroup|spec|3|lexical|norules': '865a455f721029833274441ccf9cc4266fe555d01617494625ca12c488fc5e1e',
 'subgroup|spec|3|lexical|rules': '865a455f721029833274441ccf9cc4266fe555d01617494625ca12c488fc5e1e',
 'subgroup|spec|3|floors|norules': '73772e8ce58e5e742734f4d9be03601c63ebe6213c874151a96d7fca19449271',
 'subgroup|spec|3|floors|rules': 'd44bb97aa1013c456fa1e8864cdb30a641713cc57921b26eb13fa288f25da9f6',
 'subgroup|spec+claims|1|default|norules': 'c8ee1a12189f556ce42ac3920f21a1e5dc02d5239f8d775c8d0e92616c29a155',
 'subgroup|spec+claims|1|default|rules': 'bc94db0c718de408de2b7f1a961441e816bcaf2c69ca39c824ad206aaf648673',
 'subgroup|spec+claims|1|lexical|norules': 'e6273e62f2a1bbf7db54c9fae224e20e77116a7c7d8cd3070ebfd144aa280355',
 'subgroup|spec+claims|1|lexical|rules': 'e6273e62f2a1bbf7db54c9fae224e20e77116a7c7d8cd3070ebfd144aa280355',
 'subgroup|spec+claims|1|floors|norules': 'e634f947e60332d408c99d6b36df5f3267986fb909aba98e5e18475831fa2895',
 'subgroup|spec+claims|1|floors|rules': 'c0cf0d16b0e67f3bdf71b8258abf6661dadd3e38a4a84b151abcbf13acd532c7',
 'subgroup|spec+claims|3|default|norules': 'c8ee1a12189f556ce42ac3920f21a1e5dc02d5239f8d775c8d0e92616c29a155',
 'subgroup|spec+claims|3|default|rules': 'bc94db0c718de408de2b7f1a961441e816bcaf2c69ca39c824ad206aaf648673',
 'subgroup|spec+claims|3|lexical|norules': 'e6273e62f2a1bbf7db54c9fae224e20e77116a7c7d8cd3070ebfd144aa280355',
 'subgroup|spec+claims|3|lexical|rules': 'e6273e62f2a1bbf7db54c9fae224e20e77116a7c7d8cd3070ebfd144aa280355',
 'subgroup|spec+claims|3|floors|norules': '40fa8ab8739f668a314b2076694d12d1fbd2b834f6bd46d3f8ee9171260a9dcc',
 'subgroup|spec+claims|3|floors|rules': 'c0cf0d16b0e67f3bdf71b8258abf6661dadd3e38a4a84b151abcbf13acd532c7',
 'subclass|abstract|1|default|norules': 'd9221abd0bc931d4a80b41ae7d9c87a4fd54f377c8fc1d294121204869a3c821',
 'subclass|abstract|1|default|rules': 'd9221abd0bc931d4a80b41ae7d9c87a4fd54f377c8fc1d294121204869a3c821',
 'subclass|abstract|1|lexical|norules': '81b7c8facd4b4cbce722efeb1c801e9dfd8fed2c05e7aa0a9aa740e786ff7971',
 'subclass|abstract|1|lexical|rules': '81b7c8facd4b4cbce722efeb1c801e9dfd8fed2c05e7aa0a9aa740e786ff7971',
 'subclass|abstract|1|floors|norules': '494f324c09e42b316b160cf5f7192e2c11ef5314249a761018766bc6192f2a9a',
 'subclass|abstract|1|floors|rules': '494f324c09e42b316b160cf5f7192e2c11ef5314249a761018766bc6192f2a9a',
 'subclass|abstract|3|default|norules': '87f30453f89fabd5353fd84aa5b7415d5cb6c2c8a33b8c5d551f1dfffb798b14',
 'subclass|abstract|3|default|rules': '87f30453f89fabd5353fd84aa5b7415d5cb6c2c8a33b8c5d551f1dfffb798b14',
 'subclass|abstract|3|lexical|norules': '81b7c8facd4b4cbce722efeb1c801e9dfd8fed2c05e7aa0a9aa740e786ff7971',
 'subclass|abstract|3|lexical|rules': '81b7c8facd4b4cbce722efeb1c801e9dfd8fed2c05e7aa0a9aa740e786ff7971',
 'subclass|abstract|3|floors|norules': '38808a2c5ba1f0763f36f2de1c763e72ea9023e1a6bf41ccf43a26861511f3f2',
 'subclass|abstract|3|floors|rules': '38808a2c5ba1f0763f36f2de1c763e72ea9023e1a6bf41ccf43a26861511f3f2',
 'subclass|spec|1|default|norules': 'd882e87b58ded4d6856c8ae3d1a60c4e2f9329dd542dde124825b1ec2586b8aa',
 'subclass|spec|1|default|rules': 'd882e87b58ded4d6856c8ae3d1a60c4e2f9329dd542dde124825b1ec2586b8aa',
 'subclass|spec|1|lexical|norules': 'c38557df2ff191b907a8f7049593824c7387891b032b7b29c83d21eef7c5e731',
 'subclass|spec|1|lexical|rules': 'c38557df2ff191b907a8f7049593824c7387891b032b7b29c83d21eef7c5e731',
 'subclass|spec|1|floors|norules': 'd1b1fdd052f65d2c19c49ef5259cd56ba4ecb258c8659df64699d63d39da2f11',
 'subclass|spec|1|floors|rules': 'd1b1fdd052f65d2c19c49ef5259cd56ba4ecb258c8659df64699d63d39da2f11',
 'subclass|spec|3|default|norules': '5c54e42d17737322c4029406a27c57ec072c0139b0cd9bfd5f0c3dbeabd7cb4a',
 'subclass|spec|3|default|rules': '5c54e42d17737322c4029406a27c57ec072c0139b0cd9bfd5f0c3dbeabd7cb4a',
 'subclass|spec|3|lexical|norules': 'c38557df2ff191b907a8f7049593824c7387891b032b7b29c83d21eef7c5e731',
 'subclass|spec|3|lexical|rules': 'c38557df2ff191b907a8f7049593824c7387891b032b7b29c83d21eef7c5e731',
 'subclass|spec|3|floors|norules': '48454c2bf6f2dcb434b0dbee6ee9e7117a629f4a062a53383f67797b39c938ca',
 'subclass|spec|3|floors|rules': '48454c2bf6f2dcb434b0dbee6ee9e7117a629f4a062a53383f67797b39c938ca',
 'subclass|spec+claims|1|default|norules': 'a1f186ded2f9bc0c4a38da8bf85b2166b2ce86e910df0d17ba5947d80f41344d',
 'subclass|spec+claims|1|default|rules': 'a1f186ded2f9bc0c4a38da8bf85b2166b2ce86e910df0d17ba5947d80f41344d',
 'subclass|spec+claims|1|lexical|norules': 'a805eb6fa69c9c1b65261900df48b3f3fd71c1bf9646d780dcb341e4ffc56fb4',
 'subclass|spec+claims|1|lexical|rules': 'a805eb6fa69c9c1b65261900df48b3f3fd71c1bf9646d780dcb341e4ffc56fb4',
 'subclass|spec+claims|1|floors|norules': 'cde11a03f52ce3acb2902d78bf58fad11d1ecf8c1e2d5131e0ca7d18f77fda7f',
 'subclass|spec+claims|1|floors|rules': 'cde11a03f52ce3acb2902d78bf58fad11d1ecf8c1e2d5131e0ca7d18f77fda7f',
 'subclass|spec+claims|3|default|norules': '42a367b9f01cfba0ee7a12e91e4f55e3da48e3bcc334f362886a8a6862f9d6b9',
 'subclass|spec+claims|3|default|rules': '42a367b9f01cfba0ee7a12e91e4f55e3da48e3bcc334f362886a8a6862f9d6b9',
 'subclass|spec+claims|3|lexical|norules': 'a805eb6fa69c9c1b65261900df48b3f3fd71c1bf9646d780dcb341e4ffc56fb4',
 'subclass|spec+claims|3|lexical|rules': 'a805eb6fa69c9c1b65261900df48b3f3fd71c1bf9646d780dcb341e4ffc56fb4',
 'subclass|spec+claims|3|floors|norules': '117c30bc830fa9f47c37340406efdef76ef34393556643d7f3bf3c97620504a6',
 'subclass|spec+claims|3|floors|rules': '117c30bc830fa9f47c37340406efdef76ef34393556643d7f3bf3c97620504a6',
 'class|abstract|1|default|norules': '59ac45a2f9b2566bdb23030244c36872daba57a7eb8a0a7d8d58695a044822ab',
 'class|abstract|1|default|rules': '59ac45a2f9b2566bdb23030244c36872daba57a7eb8a0a7d8d58695a044822ab',
 'class|abstract|1|lexical|norules': 'e05c23318109ecdd68716eb388f7bac6bb0728fbce616bfeff8246964dbca18c',
 'class|abstract|1|lexical|rules': 'e05c23318109ecdd68716eb388f7bac6bb0728fbce616bfeff8246964dbca18c',
 'class|abstract|1|floors|norules': 'd2561001ca0724ba92c52a7eaf4c4095c0d07730344400bc1022c8cb97aece91',
 'class|abstract|1|floors|rules': 'd2561001ca0724ba92c52a7eaf4c4095c0d07730344400bc1022c8cb97aece91',
 'class|abstract|3|default|norules': 'c0bd52653c7b5a20bf7c6141374ad4bb3c5939d8934fa2fabdff736521f4ac4a',
 'class|abstract|3|default|rules': 'c0bd52653c7b5a20bf7c6141374ad4bb3c5939d8934fa2fabdff736521f4ac4a',
 'class|abstract|3|lexical|norules': 'af462223fb6ab00b4d50928259302568fe7de872f8aa5c277bd81ffc59908aed',
 'class|abstract|3|lexical|rules': 'af462223fb6ab00b4d50928259302568fe7de872f8aa5c277bd81ffc59908aed',
 'class|abstract|3|floors|norules': '60f62f2b9afdc2e33269588d2151cddff209e4df26316bb4ddc8d2c7d0526005',
 'class|abstract|3|floors|rules': '60f62f2b9afdc2e33269588d2151cddff209e4df26316bb4ddc8d2c7d0526005',
 'class|spec|1|default|norules': '040682900a5c151dbfe8a3822f6163c9f4f68b4130288d72e518e7571ee34e1e',
 'class|spec|1|default|rules': '040682900a5c151dbfe8a3822f6163c9f4f68b4130288d72e518e7571ee34e1e',
 'class|spec|1|lexical|norules': '09f510b88f6db66c2ca829d3391a5f954b55bda3b862851b765e137c31131a5e',
 'class|spec|1|lexical|rules': '09f510b88f6db66c2ca829d3391a5f954b55bda3b862851b765e137c31131a5e',
 'class|spec|1|floors|norules': '5b60b21c4bb0b69fa334cdd6def0399ed407196c098f6bb2aedd1430b94edb73',
 'class|spec|1|floors|rules': '5b60b21c4bb0b69fa334cdd6def0399ed407196c098f6bb2aedd1430b94edb73',
 'class|spec|3|default|norules': 'e6c3fb96dd0190d20fbe8312b3cda0f0054a677bdeb5ee876f81a367f9ff2388',
 'class|spec|3|default|rules': 'e6c3fb96dd0190d20fbe8312b3cda0f0054a677bdeb5ee876f81a367f9ff2388',
 'class|spec|3|lexical|norules': '85251b6c5f7dc0c122ebc59fd055525416184f4167091e39a1bbb26a89792bff',
 'class|spec|3|lexical|rules': '85251b6c5f7dc0c122ebc59fd055525416184f4167091e39a1bbb26a89792bff',
 'class|spec|3|floors|norules': '58ac1735d2e2937c9d4957adb5e335950db8366eca78ece834766ff73f8d9ef0',
 'class|spec|3|floors|rules': '58ac1735d2e2937c9d4957adb5e335950db8366eca78ece834766ff73f8d9ef0',
 'class|spec+claims|1|default|norules': 'cb90ba5efd695c8a07b7ccd6c2e5782d14e099457db21399fc09262072f43af8',
 'class|spec+claims|1|default|rules': 'cb90ba5efd695c8a07b7ccd6c2e5782d14e099457db21399fc09262072f43af8',
 'class|spec+claims|1|lexical|norules': 'ae8267cdafa6fb33d61834ecbd8b0462eaac75315026765f86ad2d613d078166',
 'class|spec+claims|1|lexical|rules': 'ae8267cdafa6fb33d61834ecbd8b0462eaac75315026765f86ad2d613d078166',
 'class|spec+claims|1|floors|norules': '7176949b37267dd66585467f91e7dcf3872a1d8a59e0abbb3a0bf12ca2cc825e',
 'class|spec+claims|1|floors|rules': '7176949b37267dd66585467f91e7dcf3872a1d8a59e0abbb3a0bf12ca2cc825e',
 'class|spec+claims|3|default|norules': '7ebf6f43bcf0468f0cc0b39f29554935de33179b0af379a3b64484604b7f6274',
 'class|spec+claims|3|default|rules': '7ebf6f43bcf0468f0cc0b39f29554935de33179b0af379a3b64484604b7f6274',
 'class|spec+claims|3|lexical|norules': '9d70510b09858628b196bdc9d1766e1303e08a3678edd43c604cbadc4359b6bd',
 'class|spec+claims|3|lexical|rules': '9d70510b09858628b196bdc9d1766e1303e08a3678edd43c604cbadc4359b6bd',
 'class|spec+claims|3|floors|norules': '26589a93aa96026af7933da7b252607273fa6dcc415f9eb7f6997c9ab6b77d51',
 'class|spec+claims|3|floors|rules': '26589a93aa96026af7933da7b252607273fa6dcc415f9eb7f6997c9ab6b77d51'}
GOLDEN_1X_FULL: dict[str, Any] = {'alternatives': [{'code': 'H04L0009080000',
                   'confidence': {'calibrator_id': None,
                                  'detail': 'No validated calibrator artifact is active.',
                                  'interpretation': None,
                                  'score': None,
                                  'status': 'not_calibrated'},
                   'display_code': 'H04L 9/08',
                   'level': 'subgroup',
                   'reason': 'Relevance well below the best supported candidate; listed for review only.',
                   'scores': {'cosine_similarity': 0.86,
                              'fused_rank': 5,
                              'keyword_rank': 5,
                              'keyword_score': 0.2,
                              'reranker_score': 2.0,
                              'reranker_source': 'abstract',
                              'rrf_score': 0.03076923076923077,
                              'vector_rank': 5},
                   'supporting_passages': [{'claim_number': None,
                                            'end': 73,
                                            'evidence_type': 'retrieval_evidence_requires_review',
                                            'quote': 'Quantum keys are distributed among smart grid nodes for key '
                                                     'distribution.',
                                            'relevance': 1.0,
                                            'section': None,
                                            'source': 'abstract',
                                            'start': 0}],
                   'title': 'Key distribution',
                   'unresolved_ambiguity': ['Scope check (lexical heuristic, needs review): Subclass scope (H04L: '
                                            'TRANSMISSION OF DIGITAL INFORMATION) not evidenced lexically; none of '
                                            'its distinctive terms appear in the reviewed text: transmission, '
                                            'digital.']},
                  {'code': 'G06Q0050000000',
                   'confidence': {'calibrator_id': None,
                                  'detail': 'No validated calibrator artifact is active.',
                                  'interpretation': None,
                                  'score': None,
                                  'status': 'not_calibrated'},
                   'display_code': 'G06Q 50/00',
                   'level': 'maingroup',
                   'reason': 'Relevance well below the best supported candidate; listed for review only.',
                   'scores': {'cosine_similarity': 0.9,
                              'fused_rank': 1,
                              'keyword_rank': 1,
                              'keyword_score': 1.0,
                              'reranker_score': 1.0,
                              'reranker_source': 'abstract',
                              'rrf_score': 0.03278688524590164,
                              'vector_rank': 1},
                   'supporting_passages': [{'claim_number': None,
                                            'end': 148,
                                            'evidence_type': 'retrieval_evidence_requires_review',
                                            'quote': 'Each node verifying the identity of a user before encrypted '
                                                     'communication.',
                                            'relevance': 1.0,
                                            'section': None,
                                            'source': 'abstract',
                                            'start': 74}],
                   'title': 'Information and communication technology specially adapted for business sectors',
                   'unresolved_ambiguity': ['Scope check (lexical heuristic, needs review): Subclass scope (G06Q: '
                                            'INFORMATION AND COMMUNICATION TECHNOLOGY SPECIALLY ADAPTED FOR '
                                            'ADMINISTRATIVE PURPOSES) not evidenced lexically; none of its '
                                            'distinctive terms appear in the reviewed text: technology, '
                                            'administrative, purposes.']},
                  {'code': 'G06Q0050060000',
                   'confidence': {'calibrator_id': None,
                                  'detail': 'No validated calibrator artifact is active.',
                                  'interpretation': None,
                                  'score': None,
                                  'status': 'not_calibrated'},
                   'display_code': 'G06Q 50/06',
                   'level': 'subgroup',
                   'reason': 'Relevance well below the best supported candidate; listed for review only.',
                   'scores': {'cosine_similarity': 0.89,
                              'fused_rank': 2,
                              'keyword_rank': 2,
                              'keyword_score': 0.5,
                              'reranker_score': 1.0,
                              'reranker_source': 'abstract',
                              'rrf_score': 0.03225806451612903,
                              'vector_rank': 2},
                   'supporting_passages': [{'claim_number': None,
                                            'end': 148,
                                            'evidence_type': 'retrieval_evidence_requires_review',
                                            'quote': 'Each node verifying the identity of a user before encrypted '
                                                     'communication.',
                                            'relevance': 1.0,
                                            'section': None,
                                            'source': 'abstract',
                                            'start': 74}],
                   'title': 'Energy or water supply',
                   'unresolved_ambiguity': ['Scope check (lexical heuristic, needs review): Subclass scope (G06Q: '
                                            'INFORMATION AND COMMUNICATION TECHNOLOGY SPECIALLY ADAPTED FOR '
                                            'ADMINISTRATIVE PURPOSES) not evidenced lexically; none of its '
                                            'distinctive terms appear in the reviewed text: technology, '
                                            'administrative, purposes.',
                                            'Scope check (lexical heuristic, needs review): Defining feature(s) of '
                                            'this subgroup not found verbatim in the reviewed text: energy, water, '
                                            'supply.']},
                  {'code': 'H02J0013000000',
                   'confidence': {'calibrator_id': None,
                                  'detail': 'No validated calibrator artifact is active.',
                                  'interpretation': None,
                                  'score': None,
                                  'status': 'not_calibrated'},
                   'display_code': 'H02J 13/00',
                   'level': 'maingroup',
                   'reason': 'Relevance well below the best supported candidate; listed for review only.',
                   'scores': {'cosine_similarity': 0.88,
                              'fused_rank': 3,
                              'keyword_rank': 3,
                              'keyword_score': 0.3333333333333333,
                              'reranker_score': 1.0,
                              'reranker_source': 'abstract',
                              'rrf_score': 0.031746031746031744,
                              'vector_rank': 3},
                   'supporting_passages': [{'claim_number': None,
                                            'end': 73,
                                            'evidence_type': 'retrieval_evidence_requires_review',
                                            'quote': 'Quantum keys are distributed among smart grid nodes for key '
                                                     'distribution.',
                                            'relevance': 1.0,
                                            'section': None,
                                            'source': 'abstract',
                                            'start': 0}],
                   'title': 'Circuit arrangements for providing remote indication of network conditions',
                   'unresolved_ambiguity': []},
                  {'code': 'H04L0009000000',
                   'confidence': {'calibrator_id': None,
                                  'detail': 'No validated calibrator artifact is active.',
                                  'interpretation': None,
                                  'score': None,
                                  'status': 'not_calibrated'},
                   'display_code': 'H04L 9/00',
                   'level': 'maingroup',
                   'reason': 'Relevance well below the best supported candidate; listed for review only.',
                   'scores': {'cosine_similarity': 0.87,
                              'fused_rank': 4,
                              'keyword_rank': 4,
                              'keyword_score': 0.25,
                              'reranker_score': 1.0,
                              'reranker_source': 'abstract',
                              'rrf_score': 0.03125,
                              'vector_rank': 4},
                   'supporting_passages': [{'claim_number': None,
                                            'end': 148,
                                            'evidence_type': 'retrieval_evidence_requires_review',
                                            'quote': 'Each node verifying the identity of a user before encrypted '
                                                     'communication.',
                                            'relevance': 1.0,
                                            'section': None,
                                            'source': 'abstract',
                                            'start': 74}],
                   'title': 'Arrangements for secret or secure communications',
                   'unresolved_ambiguity': ['Scope check (lexical heuristic, needs review): Subclass scope (H04L: '
                                            'TRANSMISSION OF DIGITAL INFORMATION) not evidenced lexically; none of '
                                            'its distinctive terms appear in the reviewed text: transmission, '
                                            'digital.']},
                  {'code': 'H04L0067000000',
                   'confidence': {'calibrator_id': None,
                                  'detail': 'No validated calibrator artifact is active.',
                                  'interpretation': None,
                                  'score': None,
                                  'status': 'not_calibrated'},
                   'display_code': 'H04L 67/00',
                   'level': 'maingroup',
                   'reason': 'Relevance well below the best supported candidate; listed for review only.',
                   'scores': {'cosine_similarity': 0.84,
                              'fused_rank': 7,
                              'keyword_rank': 7,
                              'keyword_score': 0.14285714285714285,
                              'reranker_score': 1.0,
                              'reranker_source': 'abstract',
                              'rrf_score': 0.029850746268656716,
                              'vector_rank': 7},
                   'supporting_passages': [{'claim_number': None,
                                            'end': 148,
                                            'evidence_type': 'retrieval_evidence_requires_review',
                                            'quote': 'Each node verifying the identity of a user before encrypted '
                                                     'communication.',
                                            'relevance': 1.0,
                                            'section': None,
                                            'source': 'abstract',
                                            'start': 74}],
                   'title': 'Network arrangements or protocols for supporting network services',
                   'unresolved_ambiguity': ['Scope check (lexical heuristic, needs review): Subclass scope (H04L: '
                                            'TRANSMISSION OF DIGITAL INFORMATION) not evidenced lexically; none of '
                                            'its distinctive terms appear in the reviewed text: transmission, '
                                            'digital.']},
                  {'code': 'H04L0067120000',
                   'confidence': {'calibrator_id': None,
                                  'detail': 'No validated calibrator artifact is active.',
                                  'interpretation': None,
                                  'score': None,
                                  'status': 'not_calibrated'},
                   'display_code': 'H04L 67/12',
                   'level': 'subgroup',
                   'reason': 'Relevance well below the best supported candidate; listed for review only.',
                   'scores': {'cosine_similarity': 0.83,
                              'fused_rank': 8,
                              'keyword_rank': 8,
                              'keyword_score': 0.125,
                              'reranker_score': 1.0,
                              'reranker_source': 'abstract',
                              'rrf_score': 0.029411764705882353,
                              'vector_rank': 8},
                   'supporting_passages': [{'claim_number': None,
                                            'end': 148,
                                            'evidence_type': 'retrieval_evidence_requires_review',
                                            'quote': 'Each node verifying the identity of a user before encrypted '
                                                     'communication.',
                                            'relevance': 1.0,
                                            'section': None,
                                            'source': 'abstract',
                                            'start': 74}],
                   'title': 'adapted for proprietary or special-purpose networking environments',
                   'unresolved_ambiguity': ['Scope check (lexical heuristic, needs review): Subclass scope (H04L: '
                                            'TRANSMISSION OF DIGITAL INFORMATION) not evidenced lexically; none of '
                                            'its distinctive terms appear in the reviewed text: transmission, '
                                            'digital.',
                                            'Scope check (lexical heuristic, needs review): Defining feature(s) of '
                                            'this subgroup not found verbatim in the reviewed text: proprietary, '
                                            'special-purpose, networking, environments.']}],
 'candidates': [{'code': 'H04L0009320000',
                 'confidence': {'calibrator_id': None,
                                'detail': 'No validated calibrator artifact is active.',
                                'interpretation': None,
                                'score': None,
                                'status': 'not_calibrated'},
                 'depends_on_optional_documents': False,
                 'display_code': 'H04L 9/32',
                 'level': 'subgroup',
                 'missing_defining_terms': ['authority'],
                 'origin': ['keyword:abstract', 'vector:abstract'],
                 'original_title': None,
                 'parent_path': [{'code': 'H', 'display_code': 'H', 'level': 'section', 'title': 'ELECTRICITY'},
                                 {'code': 'H04',
                                  'display_code': 'H04',
                                  'level': 'class',
                                  'title': 'ELECTRIC COMMUNICATION TECHNIQUE'},
                                 {'code': 'H04L',
                                  'display_code': 'H04L',
                                  'level': 'subclass',
                                  'title': 'TRANSMISSION OF DIGITAL INFORMATION'},
                                 {'code': 'H04L0009000000',
                                  'display_code': 'H04L 9/00',
                                  'level': 'maingroup',
                                  'title': 'Arrangements for secret or secure communications'}],
                 'rationale': 'The abstract passage shares the defining terms (verifying, identity, user) of H04L '
                              '9/32 "including means for verifying the identity or authority of a user" within '
                              'Arrangements for secret or secure communications; ranked by the cross-encoder against '
                              "the candidate's scope wording.",
                 'references': [],
                 'scores': {'cosine_similarity': 0.85,
                            'fused_rank': 6,
                            'keyword_rank': 6,
                            'keyword_score': 0.16666666666666666,
                            'reranker_score': 4.0,
                            'reranker_source': 'abstract',
                            'rrf_score': 0.030303030303030304,
                            'vector_rank': 6},
                 'supporting_passages': [{'claim_number': None,
                                          'end': 148,
                                          'evidence_type': 'retrieval_evidence_requires_review',
                                          'quote': 'Each node verifying the identity of a user before encrypted '
                                                   'communication.',
                                          'relevance': 4.0,
                                          'section': None,
                                          'source': 'abstract',
                                          'start': 74}],
                 'title': 'including means for verifying the identity or authority of a user',
                 'unresolved_ambiguity': ['Retrieval evidence requiring review: sentence similarity does not prove '
                                          'that all classification conditions are met.',
                                          'Scope check (lexical heuristic, needs review): Subclass scope (H04L: '
                                          'TRANSMISSION OF DIGITAL INFORMATION) not evidenced lexically; none of its '
                                          'distinctive terms appear in the reviewed text: transmission, digital.',
                                          'WIPO precedence: WIPO scheme note for H04L 9/32, edition 2026.01: "H04L '
                                          '9/08 takes precedence over H04L 9/32.". H04L 9/08 was ranked but not '
                                          'supported by the evidence policy; if its subject matter is also '
                                          'disclosed, it takes precedence over H04L 9/32.',
                                          'Abstract-only candidate review; applicable scope notes and priority rules '
                                          'still need verification against the official scheme.',
                                          'Version-matched WIPO notes, definitions and reference wording apply to '
                                          'this code; confirm any conditional or precedence rule applies to this '
                                          'disclosure.',
                                          'WIPO precedence rule for H04L 9/32: H04L 9/08 takes precedence over H04L '
                                          '9/32.',
                                          'Title terms not found in the reviewed text: authority.']}],
 'degraded_mode': None,
 'extracted_terms': [{'source': 'abstract', 'term': 'key distribution', 'weight': 3.2},
                     {'source': 'abstract', 'term': 'smart grid', 'weight': 3.2},
                     {'source': 'abstract', 'term': 'quantum key distribution', 'weight': 2.2},
                     {'source': 'abstract', 'term': 'distributed among smart', 'weight': 2.2},
                     {'source': 'abstract', 'term': 'among smart grid', 'weight': 2.2},
                     {'source': 'abstract', 'term': 'smart grid nodes', 'weight': 2.2},
                     {'source': 'abstract', 'term': 'user before encrypted', 'weight': 2.2},
                     {'source': 'abstract', 'term': 'before encrypted communication', 'weight': 2.2},
                     {'source': 'abstract', 'term': 'quantum key', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'quantum keys', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'distributed among', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'among smart', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'grid nodes', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'node verifying', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'user before', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'before encrypted', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'encrypted communication', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'smart meters', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'control centers', 'weight': 1.6},
                     {'source': 'abstract', 'term': 'identity', 'weight': 1.0},
                     {'source': 'abstract', 'term': 'system', 'weight': 1.0},
                     {'source': 'abstract', 'term': 'substations', 'weight': 1.0},
                     {'source': 'abstract', 'term': 'inverters', 'weight': 1.0}],
 'field_of_invention': {'additional_labels': [],
                        'label': 'COMMUNICATION',
                        'mapping_key': 'H04',
                        'mapping_version': '2',
                        'rationale': 'Primary supported candidate H04L 9/32 (including means for verifying the '
                                     'identity or authority of a user) is in class H04 (ELECTRIC COMMUNICATION '
                                     'TECHNIQUE); mapped via field mapping v2.'},
 'input_coverage': {'abstract': 'complete', 'claims': 'not_supplied', 'details': {}, 'specification': 'not_supplied'},
 'provenance': {'calibrator_id': None,
                'embedding_model': 'fake/bge',
                'embedding_provider': 'fake',
                'embedding_run_id': 'run-test',
                'encoding_policy': {'abstract_max_tokens': 2048,
                                    'dimension': 8,
                                    'document_template': '{text}',
                                    'dtype': 'float32',
                                    'normalized': True,
                                    'query_chunk_tokens': 1024,
                                    'query_template': '{text}',
                                    'taxonomy_max_tokens': 512},
                'field_mapping_version': '2',
                'llm_model': None,
                'model_revision': 'rev-1',
                'reranker_model': 'fake/reranker',
                'reranker_revision': 'rr-1',
                'retrieval_settings': {'apply_place_rules': False,
                                       'apply_precedence_rules': True,
                                       'claims_retrieval': 'excluded',
                                       'enforce_lexical_checks': False,
                                       'expansion_max_siblings': 4,
                                       'expansion_top_n': 20,
                                       'fusion': 'rrf_two_level_weighted_source_caps',
                                       'hierarchical_retrieval': True,
                                       'hierarchical_subclass_limit': 30,
                                       'hnsw_ef_search': 500,
                                       'keyword_max_terms': 24,
                                       'keyword_mode': 'postgresql_fulltext_ts_rank_cd',
                                       'keyword_pool': 100,
                                       'max_query_views': 24,
                                       'max_reranker_gap': 0.5,
                                       'min_evidence_score': -5.0,
                                       'min_reranker_score': None,
                                       'per_source_candidate_cap': 100,
                                       'rerank_official_context': False,
                                       'rerank_pool': 100,
                                       'routing_group_pool': 300,
                                       'rrf_k': 60,
                                       'subclass_routing': 'group_vote',
                                       'vector_pool': 100,
                                       'vector_search_mode': 'hnsw'},
                'scope_formatter': 'scope-v3',
                'source_snapshot': {'ipc_scheme': 'sha256:test'},
                'taxonomy_version': '20260101'},
 'ranked_debug': [{'application_context_only': False,
                   'best_evidence_score': 4.0,
                   'code': 'H04L0009320000',
                   'defining_terms_ok': True,
                   'display_code': 'H04L 9/32',
                   'maingroup_code': 'H04L0009000000',
                   'outcome': 'supported',
                   'record_type': 'subgroup',
                   'reranker_score': 4.0,
                   'subclass_scope_ok': False,
                   'title': 'including means for verifying the identity or authority of a user'},
                  {'application_context_only': False,
                   'best_evidence_score': 1.0,
                   'code': 'H04L0009080000',
                   'defining_terms_ok': True,
                   'display_code': 'H04L 9/08',
                   'maingroup_code': 'H04L0009000000',
                   'outcome': 'alternative: Relevance well below the best supported candidate; listed for review '
                              'only.',
                   'record_type': 'subgroup',
                   'reranker_score': 2.0,
                   'subclass_scope_ok': False,
                   'title': 'Key distribution'},
                  {'application_context_only': False,
                   'best_evidence_score': 1.0,
                   'code': 'G06Q0050000000',
                   'defining_terms_ok': True,
                   'display_code': 'G06Q 50/00',
                   'maingroup_code': 'G06Q0050000000',
                   'outcome': 'alternative: Relevance well below the best supported candidate; listed for review '
                              'only.',
                   'record_type': 'maingroup',
                   'reranker_score': 1.0,
                   'subclass_scope_ok': False,
                   'title': 'Information and communication technology specially adapted for business sectors'},
                  {'application_context_only': False,
                   'best_evidence_score': 1.0,
                   'code': 'G06Q0050060000',
                   'defining_terms_ok': False,
                   'display_code': 'G06Q 50/06',
                   'maingroup_code': 'G06Q0050000000',
                   'outcome': 'alternative: Relevance well below the best supported candidate; listed for review '
                              'only.',
                   'record_type': 'subgroup',
                   'reranker_score': 1.0,
                   'subclass_scope_ok': False,
                   'title': 'Energy or water supply'},
                  {'application_context_only': False,
                   'best_evidence_score': 1.0,
                   'code': 'H02J0013000000',
                   'defining_terms_ok': True,
                   'display_code': 'H02J 13/00',
                   'maingroup_code': 'H02J0013000000',
                   'outcome': 'alternative: Relevance well below the best supported candidate; listed for review '
                              'only.',
                   'record_type': 'maingroup',
                   'reranker_score': 1.0,
                   'subclass_scope_ok': True,
                   'title': 'Circuit arrangements for providing remote indication of network conditions'},
                  {'application_context_only': False,
                   'best_evidence_score': 1.0,
                   'code': 'H04L0009000000',
                   'defining_terms_ok': True,
                   'display_code': 'H04L 9/00',
                   'maingroup_code': 'H04L0009000000',
                   'outcome': 'alternative: Relevance well below the best supported candidate; listed for review '
                              'only.',
                   'record_type': 'maingroup',
                   'reranker_score': 1.0,
                   'subclass_scope_ok': False,
                   'title': 'Arrangements for secret or secure communications'},
                  {'application_context_only': False,
                   'best_evidence_score': 1.0,
                   'code': 'H04L0067000000',
                   'defining_terms_ok': True,
                   'display_code': 'H04L 67/00',
                   'maingroup_code': 'H04L0067000000',
                   'outcome': 'alternative: Relevance well below the best supported candidate; listed for review '
                              'only.',
                   'record_type': 'maingroup',
                   'reranker_score': 1.0,
                   'subclass_scope_ok': False,
                   'title': 'Network arrangements or protocols for supporting network services'},
                  {'application_context_only': False,
                   'best_evidence_score': 1.0,
                   'code': 'H04L0067120000',
                   'defining_terms_ok': False,
                   'display_code': 'H04L 67/12',
                   'maingroup_code': 'H04L0067000000',
                   'outcome': 'alternative: Relevance well below the best supported candidate; listed for review '
                              'only.',
                   'record_type': 'subgroup',
                   'reranker_score': 1.0,
                   'subclass_scope_ok': False,
                   'title': 'adapted for proprietary or special-purpose networking environments'}],
 'retrieval_debug': {'reranked_candidates': ['H04L 9/32',
                                             'H04L 9/08',
                                             'G06Q 50/00',
                                             'G06Q 50/06',
                                             'H02J 13/00',
                                             'H04L 9/00',
                                             'H04L 67/00',
                                             'H04L 67/12'],
                     'stage1_subclasses': ['H04L', 'G06Q', 'H02J'],
                     'stage2_group_candidates': ['G06Q 50/00',
                                                 'G06Q 50/06',
                                                 'H02J 13/00',
                                                 'H04L 9/00',
                                                 'H04L 9/08',
                                                 'H04L 9/32',
                                                 'H04L 67/00',
                                                 'H04L 67/12']},
 'status': 'needs_review',
 'summary': {'classification_cell_text': 'H04L 9/32 — confidence: not calibrated',
             'classification_lines': [{'confidence_text': 'confidence: not calibrated', 'display_code': 'H04L 9/32'}],
             'field_of_invention': 'COMMUNICATION',
             'markdown': '| Field Of Invention | COMMUNICATION |\n'
                         '| :--- | :--- |\n'
                         '| Classification (IPC) | H04L 9/32 — confidence: not calibrated |\n'},
 'taxonomy_version': '20260101',
 'warnings': ['Vector retrieval for abstract returned 8 of 300 requested in this stage.',
              'Vector retrieval for abstract returned 8 of 100 requested in this stage.',
              'Abstract-only inference: not an official classification; scope and rule checks need review.',
              'Confidence: not calibrated — no validated calibrator is active for this pipeline; numeric confidence '
              'is null.']}
# fmt: on
