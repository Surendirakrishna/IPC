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
import pytest

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

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts += list(texts)
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
        monkeypatch.setattr(pl, "repo", fake_repo)

        def keyword_search(session, *, version, query, limit, record_types, section=None, subclass=None,
                           subclasses=None):  # fmt: skip
            h.keyword_calls.append((record_types, subclasses))
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

        monkeypatch.setattr(pl, "keyword_search", keyword_search)
        monkeypatch.setattr(pl, "vector_search", vector_search)
        monkeypatch.setattr(pl, "expand", expand)
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
        ({"store": FakePatentStore(_corpus(), error=RuntimeError("connection reset"))}, "vote_failed",
         "Neighbour vote failed (RuntimeError: connection reset)"),
        ({"store": _strong_store(), "repo": FakeRepo(drop=("H04L 9/08",))}, "vote_failed",
         "its first code H04L 9/08 is not in taxonomy release 20260101"),
    ],
    ids=["no-encoder", "encoder-revision", "empty", "shortfall", "short-store", "search-error", "not-in-catalogue"],
)
def test_other_fallback_reasons(make, kwargs, reason, warning) -> None:
    kwargs.setdefault("store", _strong_store())
    h = make(**kwargs)
    resp = h.classify()
    assert_fallback(resp, h, reason)
    assert resp.vote is None  # only vote_below_minimum keeps the vote
    assert any(warning in w for w in resp.warnings), resp.warnings


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
