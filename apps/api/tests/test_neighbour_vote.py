"""TASK-0008: neighbour vote (pure) and patent search (pgvector, one partial HNSW index per corpus).

Unit tests need no database. The parity fixture at the bottom (PARITY) was recorded from the offline
PaECTER run (data/manifests/benchmarks/knn_hupd): 3 IP India test queries, their 100 nearest HUPD 2017
neighbours (ids and fp16 similarities from knn_eval.search, recomputed on the GPU exactly as offline and
checked against results_test_paecter.jsonl), labels resolved with knn_eval.split_compact, and the
output of knn_eval.vote for two parameter sets.

Integration tests write to sakthi only with an explicit opt-in (there is no local docker database):
    IPC_DB_TARGET=sakthi IPC_INTEGRATION_ALLOW_SAKTHI=1 uv run pytest -m integration tests/test_neighbour_vote.py
or run against a local database with IPC_DB_TARGET=local. They touch only the patent tables, only corpus
ids starting with "it-test-", build a partial HNSW index named it_test_* for their corpus and drop it
afterwards, and refuse to run if a non-test corpus is serving.
"""

from __future__ import annotations

import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pytest

from app.services.classification.neighbour_vote import LEVELS, LevelShares, NeighbourVote, vote
from app.services.retrieval import patent_search
from app.services.retrieval.patent_search import NeighbourShortfall, PatentNeighbour, search_patents, search_query


@dataclass(frozen=True)
class N:
    document_id: str
    similarity: float
    labels: tuple = ()
    title: str | None = None


H04L_908 = ("H04L", "H04L 9/00", "H04L 9/08")
H04L_900 = ("H04L", "H04L 9/00", "H04L 9/00")
H04L_1218 = ("H04L", "H04L 12/00", "H04L 12/18")
G06F_1600 = ("G06F", "G06F 16/00", "G06F 16/00")
G06F_SUB = ("G06F", None, None)


def _neighbours(record: dict) -> list[PatentNeighbour]:
    return [PatentNeighbour(doc_id, None, sim, tuple(tuple(lab) for lab in labels))  # type: ignore[arg-type]
            for doc_id, sim, labels in record["neighbours"]]  # fmt: skip


# --- parity with the offline knn_eval.vote ---------------------------------------------------------------


@pytest.mark.parametrize("query", [0, 1, 2], ids=lambda q: f"query{q}")
@pytest.mark.parametrize("params", [0, 1], ids=["k100-T0.005-b1.0", "k30-T0.01-b0.5"])
def test_vote_matches_offline_knn_eval(query: int, params: int) -> None:
    record = PARITY[query]
    expected = record["expected"][params]
    assert all(record["offline_check"].values()), record["offline_check"]  # fixture is faithful to the offline run
    result = vote(_neighbours(record), expected["k"], expected["temperature"], expected["beta"])

    assert [result.first_code] == expected["first"]
    shares = {"subclass": result.first_code_share.subclass, "maingroup": result.first_code_share.maingroup,
              "exact": result.first_code_share.exact}
    for level, want in expected["probability"].items():
        if want is None:
            assert shares[level] is None, level
        else:
            assert round(shares[level], 4) == round(want, 4), level
            assert shares[level] == pytest.approx(want, abs=1e-12), level  # in fact identical, not just 4 dp
    for level in LEVELS:
        # ranked codes, in order, and their shares to 4 decimals (offline rounds top_share to 4 dp)
        assert [c.code for c in result.top_codes(level)] == expected["ranked_top10"][level], level
        assert [(c.code, round(c.share, 4)) for c in result.top_codes(level)] == list(
            expected["top_share"][level].items()), level
    assert result.neighbour_count == expected["k"]
    assert [n.document_id for n in result.nearest(5)] == [n[0] for n in record["neighbours"][:5]]
    assert math.fsum(n.vote_weight for n in result.neighbours) == pytest.approx(1.0, abs=1e-12)


def test_parity_fixture_covers_each_first_code_level() -> None:
    levels = {vote(_neighbours(r), 100, 0.005, 1.0).first_code_level for r in PARITY}
    assert levels == {"subgroup", "maingroup"}  # third query: a weak subclass share with a subgroup first code
    weak = vote(_neighbours(PARITY[2]), 100, 0.005, 1.0)
    assert weak.first_code_share.subclass is not None and weak.first_code_share.subclass < 0.5


def test_vote_is_pure() -> None:
    neighbours = _neighbours(PARITY[1])
    snapshot = list(neighbours)
    a = vote(neighbours, 100, 0.005, 1.0)
    b = vote(neighbours, 100, 0.005, 1.0)
    assert neighbours == snapshot  # input not reordered or mutated
    assert a == b and a.ranked == b.ranked and a.first_code_share == b.first_code_share
    # With distinct similarities the input order does not matter: the vote orders by similarity.
    rng = random.Random(7)
    pool = [H04L_908, H04L_900, H04L_1218, G06F_1600, G06F_SUB]
    distinct = [N(f"d{i}", 0.99 - i * 0.0007, tuple(rng.sample(pool, rng.randint(0, 3)))) for i in range(40)]
    shuffled = list(distinct)
    rng.shuffle(shuffled)
    assert vote(shuffled, 30, 0.005, 0.5) == vote(distinct, 30, 0.005, 0.5)


# --- vote: semantics and edge cases --------------------------------------------------------------------------


def test_no_neighbours_returns_an_empty_result() -> None:
    result = vote([], 100, 0.005, 1.0)
    assert result.is_empty and result.first_code is None and result.first_code_level is None
    assert result.first_code_share == LevelShares(None, None, None)
    assert all(result.top_codes(level) == () for level in LEVELS)
    assert result.neighbour_count == 0 and result.nearest() == ()
    assert result.share("subclass", "H04L") == 0.0


def test_all_labels_unresolvable_returns_an_empty_vote_with_its_neighbours() -> None:
    result = vote([N("a", 0.97), N("b", 0.96), N("c", 0.95)], 100, 0.005, 1.0)
    assert result.is_empty and result.first_code is None
    assert result.first_code_share == LevelShares(None, None, None)
    assert all(result.ranked[level] == () for level in LEVELS)
    assert result.neighbour_count == 3  # the neighbours are still reported (TASK-0010 maps this to vote_empty)
    assert [n.ipc_codes for n in result.neighbours] == [[], [], []]
    assert math.fsum(n.vote_weight for n in result.neighbours) == pytest.approx(1.0)


def test_weights_follow_temperature_and_unlabelled_neighbours_dilute_shares() -> None:
    t = 0.005
    result = vote([N("a", 0.97, (H04L_908,)), N("b", 0.97 - t, ()), N("c", 0.97 - 2 * t, (G06F_1600,))], 10, t, 1.0)
    w = [1.0, math.exp(-1), math.exp(-2)]
    assert [n.vote_weight for n in result.neighbours] == pytest.approx([x / sum(w) for x in w])
    assert result.share("subclass", "H04L") == pytest.approx(1.0 / sum(w))
    assert result.share("subclass", "G06F") == pytest.approx(math.exp(-2) / sum(w))
    assert result.first_code == "H04L 9/08" and result.first_code_level == "subgroup"
    assert result.first_code_share == LevelShares(*(pytest.approx(1.0 / sum(w)),) * 3)


def test_a_neighbour_counts_once_per_code_with_its_best_label_weight() -> None:
    # Two labels in H04L: the subclass counts once, at the main label's weight 1, not 1 + beta.
    result = vote([N("a", 0.9, (H04L_908, H04L_1218)), N("b", 0.9, (G06F_1600,))], 10, 0.005, 0.5)
    assert result.share("subclass", "H04L") == pytest.approx(0.5)
    assert result.share("maingroup", "H04L 9/00") == pytest.approx(0.5)
    assert result.share("maingroup", "H04L 12/00") == pytest.approx(0.25)  # secondary label: beta
    assert result.share("subclass", "G06F") == pytest.approx(0.5)


def test_first_code_is_chosen_top_down() -> None:
    # G06F 16/00 is the single best main group, but H04L is the best subclass: the first code stays in H04L.
    neighbours = [N("a", 0.95, (G06F_1600,)), N("b", 0.95, (H04L_908,)), N("c", 0.95, (H04L_1218,)),
                  N("d", 0.95, (("H04L", None, None),))]
    result = vote(neighbours, 10, 0.005, 1.0)
    assert result.top_codes("subclass")[0].code == "H04L"
    assert result.top_codes("maingroup")[0].code == "G06F 16/00"  # 0.25 each: G06F 16/00 seen first
    assert result.first_code == "H04L 9/08"  # best H04L main group (9/00, first seen), then its best exact code
    assert result.first_code_share == LevelShares(0.75, 0.25, 0.25)


def test_a_main_group_label_can_be_the_first_code_at_the_exact_level() -> None:
    result = vote([N("a", 0.95, (H04L_900,)), N("b", 0.95, (H04L_900,)), N("c", 0.95, (H04L_908,))], 10, 0.005, 1.0)
    assert result.first_code == "H04L 9/00" and result.first_code_level == "maingroup"
    assert result.first_code_share == LevelShares(pytest.approx(1.0), pytest.approx(1.0), pytest.approx(2 / 3))


def test_main_group_only_label_gives_a_main_group_first_code() -> None:
    result = vote([N("a", 0.95, (("H04L", "H04L 9/00", None),))], 10, 0.005, 1.0)
    assert result.first_code == "H04L 9/00" and result.first_code_level == "maingroup"
    assert result.first_code_share == LevelShares(1.0, 1.0, None)
    sub_only = vote([N("a", 0.95, (G06F_SUB,))], 10, 0.005, 1.0)
    assert sub_only.first_code == "G06F" and sub_only.first_code_level == "subclass"
    assert sub_only.first_code_share == LevelShares(1.0, None, None)


def test_tie_in_shares_has_a_deterministic_order() -> None:
    # Same weight for both codes: the code carried by the earlier neighbour in similarity order ranks first,
    # and equal similarities keep the input order.
    tied = [N("a", 0.95, (G06F_1600,)), N("b", 0.95, (H04L_908,))]
    for _ in range(3):
        result = vote(tied, 10, 0.005, 1.0)
        assert [c.code for c in result.top_codes("subclass")] == ["G06F", "H04L"]
        assert result.top_codes("subclass")[0].share == result.top_codes("subclass")[1].share == 0.5
        assert result.first_code == "G06F 16/00"
    assert vote(list(reversed(tied)), 10, 0.005, 1.0).first_code == "H04L 9/08"
    # Within one neighbour, label order decides (main label first).
    one = vote([N("a", 0.95, (H04L_1218, G06F_1600))], 10, 0.005, 1.0)
    assert [c.code for c in one.top_codes("subclass")] == ["H04L", "G06F"] and one.first_code == "H04L 12/18"


def test_beta_zero_counts_main_labels_only() -> None:
    neighbours = [N("a", 0.95, (H04L_908, G06F_1600)), N("b", 0.95, (H04L_1218, G06F_1600)), N("c", 0.95, (G06F_SUB,))]
    result = vote(neighbours, 10, 0.005, 0.0)
    assert result.share("subclass", "G06F") == pytest.approx(1 / 3)  # only c's main label
    assert result.share("maingroup", "G06F 16/00") == 0.0  # a and b carry it only as a secondary label
    assert "G06F 16/00" not in [c.code for c in result.top_codes("exact")]
    assert result.first_code == "H04L 9/08"
    with_beta = vote(neighbours, 10, 0.005, 1.0)
    assert with_beta.share("subclass", "G06F") == pytest.approx(1.0)
    assert with_beta.first_code == "G06F 16/00"


def test_only_the_k_most_similar_vote() -> None:
    neighbours = [N(f"d{i}", 0.99 - i * 0.001, (G06F_1600,) if i < 3 else (H04L_908,)) for i in range(10)]
    result = vote(list(reversed(neighbours)), 3, 1.0, 1.0)
    assert [n.document_id for n in result.neighbours] == ["d0", "d1", "d2"]
    assert result.first_code == "G06F 16/00" and result.share("subclass", "H04L") == 0.0


def test_top_codes_and_nearest_are_capped_at_ten_by_default() -> None:
    subs = [f"A{i:02d}B" for i in range(15)]
    result = vote([N(f"d{i}", 0.99 - i * 1e-4, ((s, None, None),)) for i, s in enumerate(subs)], 100, 0.005, 1.0)
    assert len(result.top_codes("subclass")) == 10 and len(result.ranked["subclass"]) == 15
    assert len(result.nearest()) == 10 and result.neighbour_count == 15
    assert [n.similarity for n in result.nearest()] == sorted((n.similarity for n in result.nearest()), reverse=True)


def test_ipc_codes_are_deepest_codes_main_first() -> None:
    result = vote([N("a", 0.9, (H04L_908, ("G06F", "G06F 16/00", None), G06F_SUB, H04L_908))], 10, 0.005, 1.0)
    assert result.neighbours[0].ipc_codes == ["H04L 9/08", "G06F 16/00", "G06F"]


@pytest.mark.parametrize(("k", "t", "beta"), [(0, 0.005, 1.0), (10, 0.0, 1.0), (10, -1.0, 1.0), (10, math.inf, 1.0),
                                               (10, 0.005, -0.1), (10, 0.005, 1.5)])  # fmt: skip
def test_vote_rejects_bad_parameters(k: int, t: float, beta: float) -> None:
    with pytest.raises(ValueError):
        vote([N("a", 0.9, (H04L_908,))], k, t, beta)


def test_vote_rejects_non_finite_similarity() -> None:
    with pytest.raises(ValueError, match="finite"):
        vote([N("a", math.nan, (H04L_908,)), N("b", 0.9, (H04L_908,))], 10, 0.005, 1.0)


def test_result_type_is_frozen() -> None:
    result = vote([N("a", 0.9, (H04L_908,))], 10, 0.005, 1.0)
    assert isinstance(result, NeighbourVote)
    with pytest.raises(AttributeError):
        result.first_code = "G06F"  # type: ignore[misc]


# --- search: SQL shape and argument checks (no database) -----------------------------------------------------


def test_search_query_inlines_the_corpus_id_and_k_as_literals() -> None:
    rendered = search_query("it-test-abc", 100).as_string(None)
    assert "WHERE e.corpus_id = 'it-test-abc'" in rendered
    assert "d.corpus_id = 'it-test-abc'" in rendered
    assert "LIMIT 100" in rendered
    assert rendered.count("CAST(%(qv)s AS halfvec(1024))") == 2  # the only bound parameter
    assert "ORDER BY n.distance, d.document_id" in rendered
    # quoted by the driver, not formatted: a quote in the id cannot break out of the literal
    assert "WHERE e.corpus_id = 'x''; DROP TABLE patent_embeddings; --'" in search_query(
        "x'; DROP TABLE patent_embeddings; --", 5).as_string(None)


def test_halfvec_text_checks_dimension_and_values() -> None:
    v = np.zeros(1024, dtype=np.float32)
    v[0] = 0.5
    text = patent_search.halfvec_text(v)
    assert text.startswith("[0.5,0,") and text.endswith("]") and text.count(",") == 1023
    with pytest.raises(ValueError, match="768 dimensions"):
        patent_search.halfvec_text(np.zeros(768))
    v[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        patent_search.halfvec_text(v)


@pytest.mark.parametrize(("k", "ef"), [(0, 100), (100, 50), (10, 0), (10, 1001)])
def test_search_rejects_bad_arguments_before_touching_the_database(k: int, ef: int) -> None:
    with pytest.raises(ValueError):
        search_patents(object(), "it-test-x", np.zeros(1024), k, ef)  # type: ignore[arg-type]


class _Cursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple]:
        return self.rows


@dataclass
class _FakeConn:
    rows: list[tuple]
    autocommit: bool = False
    executed: list[tuple] = field(default_factory=list)

    def execute(self, query, params=None):  # noqa: ANN001
        self.executed.append((query, params))
        return _Cursor(self.rows if not isinstance(query, str) else [])


def test_search_maps_rows_and_raises_on_fewer_than_k() -> None:
    rows = [("d1", "T1", [["H04L", "H04L 9/00", "H04L 9/08"], ["G06F", None, None]], 0.97),
            ("d2", None, [], 0.95)]  # fmt: skip
    conn = _FakeConn(rows)
    got = search_patents(conn, "it-test-x", np.ones(1024), 2, 100)  # type: ignore[arg-type]
    assert got[0] == PatentNeighbour("d1", "T1", 0.97, (("H04L", "H04L 9/00", "H04L 9/08"), ("G06F", None, None)))
    assert got[0].labels[0].exact == "H04L 9/08" and got[1].labels == ()
    assert conn.executed[0] == ("SELECT set_config('hnsw.ef_search', %s, true)", ["100"])  # transaction-local
    with pytest.raises(NeighbourShortfall, match="returned 2 of k=3") as err:
        search_patents(_FakeConn(rows), "it-test-x", np.ones(1024), 3, 100)  # type: ignore[arg-type]
    assert [n.document_id for n in err.value.neighbours] == ["d1", "d2"] and err.value.k == 3


# --- integration: sakthi with an explicit opt-in (or a local database) ---------------------------------------

TEST_PREFIX = "it-test-"
INDEX_PREFIX = "it_test_"


def _integration_target() -> str | None:
    target = os.environ.get("IPC_DB_TARGET")
    if target == "local":
        return target
    if target == "sakthi" and os.environ.get("IPC_INTEGRATION_ALLOW_SAKTHI") == "1":
        return target
    return None


@pytest.fixture
def db_conn():
    """(conn, created_corpora, created_indexes). Teardown deletes only it-test-* corpora and it_test_* indexes."""
    target = _integration_target()
    if target is None:
        pytest.skip("set IPC_DB_TARGET=local (or IPC_DB_TARGET=sakthi with IPC_INTEGRATION_ALLOW_SAKTHI=1) "
                    "to run DB integration tests")
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().ipc_db_target == target
    from app.db.session import connect_direct

    conn = connect_direct()
    row = conn.execute("SELECT corpus_id FROM patent_corpora WHERE is_serving").fetchone()
    conn.commit()
    if row is not None and not row[0].startswith(TEST_PREFIX):
        conn.close()
        pytest.fail(f"refusing to run: corpus {row[0]!r} is serving")
    created: list[str] = []
    indexes: list[str] = []
    try:
        yield conn, created, indexes
    finally:
        from psycopg import sql

        assert all(c.startswith(TEST_PREFIX) for c in created), created
        assert all(i.startswith(INDEX_PREFIX) for i in indexes), indexes
        conn.rollback()
        conn.autocommit = True
        for name in indexes:
            conn.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(name)))
        conn.execute("DELETE FROM patent_corpora WHERE corpus_id = ANY(%s) AND corpus_id LIKE %s AND NOT is_serving",
                     [created, TEST_PREFIX + "%"])
        conn.close()


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _load_corpus(conn, created: list[str], vectors: dict[str, np.ndarray]) -> str:  # noqa: ANN001
    from app.db import patents
    from app.db.patents import CorpusSpec, PatentDocument

    corpus_id = f"{TEST_PREFIX}{uuid.uuid4().hex[:10]}"
    created.append(corpus_id)
    patents.create_or_get_corpus(conn, CorpusSpec(corpus_id, "hupd", "compact", "mpi-inno-comp/paecter", "rev-test"))
    docs = [PatentDocument(doc_id, f"Title {doc_id}", "A synthetic abstract for the search test.", date(2017, 1, 1),
                           None, ["H04L908"], [("H04L", "H04L 9/00", "H04L 9/08"), ("G06F", None, None)])
            for doc_id in vectors]  # fmt: skip
    patents.upsert_documents(conn, corpus_id, docs, batch_size=500)
    patents.insert_embeddings(conn, corpus_id, vectors.items(), batch_size=500)
    return corpus_id


def _build_partial_index(conn, indexes: list[str], corpus_id: str) -> str:  # noqa: ANN001
    """What TASK-0007 builds per corpus, for this test corpus only."""
    from psycopg import sql

    name = f"{INDEX_PREFIX}{corpus_id.removeprefix(TEST_PREFIX)}_hnsw"
    indexes.append(name)
    conn.execute("SET lock_timeout = '30s'")
    conn.execute(sql.SQL(
        "CREATE INDEX {} ON patent_embeddings USING hnsw (embedding halfvec_cosine_ops) WHERE corpus_id = {}"
    ).format(sql.Identifier(name), sql.Literal(corpus_id)))
    conn.execute("ANALYZE patent_embeddings")
    conn.commit()
    return name


def _plan(conn, query, params) -> str:  # noqa: ANN001
    from psycopg import sql

    rows = conn.execute(sql.SQL("EXPLAIN (FORMAT TEXT) ") + query, params).fetchall()
    return "\n".join(r[0] for r in rows)


@pytest.mark.integration
def test_search_patents_on_a_synthetic_corpus(db_conn) -> None:
    conn, created, indexes = db_conn
    rng = np.random.default_rng(20261013)
    vectors = {f"d{i:03d}": _unit(rng) for i in range(300)}
    target = "d137"
    query = vectors[target] + 0.02 * _unit(rng)  # nearest neighbour is d137 by construction
    query = query / np.linalg.norm(query)
    corpus_id = _load_corpus(conn, created, vectors)
    # Another corpus holding the query vector itself: it must never leak into corpus_id's results.
    other_id = _load_corpus(conn, created, {"intruder": query.copy(), **{f"o{i}": _unit(rng) for i in range(20)}})
    index = _build_partial_index(conn, indexes, corpus_id)

    k = 25
    got = search_patents(conn, corpus_id, query, k, 100)
    conn.commit()
    assert len(got) == k
    assert got[0].document_id == target
    sims = [n.similarity for n in got]
    assert sims == sorted(sims, reverse=True)
    assert all(n.document_id in vectors for n in got)  # nothing from other_id
    # similarity is 1 - cosine distance (halfvec storage: ~1e-3 agreement with float32)
    exact = {d: float(np.dot(v, query)) for d, v in vectors.items()}
    for n in got:
        assert n.similarity == pytest.approx(exact[n.document_id], abs=2e-3)
    want = sorted(exact, key=exact.__getitem__, reverse=True)[:k]
    assert len(set(want) & {n.document_id for n in got}) >= k - 1  # HNSW recall on 300 points
    assert got[0].title == f"Title {target}"
    assert got[0].labels == (("H04L", "H04L 9/00", "H04L 9/08"), ("G06F", None, None))  # main label first
    assert search_patents(conn, other_id, query, 5, 40)[0].document_id == "intruder"
    conn.commit()

    # ef_search is transaction-local: nothing leaks to the connection after the transaction ends.
    default_ef = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
    conn.commit()
    search_patents(conn, corpus_id, query, k, 123)
    assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == "123"
    conn.commit()
    assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == default_ef
    conn.commit()
    conn.autocommit = True  # an autocommit connection gets its own transaction for the search
    try:
        assert search_patents(conn, corpus_id, query, k, 100)[0].document_id == target
        assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == default_ef
    finally:
        conn.autocommit = False
    assert index in indexes


@pytest.mark.integration
def test_search_uses_the_corpus_partial_index(db_conn) -> None:
    from psycopg import sql

    conn, created, indexes = db_conn
    rng = np.random.default_rng(7)
    vectors = {f"d{i:03d}": _unit(rng) for i in range(300)}
    corpus_id = _load_corpus(conn, created, vectors)
    index = _build_partial_index(conn, indexes, corpus_id)
    params = {"qv": patent_search.halfvec_text(_unit(rng))}

    # For a 300-row test corpus a primary-key scan plus a sort is cheaper than the HNSW index, which a
    # 294k-row corpus reverses. The question here is whether the planner *can* use the partial index for
    # this statement, so the competing paths are disabled: sequential and bitmap scans, and sorts (an HNSW
    # scan returns rows in distance order; any other access path has to sort to honour the LIMIT).
    for setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
        conn.execute(f"SET LOCAL {setting} = off")
    plan = _plan(conn, search_query(corpus_id, 25), params)
    assert f"Index Scan using {index}" in plan, plan

    # Why the literal matters: the same statement with the corpus id bound as $1 under a generic plan
    # (what a prepared statement on a pooled connection ends up with) cannot match the partial index.
    conn.execute("SET LOCAL plan_cache_mode = force_generic_plan")
    conn.execute(
        "PREPARE it_test_generic (text, text) AS SELECT e.document_id FROM patent_embeddings e"
        " WHERE e.corpus_id = $1 ORDER BY e.embedding <=> CAST($2 AS halfvec(1024)) LIMIT 25"
    )
    explain = sql.SQL("EXPLAIN EXECUTE it_test_generic ({}, {})").format(
        sql.Literal(corpus_id), sql.Literal(params["qv"]))
    generic_plan = "\n".join(r[0] for r in conn.execute(explain).fetchall())
    assert "on patent_embeddings" in generic_plan and index not in generic_plan, generic_plan
    conn.execute("DEALLOCATE it_test_generic")
    conn.rollback()


@pytest.mark.integration
def test_fewer_than_k_rows_is_an_error(db_conn) -> None:
    conn, created, indexes = db_conn
    rng = np.random.default_rng(11)
    vectors = {f"d{i:02d}": _unit(rng) for i in range(30)}
    corpus_id = _load_corpus(conn, created, vectors)
    _build_partial_index(conn, indexes, corpus_id)
    with pytest.raises(NeighbourShortfall) as err:
        search_patents(conn, corpus_id, _unit(rng), 50, 100)
    conn.rollback()
    assert len(err.value.neighbours) == 30 and err.value.corpus_id == corpus_id
    with pytest.raises(NeighbourShortfall):  # an unknown corpus has no rows at all
        search_patents(conn, f"{TEST_PREFIX}missing-{uuid.uuid4().hex[:6]}", _unit(rng), 5, 40)
    conn.rollback()


@pytest.mark.integration
def test_search_runs_on_a_sqlalchemy_session(db_conn) -> None:
    conn, created, indexes = db_conn
    rng = np.random.default_rng(3)
    vectors = {f"d{i:02d}": _unit(rng) for i in range(40)}
    corpus_id = _load_corpus(conn, created, vectors)
    _build_partial_index(conn, indexes, corpus_id)
    from app.db.session import job_session

    with job_session() as session:
        got = search_patents(session, corpus_id, vectors["d05"], 10, 40)
        session.rollback()
    assert got[0].document_id == "d05" and got[0].similarity == pytest.approx(1.0, abs=2e-3)


# --- recorded offline data (generated once; see the module docstring) ----------------------------------------
# fmt: off
PARITY: list[dict] = json.loads(r"""
[
{"application_number": "202514124304", "offline_check": {"nearest_matches_offline": true, "nearest_similarity_matches_offline": true, "first_matches_offline": true, "probability_matches_offline": true}, "expected": [{"k": 100, "temperature": 0.005, "beta": 1.0, "first": ["F16H 25/22"], "probability": {"subclass": 0.9937830336899969, "maingroup": 0.9925605746620637, "exact": 0.9857802248474986}, "top_share": {"subclass": {"F16H": 0.9938, "B62D": 0.1125, "A01G": 0.0228, "B21D": 0.017, "B23K": 0.0157, "B21H": 0.0157, "F16C": 0.0036, "B23P": 0.0032, "F16D": 0.0031, "B60T": 0.0031}, "maingroup": {"F16H 25/00": 0.9926, "B62D 5/00": 0.1068, "A01G 3/00": 0.0228, "B21H 3/00": 0.0156, "B21D 22/00": 0.0154, "B23K 26/00": 0.0154, "B62D 3/00": 0.0103, "F16H 57/00": 0.005, "F16H 7/00": 0.0042, "B23P 15/00": 0.0032}, "exact": {"F16H 25/22": 0.9858, "B62D 5/04": 0.1068, "F16H 25/24": 0.0509, "A01G 3/037": 0.0228, "F16H 25/20": 0.0161, "B21D 22/00": 0.0154, "B21H 3/08": 0.0154, "B23K 26/21": 0.0154, "B62D 3/08": 0.0101, "F16H 25/18": 0.0057}}, "ranked_top10": {"subclass": ["F16H", "B62D", "A01G", "B21D", "B23K", "B21H", "F16C", "B23P", "F16D", "B60T"], "maingroup": ["F16H 25/00", "B62D 5/00", "A01G 3/00", "B21H 3/00", "B21D 22/00", "B23K 26/00", "B62D 3/00", "F16H 57/00", "F16H 7/00", "B23P 15/00"], "exact": ["F16H 25/22", "B62D 5/04", "F16H 25/24", "A01G 3/037", "F16H 25/20", "B21D 22/00", "B21H 3/08", "B23K 26/21", "B62D 3/08", "F16H 25/18"]}}, {"k": 30, "temperature": 0.01, "beta": 0.5, "first": ["F16H 25/22"], "probability": {"subclass": 0.915625662277807, "maingroup": 0.915625662277807, "exact": 0.8777880677969595}, "top_share": {"subclass": {"F16H": 0.9156, "B62D": 0.1674, "B21D": 0.0244, "A01G": 0.0236, "F16C": 0.0206, "B21H": 0.0194, "B23K": 0.0194, "B23P": 0.0178, "B60T": 0.007, "B22F": 0.0049}, "maingroup": {"F16H 25/00": 0.9156, "B62D 5/00": 0.1522, "B62D 3/00": 0.0301, "A01G 3/00": 0.0236, "F16C 29/00": 0.0206, "B21D 22/00": 0.0194, "B21H 3/00": 0.0194, "B23K 26/00": 0.0194, "B23P 15/00": 0.0178, "F16H 7/00": 0.0089}, "exact": {"F16H 25/22": 0.8778, "B62D 5/04": 0.1522, "F16H 25/24": 0.0911, "F16H 25/20": 0.0553, "B62D 3/08": 0.0301, "A01G 3/037": 0.0236, "F16C 29/06": 0.0206, "B21D 22/00": 0.0194, "B21H 3/08": 0.0194, "B23K 26/21": 0.0194}}, "ranked_top10": {"subclass": ["F16H", "B62D", "B21D", "A01G", "F16C", "B21H", "B23K", "B23P", "B60T", "B22F"], "maingroup": ["F16H 25/00", "B62D 5/00", "B62D 3/00", "A01G 3/00", "F16C 29/00", "B21D 22/00", "B21H 3/00", "B23K 26/00", "B23P 15/00", "F16H 7/00"], "exact": ["F16H 25/22", "B62D 5/04", "F16H 25/24", "F16H 25/20", "B62D 3/08", "A01G 3/037", "F16C 29/06", "B21D 22/00", "B21H 3/08", "B23K 26/21"]}}], "neighbours": [
["15573243", 0.97216796875, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15415595", 0.96142578125, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15461132", 0.95654296875, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15507827", 0.95556640625, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 25/00", "F16H 25/24"]]],
["15513273", 0.955078125, [["F16H", "F16H 25/00", "F16H 25/22"], ["A01G", "A01G 3/00", "A01G 3/037"]]],
["15441970", 0.95361328125, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15517059", 0.953125, [["F16H", "F16H 25/00", "F16H 25/22"], ["B21D", "B21D 22/00", "B21D 22/00"], ["B21H", "B21H 3/00", "B21H 3/08"], ["B23K", "B23K 26/00", "B23K 26/21"]]],
["15723702", 0.95263671875, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 5/00", "B62D 5/04"]]],
["15461165", 0.95166015625, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15815930", 0.95166015625, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 25/00", "F16H 25/24"]]],
["15529717", 0.95166015625, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15502558", 0.951171875, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15787025", 0.94677734375, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 25/00", "F16H 25/08"]]],
["15540445", 0.9462890625, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 25/00", "F16H 25/24"]]],
["15543772", 0.94580078125, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 5/00", "B62D 5/04"]]],
["15539067", 0.9453125, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 7/00", "F16H 7/02"], ["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 3/00", "B62D 3/08"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15510036", 0.9453125, [["B23P", "B23P 15/00", "B23P 15/00"], ["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 25/00", "F16H 25/24"]]],
["15607895", 0.9453125, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 3/00", "B62D 3/08"], ["F16H", "F16H 25/00", "F16H 25/20"], ["F16H", "F16H 25/00", "F16H 25/18"], ["B62D", "B62D 3/00", "B62D 3/06"], ["B62D", "B62D 3/00", "B62D 3/04"], ["B62D", "B62D 3/00", "B62D 3/02"]]],
["15739211", 0.94482421875, [["F16H", "F16H 25/00", "F16H 25/24"]]],
["15513016", 0.94287109375, [["F16H", "F16H 25/00", "F16H 25/24"], ["F16H", "F16H 25/00", "F16H 25/20"], ["F16H", "F16H 25/00", "F16H 25/22"], ["B60T", "B60T 13/00", "B60T 13/74"]]],
["15538522", 0.94287109375, [["F16H", "F16H 25/00", "F16H 25/20"], ["F16H", "F16H 57/00", "F16H 57/04"]]],
["15600887", 0.94189453125, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 3/00", "B62D 3/08"], ["F16H", "F16H 25/00", "F16H 25/20"], ["F16H", "F16H 25/00", "F16H 25/18"], ["B62D", "B62D 3/00", "B62D 3/06"], ["B62D", "B62D 3/00", "B62D 3/04"], ["B62D", "B62D 3/00", "B62D 3/02"]]],
["15511830", 0.94140625, [["F16C", "F16C 29/00", "F16C 29/06"]]],
["15602288", 0.94140625, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 3/00", "B62D 3/08"], ["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15596466", 0.94091796875, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15668842", 0.939453125, [["F16H", "F16H 25/00", "F16H 25/22"], ["B22F", "B22F 3/00", "B22F 3/12"], ["B21K", "B21K 1/00", "B21K 1/70"], ["C21D", "C21D 1/00", "C21D 1/18"], ["B21D", "B21D 39/00", "B21D 39/00"]]],
["15568472", 0.93798828125, [["F16C", "F16C 29/00", "F16C 29/06"], ["F16C", "F16C 33/00", "F16C 33/32"], ["F16C", "F16C 33/00", "F16C 33/374"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15850717", 0.93798828125, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15850881", 0.93798828125, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16D", "F16D 65/00", "F16D 65/18"], ["F16D", "F16D 121/00", "F16D 121/24"], ["F16D", "F16D 125/00", "F16D 125/48"], ["F16D", "F16D 125/00", "F16D 125/40"]]],
["15597348", 0.93798828125, [["F16H", "F16H 25/00", "F16H 25/22"], ["B64C", "B64C 13/00", "B64C 13/36"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15573849", 0.9375, [["F16J", "F16J 15/00", "F16J 15/46"], ["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 25/00", "F16H 25/24"], ["F16H", "F16H 57/00", "F16H 57/04"]]],
["15554099", 0.9375, [["F16H", "F16H 7/00", "F16H 7/18"], ["F16H", "F16H 57/00", "F16H 57/00"], ["F16H", "F16H 9/00", "F16H 9/24"]]],
["15816661", 0.9375, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 5/00", "B62D 5/04"]]],
["15736014", 0.9375, [["F16H", "F16H 57/00", "F16H 57/04"], ["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 25/00", "F16H 25/24"]]],
["15439377", 0.93701171875, [["F16D", "F16D 65/00", "F16D 65/16"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15709149", 0.93701171875, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16D", "F16D 65/00", "F16D 65/18"], ["B60T", "B60T 13/00", "B60T 13/74"], ["F16D", "F16D 121/00", "F16D 121/24"], ["B64C", "B64C 25/00", "B64C 25/44"], ["F16D", "F16D 125/00", "F16D 125/50"], ["F16H", "F16H 25/00", "F16H 25/20"], ["F16D", "F16D 125/00", "F16D 125/40"]]],
["15615098", 0.93603515625, [["F16H", "F16H 25/00", "F16H 25/22"]]],
["15546286", 0.935546875, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16D", "F16D 65/00", "F16D 65/14"], ["B60T", "B60T 13/00", "B60T 13/74"], ["F16D", "F16D 125/00", "F16D 125/40"], ["F16D", "F16D 125/00", "F16D 125/36"]]],
["15481449", 0.935546875, [["B62D", "B62D 1/00", "B62D 1/185"], ["F16D", "F16D 3/00", "F16D 3/22"], ["F16D", "F16D 3/00", "F16D 3/18"], ["F16D", "F16D 3/00", "F16D 3/221"], ["B62D", "B62D 1/00", "B62D 1/18"]]],
["15552811", 0.935546875, [["F16H", "F16H 25/00", "F16H 25/24"]]],
["15828522", 0.93408203125, [["F04C", "F04C 2/00", "F04C 2/107"], ["F04C", "F04C 14/00", "F04C 14/10"], ["F15B", "F15B 15/00", "F15B 15/14"], ["F15B", "F15B 15/00", "F15B 15/06"]]],
["15499258", 0.93408203125, [["F02M", "F02M 26/00", "F02M 26/67"], ["F16H", "F16H 37/00", "F16H 37/12"], ["F16K", "F16K 31/00", "F16K 31/53"], ["F16H", "F16H 25/00", "F16H 25/18"], ["F16K", "F16K 31/00", "F16K 31/524"], ["F02M", "F02M 26/00", "F02M 26/54"], ["F16H", "F16H 19/00", "F16H 19/00"]]],
["15542312", 0.93408203125, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 5/00", "B62D 5/04"], ["B21J", "B21J 5/00", "B21J 5/02"]]],
["15681471", 0.93359375, [["B24B", "B24B 21/00", "B24B 21/16"], ["B24B", "B24B 29/00", "B24B 29/02"], ["B23G", "B23G 9/00", "B23G 9/00"], ["B24B", "B24B 19/00", "B24B 19/02"]]],
["15606681", 0.93310546875, [["F16H", "F16H 25/00", "F16H 25/22"], ["B23K", "B23K 11/00", "B23K 11/00"], ["B21D", "B21D 9/00", "B21D 9/04"], ["B62D", "B62D 3/00", "B62D 3/08"], ["B62D", "B62D 3/00", "B62D 3/04"], ["F16H", "F16H 25/00", "F16H 25/18"], ["B62D", "B62D 3/00", "B62D 3/02"], ["B62D", "B62D 3/00", "B62D 3/06"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15709888", 0.9326171875, [["F16H", "F16H 19/00", "F16H 19/00"], ["F16H", "F16H 25/00", "F16H 25/04"]]],
["15513997", 0.9326171875, [["B27B", "B27B 17/00", "B27B 17/14"], ["B27B", "B27B 17/00", "B27B 17/02"]]],
["15735009", 0.93212890625, [["B21B", "B21B 35/00", "B21B 35/14"]]],
["15451594", 0.931640625, [["F16C", "F16C 29/00", "F16C 29/00"], ["F16C", "F16C 29/00", "F16C 29/02"], ["A47B", "A47B 88/00", "A47B 88/483"], ["F16C", "F16C 29/00", "F16C 29/10"]]],
["15547646", 0.931640625, [["B25B", "B25B 23/00", "B25B 23/147"], ["B25D", "B25D 11/00", "B25D 11/10"], ["B25B", "B25B 21/00", "B25B 21/02"], ["B25D", "B25D 16/00", "B25D 16/00"]]],
["15831473", 0.93115234375, [["F04B", "F04B 9/00", "F04B 9/04"], ["F16H", "F16H 25/00", "F16H 25/18"], ["F16H", "F16H 53/00", "F16H 53/02"], ["F16H", "F16H 53/00", "F16H 53/06"]]],
["15685845", 0.93115234375, [["B21H", "B21H 3/00", "B21H 3/02"]]],
["15730887", 0.9306640625, [["F16C", "F16C 29/00", "F16C 29/06"], ["F16C", "F16C 29/00", "F16C 29/00"]]],
["15324937", 0.93017578125, [["F16H", "F16H 57/00", "F16H 57/04"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15463663", 0.93017578125, [["B61G", "B61G 5/00", "B61G 5/10"], ["H02G", "H02G 3/00", "H02G 3/04"], ["F16L", "F16L 3/00", "F16L 3/10"], ["F16L", "F16L 3/00", "F16L 3/01"], ["F16L", "F16L 3/00", "F16L 3/237"]]],
["15531132", 0.93017578125, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 7/00", "F16H 7/02"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15616367", 0.93017578125, [["F16H", "F16H 57/00", "F16H 57/00"], ["F16H", "F16H 25/00", "F16H 25/24"], ["F16C", "F16C 27/00", "F16C 27/06"]]],
["15468071", 0.9296875, [["F16H", "F16H 19/00", "F16H 19/06"], ["F16H", "F16H 57/00", "F16H 57/04"], ["G01B", "G01B 5/00", "G01B 5/008"]]],
["15791719", 0.9296875, [["B62D", "B62D 5/00", "B62D 5/04"], ["B62D", "B62D 3/00", "B62D 3/12"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15619360", 0.9296875, [["G05G", "G05G 23/00", "G05G 23/00"], ["G05G", "G05G 1/00", "G05G 1/08"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15584751", 0.92919921875, [["B60N", "B60N 2/00", "B60N 2/07"], ["F16C", "F16C 29/00", "F16C 29/04"], ["F16C", "F16C 29/00", "F16C 29/06"], ["B60N", "B60N 2/00", "B60N 2/04"], ["B60N", "B60N 2/00", "B60N 2/50"]]],
["15639494", 0.92919921875, [["F16H", "F16H 57/00", "F16H 57/022"], ["F16C", "F16C 19/00", "F16C 19/54"], ["F16H", "F16H 57/00", "F16H 57/12"], ["F16H", "F16H 1/00", "F16H 1/16"], ["F16H", "F16H 57/00", "F16H 57/021"]]],
["15701690", 0.92919921875, [["F16D", "F16D 27/00", "F16D 27/115"], ["F16H", "F16H 63/00", "F16H 63/30"]]],
["15520761", 0.9287109375, [["F15B", "F15B 15/00", "F15B 15/08"]]],
["15562869", 0.9287109375, [["B65G", "B65G 17/00", "B65G 17/16"], ["B65G", "B65G 17/00", "B65G 17/42"], ["B65G", "B65G 17/00", "B65G 17/36"]]],
["15552110", 0.9287109375, [["F16H", "F16H 57/00", "F16H 57/027"], ["F16H", "F16H 57/00", "F16H 57/04"], ["F16H", "F16H 57/00", "F16H 57/031"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15421564", 0.9287109375, [["B61B", "B61B 3/00", "B61B 3/00"], ["B65G", "B65G 9/00", "B65G 9/00"]]],
["15506011", 0.9287109375, [["E01B", "E01B 25/00", "E01B 25/24"], ["B65G", "B65G 19/00", "B65G 19/02"], ["B61B", "B61B 13/00", "B61B 13/04"], ["B65G", "B65G 39/00", "B65G 39/18"], ["B65G", "B65G 9/00", "B65G 9/00"]]],
["15525392", 0.9287109375, [["B21D", "B21D 5/00", "B21D 5/02"]]],
["15666662", 0.92822265625, [["F16H", "F16H 25/00", "F16H 25/20"], ["F16H", "F16H 25/00", "F16H 25/22"], ["C22C", "C22C 38/00", "C22C 38/42"], ["C22C", "C22C 38/00", "C22C 38/04"], ["C22C", "C22C 38/00", "C22C 38/02"], ["C22C", "C22C 33/00", "C22C 33/02"], ["C22C", "C22C 38/00", "C22C 38/44"], ["B22F", "B22F 3/00", "B22F 3/22"]]],
["15562489", 0.92822265625, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16C", "F16C 33/00", "F16C 33/04"], ["B62D", "B62D 3/00", "B62D 3/08"]]],
["15553501", 0.92822265625, [["F16H", "F16H 25/00", "F16H 25/20"], ["F16H", "F16H 57/00", "F16H 57/04"], ["F16H", "F16H 25/00", "F16H 25/24"]]],
["15544860", 0.92822265625, [["F16H", "F16H 25/00", "F16H 25/22"], ["F16K", "F16K 31/00", "F16K 31/50"], ["F16H", "F16H 57/00", "F16H 57/04"], ["F01D", "F01D 17/00", "F01D 17/10"], ["F16K", "F16K 31/00", "F16K 31/04"]]],
["15453697", 0.927734375, [["F16K", "F16K 31/00", "F16K 31/528"], ["B67D", "B67D 1/00", "B67D 1/00"], ["F16K", "F16K 31/00", "F16K 31/122"]]],
["15506056", 0.927734375, [["B65G", "B65G 9/00", "B65G 9/00"], ["B65G", "B65G 39/00", "B65G 39/12"], ["B65G", "B65G 19/00", "B65G 19/02"], ["B65G", "B65G 17/00", "B65G 17/20"]]],
["15451919", 0.927734375, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 7/00", "F16H 7/02"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15551114", 0.927734375, [["F16C", "F16C 29/00", "F16C 29/04"], ["F16C", "F16C 33/00", "F16C 33/34"]]],
["15604862", 0.92724609375, [["F16H", "F16H 25/00", "F16H 25/22"], ["B62D", "B62D 3/00", "B62D 3/08"], ["B62D", "B62D 5/00", "B62D 5/04"]]],
["15842732", 0.92724609375, [["F16H", "F16H 25/00", "F16H 25/20"], ["B62D", "B62D 3/00", "B62D 3/06"], ["B62D", "B62D 5/00", "B62D 5/04"]]],
["15552144", 0.92724609375, [["B22D", "B22D 11/00", "B22D 11/053"], ["F16H", "F16H 57/00", "F16H 57/029"], ["F16H", "F16H 25/00", "F16H 25/22"], ["F16H", "F16H 57/00", "F16H 57/04"], ["H02K", "H02K 7/00", "H02K 7/06"], ["H02K", "H02K 7/00", "H02K 7/10"]]],
["15724345", 0.92724609375, [["A61B", "A61B 17/00", "A61B 17/86"]]],
["15560051", 0.92724609375, [["F16C", "F16C 29/00", "F16C 29/06"], ["F16C", "F16C 33/00", "F16C 33/38"]]],
["15548515", 0.92724609375, [["D07B", "D07B 5/00", "D07B 5/12"], ["D07B", "D07B 3/00", "D07B 3/10"], ["D07B", "D07B 7/00", "D07B 7/02"], ["D07B", "D07B 7/00", "D07B 7/10"], ["D07B", "D07B 1/00", "D07B 1/06"], ["B21C", "B21C 47/00", "B21C 47/12"], ["B21F", "B21F 15/00", "B21F 15/04"]]],
["15529887", 0.92724609375, [["F16C", "F16C 3/00", "F16C 3/035"], ["B21H", "B21H 1/00", "B21H 1/18"], ["B21D", "B21D 53/00", "B21D 53/88"], ["B62D", "B62D 1/00", "B62D 1/185"]]],
["15562654", 0.92724609375, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 25/00", "F16H 25/22"], ["F16C", "F16C 17/00", "F16C 17/02"]]],
["15520507", 0.9267578125, [["B23G", "B23G 3/00", "B23G 3/00"], ["B26F", "B26F 1/00", "B26F 1/14"], ["B23G", "B23G 1/00", "B23G 1/44"]]],
["15479579", 0.9267578125, [["B23Q", "B23Q 3/00", "B23Q 3/18"], ["B23C", "B23C 5/00", "B23C 5/26"]]],
["15329138", 0.9267578125, [["H02K", "H02K 7/00", "H02K 7/06"], ["H02K", "H02K 5/00", "H02K 5/04"], ["H02K", "H02K 7/00", "H02K 7/116"]]],
["15552827", 0.9267578125, [["B65G", "B65G 25/00", "B65G 25/02"], ["B23Q", "B23Q 7/00", "B23Q 7/04"], ["B21D", "B21D 43/00", "B21D 43/05"]]],
["15667756", 0.9267578125, [["B62D", "B62D 5/00", "B62D 5/04"]]],
["15589964", 0.9267578125, [["B65G", "B65G 54/00", "B65G 54/02"], ["H02K", "H02K 16/00", "H02K 16/04"], ["H02K", "H02K 41/00", "H02K 41/03"], ["H02K", "H02K 7/00", "H02K 7/14"], ["B65G", "B65G 35/00", "B65G 35/06"], ["B65G", "B65G 47/00", "B65G 47/96"]]],
["15563113", 0.92626953125, [["F16H", "F16H 25/00", "F16H 25/24"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15556008", 0.92626953125, [["F16H", "F16H 1/00", "F16H 1/16"], ["F16H", "F16H 55/00", "F16H 55/22"], ["B25B", "B25B 17/00", "B25B 17/02"]]],
["15635196", 0.92626953125, [["B62D", "B62D 5/00", "B62D 5/04"], ["F16H", "F16H 25/00", "F16H 25/22"], ["F16C", "F16C 19/00", "F16C 19/06"], ["F16H", "F16H 25/00", "F16H 25/20"]]],
["15477535", 0.92626953125, [["F16H", "F16H 55/00", "F16H 55/08"], ["B23F", "B23F 19/00", "B23F 19/10"], ["F16H", "F16H 48/00", "F16H 48/285"]]],
["15451824", 0.92626953125, [["B23B", "B23B 3/00", "B23B 3/26"], ["B23B", "B23B 5/00", "B23B 5/16"], ["B23D", "B23D 21/00", "B23D 21/04"], ["B23B", "B23B 5/00", "B23B 5/08"]]],
["15472405", 0.92578125, [["B65G", "B65G 9/00", "B65G 9/00"]]],
["15830763", 0.92578125, [["F16H", "F16H 25/00", "F16H 25/20"], ["F16H", "F16H 25/00", "F16H 25/22"]]],
["15519751", 0.92578125, [["F16C", "F16C 3/00", "F16C 3/03"], ["F16D", "F16D 3/00", "F16D 3/06"], ["B21K", "B21K 1/00", "B21K 1/06"]]],
["15528779", 0.92529296875, [["A61B", "A61B 1/00", "A61B 1/005"], ["A61M", "A61M 25/00", "A61M 25/01"]]]
]},
{"application_number": "202514125965", "offline_check": {"nearest_matches_offline": true, "nearest_similarity_matches_offline": true, "first_matches_offline": true, "probability_matches_offline": true}, "expected": [{"k": 100, "temperature": 0.005, "beta": 1.0, "first": ["A61B 1/00"], "probability": {"subclass": 0.48647557537251646, "maingroup": 0.16164904488777487, "exact": 0.14103980299537405}, "top_share": {"subclass": {"A61B": 0.4865, "H04N": 0.2176, "G06F": 0.1316, "B41J": 0.1202, "G06T": 0.0984, "G02B": 0.0865, "H05K": 0.0692, "A44C": 0.0548, "A61M": 0.039, "G03G": 0.0358}, "maingroup": {"A61B 1/00": 0.1616, "A61B 5/00": 0.1462, "H04N 1/00": 0.1105, "B41J 3/00": 0.1029, "B41J 29/00": 0.0982, "A61B 90/00": 0.0892, "G06T 7/00": 0.0804, "H04N 5/00": 0.0779, "G06F 3/00": 0.0772, "A61B 6/00": 0.0766}, "exact": {"A61B 5/00": 0.1417, "A61B 1/00": 0.141, "A61B 1/04": 0.1194, "H04N 1/00": 0.1045, "G06T 7/00": 0.0744, "A61B 6/00": 0.0721, "G02B 21/00": 0.0718, "G06F 1/16": 0.0705, "H05K 5/00": 0.0692, "H05K 5/02": 0.0692}}, "ranked_top10": {"subclass": ["A61B", "H04N", "G06F", "B41J", "G06T", "G02B", "H05K", "A44C", "A61M", "G03G"], "maingroup": ["A61B 1/00", "A61B 5/00", "H04N 1/00", "B41J 3/00", "B41J 29/00", "A61B 90/00", "G06T 7/00", "H04N 5/00", "G06F 3/00", "A61B 6/00"], "exact": ["A61B 5/00", "A61B 1/00", "A61B 1/04", "H04N 1/00", "G06T 7/00", "A61B 6/00", "G02B 21/00", "G06F 1/16", "H05K 5/00", "H05K 5/02"]}}, {"k": 30, "temperature": 0.01, "beta": 0.5, "first": ["A61B 5/00"], "probability": {"subclass": 0.4127744120741385, "maingroup": 0.1394472761048468, "exact": 0.1394472761048468}, "top_share": {"subclass": {"A61B": 0.4128, "H04N": 0.1748, "B41J": 0.1197, "G06F": 0.0923, "G02B": 0.0758, "G06T": 0.0661, "H05K": 0.0643, "G03B": 0.0458, "A44C": 0.0404, "G01D": 0.0309}, "maingroup": {"A61B 5/00": 0.1394, "A61B 1/00": 0.1087, "H04N 1/00": 0.1064, "B41J 3/00": 0.1035, "G06F 1/00": 0.0923, "B41J 29/00": 0.0855, "A61B 8/00": 0.0813, "A61B 6/00": 0.0795, "G02B 21/00": 0.0758, "H04N 5/00": 0.0683}, "exact": {"A61B 5/00": 0.1394, "A61B 1/00": 0.1087, "H04N 1/00": 0.1064, "G06F 1/16": 0.0923, "A61B 6/00": 0.0795, "G02B 21/00": 0.0758, "A61B 8/00": 0.0692, "H05K 5/00": 0.0643, "B41J 3/407": 0.0564, "A61B 1/04": 0.0543}}, "ranked_top10": {"subclass": ["A61B", "H04N", "B41J", "G06F", "G02B", "G06T", "H05K", "G03B", "A44C", "G01D"], "maingroup": ["A61B 5/00", "A61B 1/00", "H04N 1/00", "B41J 3/00", "G06F 1/00", "B41J 29/00", "A61B 8/00", "A61B 6/00", "G02B 21/00", "H04N 5/00"], "exact": ["A61B 5/00", "A61B 1/00", "H04N 1/00", "G06F 1/16", "A61B 6/00", "G02B 21/00", "A61B 8/00", "H05K 5/00", "B41J 3/407", "A61B 1/04"]}}], "neighbours": [
["15426151", 0.95654296875, [["H05K", "H05K 5/00", "H05K 5/00"], ["H05K", "H05K 5/00", "H05K 5/02"]]],
["15710207", 0.95556640625, [["A61B", "A61B 1/00", "A61B 1/00"], ["A61B", "A61B 1/00", "A61B 1/04"], ["G06T", "G06T 7/00", "G06T 7/00"]]],
["15328522", 0.9541015625, [["G02B", "G02B 21/00", "G02B 21/00"], ["G02B", "G02B 7/00", "G02B 7/00"], ["G02B", "G02B 21/00", "G02B 21/36"], ["A61B", "A61B 90/00", "A61B 90/25"]]],
["15522142", 0.9541015625, [["A61B", "A61B 1/00", "A61B 1/00"], ["A61B", "A61B 1/00", "A61B 1/06"], ["A61B", "A61B 1/00", "A61B 1/313"], ["A61B", "A61B 1/00", "A61B 1/04"]]],
["15484993", 0.9521484375, [["A61B", "A61B 5/00", "A61B 5/00"], ["A61B", "A61B 5/00", "A61B 5/11"], ["A44C", "A44C 5/00", "A44C 5/00"]]],
["15591612", 0.95166015625, [["A61B", "A61B 5/00", "A61B 5/00"], ["A44C", "A44C 5/00", "A44C 5/24"], ["A61B", "A61B 5/00", "A61B 5/01"], ["A61B", null, null]]],
["15560924", 0.951171875, [["B41J", "B41J 3/00", "B41J 3/407"], ["B41J", "B41J 2/00", "B41J 2/32"], ["B41J", "B41J 29/00", "B41J 29/13"], ["B41J", "B41J 15/00", "B41J 15/04"]]],
["15432503", 0.951171875, [["H04N", "H04N 1/00", "H04N 1/00"], ["B41J", "B41J 3/00", "B41J 3/407"], ["B41J", "B41J 29/00", "B41J 29/02"]]],
["15541828", 0.95068359375, [["A61B", "A61B 6/00", "A61B 6/00"]]],
["15673607", 0.9501953125, [["A61B", "A61B 8/00", "A61B 8/00"], ["A61B", "A61B 5/00", "A61B 5/00"], ["A61B", "A61B 8/00", "A61B 8/12"], ["A61B", "A61B 6/00", "A61B 6/00"]]],
["15559784", 0.94970703125, [["B41J", "B41J 29/00", "B41J 29/13"], ["B41J", "B41J 3/00", "B41J 3/36"], ["B41J", "B41J 15/00", "B41J 15/04"]]],
["15834641", 0.94970703125, [["G03B", "G03B 17/00", "G03B 17/56"], ["H04N", "H04N 5/00", null]]],
["15543148", 0.94970703125, [["H04N", "H04N 1/00", "H04N 1/00"], ["H04N", "H04N 1/00", "H04N 1/44"], ["G03G", "G03G 15/00", "G03G 15/00"]]],
["15641845", 0.94921875, [["G01D", "G01D 7/00", "G01D 7/00"]]],
["15704192", 0.94921875, [["H01Q", "H01Q 1/00", "H01Q 1/22"], ["H01Q", "H01Q 7/00", "H01Q 7/00"], ["H01Q", "H01Q 1/00", "H01Q 1/06"]]],
["15560998", 0.94921875, [["B41J", "B41J 3/00", "B41J 3/36"], ["B41J", "B41J 29/00", "B41J 29/02"], ["B41J", "B41J 2/00", "B41J 2/32"]]],
["15477539", 0.94873046875, [["B65D", "B65D 43/00", "B65D 43/18"], ["B65D", "B65D 43/00", "B65D 43/26"]]],
["15710701", 0.94873046875, [["A61B", "A61B 5/00", "A61B 5/00"], ["A61B", "A61B 5/00", "A61B 5/021"]]],
["15629418", 0.9482421875, [["G06F", "G06F 1/00", "G06F 1/16"]]],
["15627655", 0.94775390625, [["A61B", "A61B 6/00", "A61B 6/00"], ["A61B", "A61B 6/00", "A61B 6/10"]]],
["15822588", 0.94775390625, [["H04N", "H04N 5/00", null], ["H04N", "H04N 5/00", "H04N 5/44"], ["H04N", "H04N 21/00", "H04N 21/218"], ["G06F", "G06F 3/00", "G06F 3/0362"], ["G03B", "G03B 17/00", "G03B 17/38"], ["G03B", "G03B 17/00", "G03B 17/20"], ["H04N", "H04N 5/00", null], ["G06F", "G06F 1/00", "G06F 1/16"], ["H04N", "H04N 21/00", "H04N 21/488"], ["H04N", "H04N 21/00", "H04N 21/462"], ["H04N", "H04N 21/00", "H04N 21/41"], ["H04N", "H04N 7/00", "H04N 7/18"]]],
["15557830", 0.94775390625, [["G06F", "G06F 1/00", "G06F 1/16"], ["H04N", "H04N 1/00", "H04N 1/00"]]],
["15564050", 0.947265625, [["G02B", "G02B 21/00", "G02B 21/00"], ["G02B", "G02B 21/00", "G02B 21/36"], ["A61B", "A61B 90/00", "A61B 90/00"]]],
["15585179", 0.947265625, [["H04N", "H04N 5/00", null], ["G06T", "G06T 7/00", "G06T 7/73"], ["H04N", "H04N 5/00", null], ["G06T", "G06T 7/00", "G06T 7/00"]]],
["15570177", 0.94677734375, [["H04M", "H04M 1/00", "H04M 1/02"], ["G09F", "G09F 3/00", "G09F 3/00"], ["H04B", "H04B 1/00", "H04B 1/3816"], ["G06K", "G06K 13/00", "G06K 13/08"], ["H04M", "H04M 1/00", "H04M 1/21"]]],
["15559075", 0.94677734375, [["G06F", "G06F 1/00", "G06F 1/16"]]],
["15498496", 0.94677734375, [["G06T", "G06T 11/00", "G06T 11/20"], ["G06T", "G06T 15/00", "G06T 15/20"], ["A61B", "A61B 5/00", "A61B 5/00"]]],
["15602413", 0.94677734375, [["A61B", "A61B 8/00", "A61B 8/08"], ["A61B", "A61B 8/00", "A61B 8/00"], ["A61B", "A61B 8/00", "A61B 8/14"]]],
["15549594", 0.9462890625, [["A61B", "A61B 8/00", "A61B 8/00"]]],
["15630176", 0.9462890625, [["H04N", "H04N 1/00", "H04N 1/00"]]],
["15563812", 0.9462890625, [["A61B", "A61B 1/00", "A61B 1/00"]]],
["15854842", 0.9462890625, [["H04N", "H04N 1/00", "H04N 1/00"]]],
["15642573", 0.94580078125, [["G04B", "G04B 47/00", "G04B 47/00"], ["G02F", "G02F 1/00", "G02F 1/1333"], ["G06F", "G06F 3/00", "G06F 3/044"]]],
["15423310", 0.94580078125, [["G06F", "G06F 1/00", "G06F 1/16"], ["F16M", "F16M 11/00", "F16M 11/10"]]],
["15401610", 0.94580078125, [["A61B", "A61B 5/00", "A61B 5/00"]]],
["15635903", 0.94580078125, [["A61B", null, null], ["A61B", "A61B 5/00", "A61B 5/00"]]],
["15531123", 0.94580078125, [["A61M", "A61M 5/00", "A61M 5/168"], ["A61M", "A61M 5/00", "A61M 5/36"], ["A61M", "A61M 5/00", "A61M 5/142"]]],
["15484582", 0.94580078125, [["H04N", "H04N 5/00", null], ["H04N", "H04N 5/00", null]]],
["15846465", 0.94580078125, [["A61B", "A61B 8/00", "A61B 8/00"], ["A61B", "A61B 8/00", "A61B 8/12"], ["B08B", "B08B 1/00", "B08B 1/00"], ["G06F", "G06F 3/00", "G06F 3/0354"], ["G06F", "G06F 3/00", "G06F 3/03"]]],
["15637187", 0.9453125, [["H04N", "H04N 1/00", "H04N 1/00"], ["B41J", "B41J 3/00", "B41J 3/46"]]],
["15510124", 0.9453125, [["H04N", null, null], ["H04N", "H04N 13/00", "H04N 13/00"], ["A61B", "A61B 1/00", "A61B 1/06"], ["A61B", "A61B 1/00", "A61B 1/045"], ["A61B", "A61B 1/00", "A61B 1/05"], ["A61B", "A61B 1/00", "A61B 1/00"]]],
["15699485", 0.9453125, [["A61B", "A61B 1/00", "A61B 1/005"], ["A61B", "A61B 1/00", "A61B 1/12"], ["A61B", "A61B 1/00", "A61B 1/06"], ["G02B", "G02B 23/00", "G02B 23/24"], ["A61B", "A61B 1/00", "A61B 1/00"]]],
["15509712", 0.9453125, [["A61B", "A61B 46/00", "A61B 46/10"], ["A61B", "A61B 34/00", "A61B 34/35"], ["B25J", "B25J 19/00", "B25J 19/00"]]],
["15498411", 0.9453125, [["H04N", "H04N 13/00", "H04N 13/00"], ["A61B", "A61B 5/00", "A61B 5/00"]]],
["15573561", 0.9453125, [["A61B", "A61B 1/00", "A61B 1/04"], ["A61B", "A61B 90/00", "A61B 90/20"], ["H04N", "H04N 13/00", "H04N 13/00"]]],
["15847300", 0.94482421875, [["H04N", "H04N 5/00", null], ["H04N", "H04N 5/00", "H04N 5/262"]]],
["15697998", 0.94482421875, [["H04N", "H04N 1/00", "H04N 1/00"], ["H04N", "H04N 1/00", "H04N 1/44"]]],
["15644164", 0.94482421875, [["A61M", "A61M 1/00", "A61M 1/36"], ["A61M", "A61M 1/00", "A61M 1/14"], ["G06F", "G06F 3/00", "G06F 3/041"], ["G06K", null, null], ["H04N", "H04N 9/00", "H04N 9/31"]]],
["15443201", 0.94482421875, [["A61B", "A61B 1/00", "A61B 1/04"]]],
["15501216", 0.94482421875, [["G02B", "G02B 21/00", "G02B 21/36"], ["G02B", "G02B 21/00", "G02B 21/24"], ["G02B", "G02B 21/00", "G02B 21/00"]]],
["15584948", 0.94482421875, [["A61B", "A61B 1/00", "A61B 1/045"], ["G06F", "G06F 3/00", "G06F 3/14"], ["G06F", "G06F 3/00", "G06F 3/041"], ["H04N", "H04N 7/00", "H04N 7/18"], ["G06T", "G06T 7/00", "G06T 7/00"]]],
["15436866", 0.9443359375, [["G01N", "G01N 33/00", "G01N 33/487"], ["H02J", "H02J 7/00", "H02J 7/02"]]],
["15517428", 0.9443359375, [["A61B", "A61B 1/00", "A61B 1/00"], ["H01R", "H01R 13/00", "H01R 13/62"]]],
["15537767", 0.9443359375, [["A61M", "A61M 5/00", "A61M 5/24"], ["A61B", "A61B 1/00", "A61B 1/00"], ["A61B", "A61B 1/00", "A61B 1/015"], ["A61B", "A61B 1/00", "A61B 1/04"], ["A61B", "A61B 1/00", "A61B 1/06"], ["A61B", "A61B 1/00", "A61B 1/07"], ["A61B", "A61B 34/00", "A61B 34/20"], ["A61M", "A61M 5/00", "A61M 5/46"], ["A61B", "A61B 17/00", "A61B 17/34"]]],
["15636845", 0.9443359375, [["A61B", "A61B 1/00", "A61B 1/00"], ["A61B", "A61B 1/00", "A61B 1/045"]]],
["15560988", 0.9443359375, [["B41J", "B41J 3/00", "B41J 3/407"], ["B41J", "B41J 2/00", "B41J 2/32"], ["B41J", "B41J 3/00", "B41J 3/36"], ["G06F", "G06F 3/00", "G06F 3/02"]]],
["15826655", 0.9443359375, [["H04N", "H04N 5/00", null], ["G06T", "G06T 7/00", "G06T 7/70"], ["H04N", "H04N 5/00", "H04N 5/44"], ["H04N", "H04N 1/00", "H04N 1/107"], ["G09G", "G09G 5/00", "G09G 5/10"]]],
["15511518", 0.9443359375, [["G06F", "G06F 1/00", null], ["A61M", "A61M 16/00", "A61M 16/04"], ["A61B", "A61B 5/00", "A61B 5/00"]]],
["15808332", 0.9443359375, [["F16M", "F16M 13/00", "F16M 13/02"], ["A61B", "A61B 8/00", "A61B 8/00"], ["A61B", "A61B 6/00", "A61B 6/00"], ["F16M", "F16M 11/00", "F16M 11/10"], ["F16M", "F16M 11/00", "F16M 11/38"]]],
["15719457", 0.9443359375, [["G01N", "G01N 33/00", "G01N 33/487"], ["H02J", "H02J 7/00", "H02J 7/02"]]],
["15508491", 0.9443359375, [["G04G", "G04G 21/00", "G04G 21/02"], ["G01S", "G01S 19/00", "G01S 19/01"], ["G06F", "G06F 3/00", "G06F 3/0488"], ["G06F", "G06F 3/00", "G06F 3/0354"], ["G06F", "G06F 3/00", "G06F 3/041"], ["G04G", "G04G 17/00", "G04G 17/08"], ["G06F", "G06F 1/00", "G06F 1/16"]]],
["15428852", 0.94384765625, [["G03G", "G03G 15/00", "G03G 15/00"], ["H04N", "H04N 1/00", "H04N 1/06"], ["H04N", "H04N 1/00", "H04N 1/00"], ["H04N", "H04N 1/00", "H04N 1/29"]]],
["15670090", 0.94384765625, [["A61B", "A61B 5/00", "A61B 5/00"]]],
["15575874", 0.94384765625, [["A61B", "A61B 6/00", "A61B 6/00"], ["H04Q", "H04Q 9/00", "H04Q 9/00"]]],
["15633272", 0.94384765625, [["G06F", "G06F 1/00", "G06F 1/16"]]],
["15642418", 0.94384765625, [["H04N", "H04N 1/00", "H04N 1/00"]]],
["15837219", 0.94384765625, [["A61B", "A61B 18/00", "A61B 18/14"], ["A61B", "A61B 17/00", "A61B 17/32"], ["A61N", "A61N 7/00", "A61N 7/02"]]],
["15325869", 0.943359375, [["B41J", "B41J 11/00", "B41J 11/70"]]],
["15544169", 0.943359375, [["G03G", "G03G 15/00", "G03G 15/00"], ["B65H", "B65H 31/00", "B65H 31/02"], ["B41J", "B41J 29/00", "B41J 29/38"]]],
["15560974", 0.943359375, [["B41J", "B41J 3/00", "B41J 3/36"], ["B41J", "B41J 29/00", "B41J 29/13"]]],
["15542576", 0.94287109375, [["H04N", "H04N 5/00", null], ["H04N", "H04N 5/00", null], ["H04N", "H04N 5/00", "H04N 5/262"]]],
["15426608", 0.94287109375, [["A61B", "A61B 8/00", "A61B 8/08"], ["A61B", "A61B 8/00", "A61B 8/14"], ["A61B", "A61B 8/00", "A61B 8/00"]]],
["15463642", 0.94287109375, [["A61M", "A61M 5/00", "A61M 5/00"], ["B65D", "B65D 25/00", "B65D 25/10"], ["B65D", "B65D 77/00", "B65D 77/20"], ["B65B", "B65B 7/00", "B65B 7/28"], ["B65B", "B65B 55/00", "B65B 55/10"], ["B65B", "B65B 55/00", "B65B 55/18"], ["B65D", "B65D 1/00", "B65D 1/22"], ["B65D", "B65D 43/00", "B65D 43/02"]]],
["15562876", 0.94287109375, [["A61B", "A61B 6/00", "A61B 6/04"], ["A61B", "A61B 5/00", "A61B 5/055"]]],
["15434434", 0.94287109375, [["A61B", "A61B 90/00", "A61B 90/00"], ["B25J", "B25J 9/00", "B25J 9/16"], ["A61B", "A61B 90/00", "A61B 90/25"]]],
["15329763", 0.94287109375, [["A61B", "A61B 34/00", "A61B 34/10"], ["A61B", "A61B 34/00", "A61B 34/00"], ["G06T", "G06T 3/00", "G06T 3/00"], ["G06K", null, null], ["G06T", "G06T 3/00", "G06T 3/60"], ["G06T", "G06T 11/00", "G06T 11/60"], ["A61B", "A61B 17/00", "A61B 17/56"], ["A61B", "A61B 90/00", "A61B 90/00"]]],
["15847467", 0.94287109375, [["H04N", "H04N 5/00", null]]],
["15415955", 0.94287109375, [["G06K", "G06K 7/00", "G06K 7/10"], ["G06K", "G06K 15/00", "G06K 15/02"], ["H04N", "H04N 1/00", "H04N 1/00"]]],
["15583039", 0.94287109375, [["G06F", "G06F 3/00", "G06F 3/0484"], ["G06F", "G06F 3/00", "G06F 3/0481"]]],
["15448137", 0.9423828125, [["H04N", "H04N 1/00", "H04N 1/00"]]],
["15736634", 0.9423828125, [["A61B", "A61B 90/00", "A61B 90/25"], ["G02B", "G02B 21/00", "G02B 21/00"], ["G02B", "G02B 21/00", "G02B 21/24"]]],
["15455135", 0.9423828125, [["G06F", "G06F 3/00", "G06F 3/0484"], ["D05B", "D05B 19/00", "D05B 19/00"], ["G06F", "G06F 3/00", "G06F 3/0481"], ["G06F", "G06F 3/00", "G06F 3/0488"], ["G06F", "G06F 3/00", "G06F 3/0482"]]],
["15328181", 0.9423828125, [["G02B", "G02B 7/00", "G02B 7/00"], ["G03B", "G03B 13/00", "G03B 13/36"], ["A61B", "A61B 90/00", "A61B 90/25"], ["G02B", "G02B 21/00", "G02B 21/36"], ["A61B", "A61B 90/00", "A61B 90/20"], ["H04N", "H04N 5/00", null], ["G02B", "G02B 21/00", "G02B 21/00"]]],
["15499173", 0.9423828125, [["G03G", "G03G 15/00", "G03G 15/00"]]],
["15552889", 0.9423828125, [["A61M", "A61M 5/00", "A61M 5/148"], ["A61M", "A61M 5/00", "A61M 5/168"], ["A61M", "A61M 5/00", "A61M 5/145"]]],
["15849832", 0.9423828125, [["A61B", "A61B 7/00", "A61B 7/00"], ["A61B", "A61B 5/00", "A61B 5/00"]]],
["15828943", 0.9423828125, [["A61B", "A61B 8/00", "A61B 8/00"], ["F16M", "F16M 11/00", "F16M 11/06"], ["A61B", "A61B 50/00", "A61B 50/13"], ["A61B", "A61B 90/00", "A61B 90/50"]]],
["15527327", 0.9423828125, [["G06F", "G06F 1/00", "G06F 1/16"], ["G06F", "G06F 3/00", "G06F 3/041"]]],
["15502011", 0.9423828125, [["A61B", null, null], ["A61B", "A61B 5/00", "A61B 5/00"]]],
["15855390", 0.9423828125, [["A61B", "A61B 6/00", "A61B 6/00"]]],
["15500924", 0.94189453125, [["B41J", "B41J 3/00", "B41J 3/36"]]],
["15787479", 0.94189453125, [["A61B", "A61B 6/00", "A61B 6/00"], ["A61B", "A61B 6/00", "A61B 6/03"]]],
["15678329", 0.94189453125, [["A61M", "A61M 5/00", "A61M 5/50"], ["A61M", "A61M 5/00", "A61M 5/142"], ["G01D", "G01D 5/00", "G01D 5/25"], ["A61M", "A61M 5/00", "A61M 5/145"]]],
["15710653", 0.94189453125, [["B41J", "B41J 29/00", "B41J 29/13"], ["B41J", "B41J 29/00", "B41J 29/54"]]],
["15535130", 0.94189453125, [["G06T", "G06T 11/00", "G06T 11/60"], ["G02B", "G02B 27/00", "G02B 27/01"], ["G06F", "G06F 3/00", "G06F 3/01"]]],
["15531083", 0.94189453125, [["A61B", "A61B 90/00", "A61B 90/00"], ["H04N", null, null], ["H04N", null, null], ["G02B", "G02B 27/00", "G02B 27/01"], ["H04N", "H04N 13/00", "H04N 13/00"], ["H04N", "H04N 5/00", "H04N 5/262"], ["G06F", "G06F 3/00", "G06F 3/16"], ["G06F", "G06F 3/00", "G06F 3/01"]]],
["15567211", 0.94189453125, [["G02B", "G02B 21/00", "G02B 21/00"], ["A61B", "A61B 90/00", "A61B 90/25"], ["A61B", "A61B 90/00", "A61B 90/00"]]],
["15633987", 0.94189453125, [["G09G", "G09G 3/00", "G09G 3/20"], ["G06F", "G06F 3/00", "G06F 3/041"], ["G06F", "G06F 3/00", "G06F 3/02"]]],
["15516850", 0.94189453125, [["H01Q", "H01Q 1/00", "H01Q 1/22"], ["H01Q", "H01Q 21/00", "H01Q 21/28"], ["A61N", "A61N 1/00", "A61N 1/372"]]],
["15576250", 0.94189453125, [["B41J", "B41J 29/00", "B41J 29/00"], ["G04G", "G04G 17/00", "G04G 17/08"], ["G03G", "G03G 21/00", "G03G 21/00"]]]
]},
{"application_number": "202514126161", "offline_check": {"nearest_matches_offline": true, "nearest_similarity_matches_offline": true, "first_matches_offline": true, "probability_matches_offline": true}, "expected": [{"k": 100, "temperature": 0.005, "beta": 1.0, "first": ["C25D 3/46"], "probability": {"subclass": 0.38011852712839045, "maingroup": 0.31063663547870224, "exact": 0.10161607506731603}, "top_share": {"subclass": {"C25D": 0.3801, "C23C": 0.3408, "C22C": 0.1426, "C09D": 0.1399, "B32B": 0.1385, "C23F": 0.1216, "H01R": 0.1016, "H01H": 0.0984, "A44B": 0.0663, "H05K": 0.0524}, "maingroup": {"C25D 3/00": 0.3106, "C23C 22/00": 0.268, "C25D 5/00": 0.1983, "C25D 7/00": 0.1744, "B32B 15/00": 0.1358, "C09D 5/00": 0.1212, "C22C 9/00": 0.1055, "C23F 1/00": 0.102, "H01R 13/00": 0.1016, "H01R 4/00": 0.0984}, "exact": {"B32B 15/01": 0.1199, "C25D 7/00": 0.1167, "C09D 5/08": 0.1138, "C23C 22/83": 0.1045, "C25D 3/46": 0.1016, "H01R 13/03": 0.1016, "H01R 4/58": 0.0984, "H01H 1/025": 0.0984, "C25D 5/00": 0.0984, "C09D 1/00": 0.0831}}, "ranked_top10": {"subclass": ["C25D", "C23C", "C22C", "C09D", "B32B", "C23F", "H01R", "H01H", "A44B", "H05K"], "maingroup": ["C25D 3/00", "C23C 22/00", "C25D 5/00", "C25D 7/00", "B32B 15/00", "C09D 5/00", "C22C 9/00", "C23F 1/00", "H01R 13/00", "H01R 4/00"], "exact": ["B32B 15/01", "C25D 7/00", "C09D 5/08", "C23C 22/83", "C25D 3/46", "H01R 13/03", "H01R 4/58", "H01H 1/025", "C25D 5/00", "C09D 1/00"]}}, {"k": 30, "temperature": 0.01, "beta": 0.5, "first": ["C25D 3/46"], "probability": {"subclass": 0.3387245197908873, "maingroup": 0.26907321223184905, "exact": 0.07026516485092103}, "top_share": {"subclass": {"C25D": 0.3387, "C23C": 0.3062, "C23F": 0.1536, "C22C": 0.0963, "C09D": 0.061, "A44B": 0.0606, "C10M": 0.0431, "H05K": 0.0424, "H01R": 0.0351, "H01H": 0.0351}, "maingroup": {"C25D 3/00": 0.2691, "C23C 22/00": 0.2451, "C23F 1/00": 0.1329, "C22C 9/00": 0.0802, "C25D 5/00": 0.0789, "C25D 9/00": 0.0767, "C25D 7/00": 0.0755, "C23C 18/00": 0.075, "C09D 5/00": 0.061, "A44B 19/00": 0.0606}, "exact": {"C23C 22/83": 0.1243, "C25D 3/46": 0.0703, "C25D 3/62": 0.0694, "C23C 22/66": 0.0637, "C25D 3/38": 0.0584, "C23F 1/26": 0.0499, "C09D 5/08": 0.0465, "C23C 22/22": 0.0431, "C10M 125/10": 0.0431, "A44B 19/24": 0.0411}}, "ranked_top10": {"subclass": ["C25D", "C23C", "C23F", "C22C", "C09D", "A44B", "C10M", "H05K", "H01R", "H01H"], "maingroup": ["C25D 3/00", "C23C 22/00", "C23F 1/00", "C22C 9/00", "C25D 5/00", "C25D 9/00", "C25D 7/00", "C23C 18/00", "C09D 5/00", "A44B 19/00"], "exact": ["C23C 22/83", "C25D 3/46", "C25D 3/62", "C23C 22/66", "C25D 3/38", "C23F 1/26", "C09D 5/08", "C23C 22/22", "C10M 125/10", "A44B 19/24"]}}], "neighbours": [
["15542946", 0.95166015625, [["C25D", "C25D 3/00", "C25D 3/46"], ["H01R", "H01R 4/00", "H01R 4/58"], ["H01H", "H01H 1/00", "H01H 1/025"], ["C25D", "C25D 7/00", "C25D 7/00"], ["C25D", "C25D 5/00", "C25D 5/00"], ["H01R", "H01R 13/00", "H01R 13/03"], ["B32B", "B32B 15/00", "B32B 15/01"]]],
["15569224", 0.95068359375, [["C23C", "C23C 22/00", "C23C 22/66"], ["C09D", "C09D 5/00", "C09D 5/08"], ["C09D", "C09D 1/00", "C09D 1/00"]]],
["15550752", 0.9482421875, [["C23F", "C23F 1/00", "C23F 1/26"], ["C23F", "C23F 1/00", "C23F 1/44"]]],
["15526140", 0.947265625, [["C23C", "C23C 22/00", "C23C 22/83"], ["C23C", "C23C 22/00", "C23C 22/60"]]],
["15562653", 0.947265625, [["C23C", "C23C 22/00", "C23C 22/83"], ["C23C", "C23C 22/00", "C23C 22/07"], ["C23C", "C23C 22/00", "C23C 22/82"]]],
["15667972", 0.94677734375, [["C23C", "C23C 22/00", "C23C 22/22"]]],
["15569239", 0.94677734375, [["C10M", "C10M 125/00", "C10M 125/10"], ["C10M", "C10M 173/00", "C10M 173/00"], ["C10M", "C10M 125/00", "C10M 125/30"], ["C10M", "C10M 129/00", "C10M 129/34"], ["C10M", "C10M 159/00", "C10M 159/06"]]],
["15552455", 0.9462890625, [["A44B", "A44B 19/00", "A44B 19/24"], ["A44B", "A44B 19/00", "A44B 19/02"], ["C22C", "C22C 9/00", "C22C 9/04"]]],
["15326328", 0.9462890625, [["C25D", "C25D 3/00", "C25D 3/58"], ["C22C", "C22C 9/00", "C22C 9/02"], ["C25D", "C25D 3/00", "C25D 3/60"]]],
["15537511", 0.94580078125, [["C22C", "C22C 9/00", "C22C 9/06"], ["A44B", "A44B 19/00", "A44B 19/60"], ["A44B", "A44B 19/00", "A44B 19/08"], ["A44B", "A44B 19/00", "A44B 19/46"], ["C22C", "C22C 9/00", "C22C 9/04"], ["B21D", "B21D 53/00", "B21D 53/54"]]],
["15738398", 0.9453125, [["C25D", "C25D 3/00", "C25D 3/62"]]],
["15740306", 0.9443359375, [["C23C", "C23C 22/00", "C23C 22/83"], ["C25D", "C25D 9/00", "C25D 9/02"], ["C23C", "C23C 22/00", "C23C 22/07"], ["C23C", "C23C 28/00", "C23C 28/00"]]],
["15472620", 0.94384765625, [["C25D", "C25D 3/00", "C25D 3/62"], ["C22C", "C22C 5/00", "C22C 5/02"], ["C25D", "C25D 7/00", "C25D 7/12"]]],
["15577949", 0.943359375, [["C25D", "C25D 3/00", "C25D 3/38"], ["C25D", "C25D 7/00", "C25D 7/12"], ["H05K", "H05K 3/00", "H05K 3/18"], ["H05K", "H05K 3/00", "H05K 3/06"], ["H05K", "H05K 1/00", "H05K 1/09"]]],
["15560847", 0.94287109375, [["C09D", "C09D 5/00", "C09D 5/24"], ["B05D", "B05D 7/00", "B05D 7/14"], ["C09D", "C09D 5/00", "C09D 5/08"], ["C09D", "C09D 175/00", "C09D 175/04"], ["C09D", "C09D 7/00", "C09D 7/61"], ["C09D", "C09D 167/00", "C09D 167/00"]]],
["15622696", 0.94287109375, [["C01G", "C01G 49/00", "C01G 49/00"]]],
["15523452", 0.9423828125, [["G03F", "G03F 7/00", "G03F 7/038"], ["H05K", "H05K 9/00", "H05K 9/00"], ["G03F", "G03F 7/00", "G03F 7/09"], ["H01B", "H01B 5/00", "H01B 5/14"], ["G03F", "G03F 7/00", "G03F 7/11"], ["H01B", "H01B 13/00", "H01B 13/00"], ["G06F", "G06F 3/00", "G06F 3/044"]]],
["15464175", 0.9423828125, [["C25D", "C25D 3/00", "C25D 3/38"], ["C25D", "C25D 5/00", "C25D 5/44"], ["C25D", "C25D 5/00", "C25D 5/02"]]],
["15533287", 0.9423828125, [["C23C", "C23C 18/00", "C23C 18/18"], ["C23F", "C23F 1/00", "C23F 1/30"], ["C23C", "C23C 22/00", "C23C 22/50"]]],
["15488523", 0.94189453125, [["C23F", "C23F 1/00", "C23F 1/30"]]],
["15537963", 0.94189453125, [["C23C", "C23C 18/00", "C23C 18/16"], ["H05K", "H05K 3/00", "H05K 3/18"], ["H05K", "H05K 3/00", "H05K 3/26"]]],
["15739052", 0.94091796875, [["C25D", "C25D 9/00", "C25D 9/08"], ["C25D", "C25D 3/00", "C25D 3/30"], ["C25D", "C25D 3/00", "C25D 3/12"], ["C25D", "C25D 5/00", "C25D 5/12"]]],
["15647351", 0.9404296875, [["C22B", "C22B 58/00", "C22B 58/00"], ["C22B", "C22B 30/00", "C22B 30/04"], ["C07C", "C07C 321/00", "C07C 321/26"], ["C07C", "C07C 321/00", "C07C 321/22"], ["C07C", "C07C 321/00", "C07C 321/04"]]],
["15531764", 0.9404296875, [["C23F", "C23F 1/00", "C23F 1/22"], ["B29C", "B29C 45/00", "B29C 45/14"], ["B29C", "B29C 45/00", "B29C 45/02"]]],
["15702654", 0.9404296875, [["C25D", "C25D 11/00", "C25D 11/24"]]],
["15428270", 0.939453125, [["C23F", "C23F 11/00", "C23F 11/04"]]],
["15324461", 0.939453125, [["C23C", "C23C 18/00", "C23C 18/24"], ["C23C", "C23C 18/00", "C23C 18/16"], ["C25F", "C25F 1/00", "C25F 1/00"], ["C23C", "C23C 18/00", "C23C 18/38"], ["C23C", "C23C 18/00", "C23C 18/50"], ["C23C", "C23C 18/00", "C23C 18/42"], ["C23C", "C23C 18/00", "C23C 18/30"], ["C23C", "C23C 18/00", "C23C 18/32"]]],
["15323856", 0.93896484375, [["C23F", "C23F 1/00", "C23F 1/28"]]],
["15578145", 0.93798828125, [["C25D", "C25D 9/00", "C25D 9/04"], ["C25D", "C25D 3/00", "C25D 3/12"], ["C25D", "C25D 3/00", "C25D 3/22"], ["C25D", "C25D 15/00", "C25D 15/00"], ["C25D", "C25D 5/00", "C25D 5/12"], ["C25D", "C25D 5/00", "C25D 5/48"]]],
["15796830", 0.93798828125, [["C25D", "C25D 9/00", "C25D 9/04"], ["C25D", "C25D 3/00", "C25D 3/56"], ["C25D", "C25D 3/00", "C25D 3/54"], ["C25D", "C25D 5/00", "C25D 5/50"], ["C25D", "C25D 7/00", "C25D 7/12"]]],
["15538421", 0.93798828125, [["C25D", "C25D 9/00", "C25D 9/10"], ["C25D", "C25D 3/00", "C25D 3/32"]]],
["15701924", 0.93798828125, [["C23C", "C23C 22/00", "C23C 22/07"], ["C23F", "C23F 11/00", "C23F 11/167"]]],
["15802503", 0.93798828125, [["C25D", "C25D 3/00", "C25D 3/22"], ["C25D", "C25D 9/00", "C25D 9/06"], ["C25D", "C25D 17/00", "C25D 17/10"], ["C23C", "C23C 8/00", "C23C 8/10"], ["C23C", "C23C 8/00", "C23C 8/02"], ["C23C", "C23C 28/00", "C23C 28/00"], ["C23G", "C23G 1/00", "C23G 1/02"], ["F16L", "F16L 9/00", "F16L 9/02"]]],
["15502197", 0.93798828125, [["C25D", "C25D 3/00", "C25D 3/58"]]],
["15555665", 0.93798828125, [["C25D", "C25D 5/00", "C25D 5/48"], ["B21D", "B21D 22/00", "B21D 22/02"], ["C23C", "C23C 2/00", "C23C 2/06"], ["C25D", "C25D 3/00", "C25D 3/22"], ["C25D", "C25D 3/00", "C25D 3/56"]]],
["15519941", 0.93798828125, [["B32B", "B32B 15/00", "B32B 15/20"], ["C22C", "C22C 21/00", "C22C 21/10"], ["B32B", "B32B 15/00", "B32B 15/08"], ["B32B", "B32B 9/00", "B32B 9/04"], ["C22C", "C22C 21/00", "C22C 21/08"], ["C23C", "C23C 28/00", "C23C 28/04"], ["C22C", "C22C 21/00", "C22C 21/16"]]],
["15524800", 0.93798828125, [["C25D", "C25D 5/00", "C25D 5/02"], ["C25F", "C25F 3/00", "C25F 3/20"], ["C25F", "C25F 3/00", "C25F 3/24"]]],
["15497224", 0.9375, [["C11D", "C11D 3/00", "C11D 3/04"], ["C11D", "C11D 3/00", "C11D 3/30"], ["C11D", "C11D 17/00", "C11D 17/08"], ["C11D", "C11D 3/00", "C11D 3/26"]]],
["15483122", 0.9375, [["C25D", "C25D 3/00", "C25D 3/10"], ["C25D", "C25D 3/00", "C25D 3/08"], ["C25D", "C25D 3/00", "C25D 3/06"]]],
["15737487", 0.93701171875, [["C25D", "C25D 3/00", "C25D 3/12"], ["C25D", "C25D 5/00", "C25D 5/50"], ["C25D", "C25D 7/00", "C25D 7/00"], ["B32B", "B32B 15/00", "B32B 15/01"], ["B32B", "B32B 15/00", "B32B 15/18"], ["B65D", "B65D 65/00", "B65D 65/42"]]],
["15558479", 0.93701171875, [["B05D", "B05D 7/00", "B05D 7/24"], ["B05D", "B05D 5/00", "B05D 5/08"], ["C09D", "C09D 201/00", "C09D 201/00"], ["C09D", "C09D 5/00", "C09D 5/02"], ["C09D", "C09D 5/00", "C09D 5/16"], ["C09D", "C09D 7/00", null], ["F21S", "F21S 8/00", null], ["F21V", "F21V 3/00", "F21V 3/04"]]],
["15532196", 0.9365234375, [["F16J", "F16J 15/00", "F16J 15/08"], ["C09K", "C09K 3/00", "C09K 3/10"], ["F02F", "F02F 11/00", "F02F 11/00"]]],
["15539652", 0.9365234375, [["C09D", "C09D 175/00", "C09D 175/04"], ["C08G", "C08G 18/00", "C08G 18/50"], ["C09D", "C09D 5/00", "C09D 5/08"]]],
["15413741", 0.93603515625, [["H05K", "H05K 3/00", "H05K 3/18"], ["C23C", "C23C 18/00", "C23C 18/20"], ["C23C", "C23C 18/00", "C23C 18/40"], ["B29B", "B29B 9/00", "B29B 9/12"], ["B29C", "B29C 45/00", "B29C 45/37"], ["B29C", "B29C 59/00", "B29C 59/02"], ["C23C", "C23C 18/00", "C23C 18/30"], ["B29C", "B29C 45/00", "B29C 45/00"]]],
["15511023", 0.93603515625, [["C09D", "C09D 183/00", "C09D 183/08"], ["C08G", "C08G 77/00", "C08G 77/28"], ["C08G", "C08G 77/00", "C08G 77/30"], ["C08L", "C08L 83/00", "C08L 83/06"], ["C08L", "C08L 83/00", "C08L 83/08"]]],
["15606947", 0.93603515625, [["C09K", "C09K 13/00", "C09K 13/00"], ["C08K", "C08K 5/00", "C08K 5/42"], ["C08K", "C08K 5/00", "C08K 5/092"]]],
["15564098", 0.93603515625, [["H01M", "H01M 8/00", "H01M 8/0202"], ["C25D", "C25D 5/00", "C25D 5/36"], ["C25D", "C25D 7/00", "C25D 7/00"], ["H01M", "H01M 8/00", "H01M 8/1018"]]],
["15791490", 0.935546875, [["H01M", null, null]]],
["15546624", 0.935546875, [["C25D", "C25D 9/00", "C25D 9/12"], ["C25D", "C25D 5/00", "C25D 5/48"], ["B32B", "B32B 15/00", "B32B 15/18"], ["C25D", "C25D 7/00", "C25D 7/00"], ["B32B", "B32B 15/00", "B32B 15/04"]]],
["15675833", 0.935546875, [["B05D", "B05D 1/00", "B05D 1/02"], ["B05D", "B05D 1/00", "B05D 1/28"], ["B05D", "B05D 3/00", "B05D 3/00"], ["B05D", "B05D 1/00", "B05D 1/18"]]],
["15466282", 0.935546875, [["C25D", "C25D 21/00", "C25D 21/10"], ["C25D", "C25D 5/00", "C25D 5/54"], ["C25D", "C25D 7/00", "C25D 7/12"], ["C25D", "C25D 3/00", "C25D 3/38"], ["C25D", "C25D 5/00", "C25D 5/02"]]],
["15803876", 0.935546875, []],
["15325276", 0.93505859375, [["C09D", "C09D 4/00", "C09D 4/00"], ["B05D", "B05D 7/00", "B05D 7/00"], ["B05D", "B05D 5/00", "B05D 5/00"], ["C09K", "C09K 3/00", "C09K 3/00"]]],
["15324553", 0.93505859375, [["H01J", "H01J 37/00", "H01J 37/305"], ["H01J", "H01J 37/00", "H01J 37/08"]]],
["15630372", 0.9345703125, [["C23C", "C23C 18/00", "C23C 18/16"], ["C25C", "C25C 1/00", "C25C 1/12"], ["C23C", "C23C 18/00", "C23C 18/38"], ["C23C", "C23C 18/00", "C23C 18/18"], ["C25D", "C25D 21/00", "C25D 21/22"], ["C25C", "C25C 1/00", "C25C 1/06"]]],
["15728175", 0.9345703125, [["C25D", "C25D 3/00", "C25D 3/28"], ["C25D", "C25D 17/00", "C25D 17/02"], ["C25D", "C25D 17/00", "C25D 17/10"], ["C25D", "C25D 21/00", "C25D 21/12"], ["C01G", "C01G 3/00", "C01G 3/02"]]],
["15837769", 0.9345703125, [["C25D", "C25D 5/00", "C25D 5/12"], ["H01R", "H01R 13/00", "H01R 13/03"], ["C25F", "C25F 5/00", "C25F 5/00"], ["C25F", "C25F 1/00", "C25F 1/00"], ["C25D", "C25D 5/00", "C25D 5/50"], ["C25D", "C25D 5/00", "C25D 5/48"], ["C25D", "C25D 3/00", "C25D 3/12"], ["C25D", "C25D 3/00", "C25D 3/48"], ["C25D", "C25D 3/00", "C25D 3/46"], ["C25D", "C25D 3/00", "C25D 3/40"], ["C25D", "C25D 3/00", "C25D 3/50"], ["C25D", "C25D 3/00", "C25D 3/30"]]],
["15511638", 0.9345703125, [["C10M", "C10M 103/00", "C10M 103/02"], ["C23C", "C23C 14/00", "C23C 14/06"], ["C23C", "C23C 14/00", "C23C 14/32"]]],
["15564406", 0.9345703125, [["H01M", null, null], ["H01G", "H01G 9/00", "H01G 9/035"], ["H01G", "H01G 9/00", "H01G 9/08"], ["H01G", "H01G 9/00", "H01G 9/145"], ["H01M", "H01M 10/00", "H01M 10/0525"]]],
["15572604", 0.9345703125, [["F16B", "F16B 2/00", "F16B 2/08"], ["C23C", "C23C 18/00", "C23C 18/16"], ["C23C", "C23C 22/00", "C23C 22/82"], ["C25D", "C25D 3/00", "C25D 3/56"], ["C25D", "C25D 5/00", "C25D 5/48"], ["F16D", "F16D 3/00", "F16D 3/84"]]],
["15563208", 0.9345703125, [["C25D", "C25D 5/00", "C25D 5/12"], ["C25D", "C25D 5/00", "C25D 5/50"], ["C25D", "C25D 9/00", "C25D 9/08"], ["C25D", "C25D 7/00", "C25D 7/06"], ["B32B", "B32B 15/00", "B32B 15/01"]]],
["15554645", 0.9345703125, [["C09D", "C09D 11/00", "C09D 11/324"], ["C09D", "C09D 11/00", "C09D 11/037"], ["C09C", "C09C 1/00", "C09C 1/56"]]],
["15654294", 0.93408203125, [["C23C", "C23C 22/00", "C23C 22/36"], ["C23C", "C23C 22/00", "C23C 22/17"]]],
["15548038", 0.93408203125, [["C23C", "C23C 2/00", "C23C 2/26"], ["B32B", "B32B 15/00", "B32B 15/01"], ["B32B", "B32B 15/00", "B32B 15/18"], ["C25D", "C25D 5/00", "C25D 5/48"], ["C22C", "C22C 18/00", "C22C 18/04"], ["C25D", "C25D 3/00", "C25D 3/22"], ["C25D", "C25D 3/00", "C25D 3/56"], ["C23C", "C23C 2/00", "C23C 2/06"], ["C23C", "C23C 28/00", "C23C 28/00"]]],
["15551052", 0.93408203125, [["C25D", "C25D 21/00", "C25D 21/11"], ["C09K", "C09K 15/00", "C09K 15/26"], ["C08J", "C08J 7/00", "C08J 7/06"], ["C25D", "C25D 5/00", "C25D 5/56"], ["C25D", "C25D 5/00", "C25D 5/34"], ["C23C", "C23C 18/00", "C23C 18/38"], ["C23C", "C23C 18/00", "C23C 18/16"], ["C23C", "C23C 18/00", "C23C 18/20"], ["C23C", "C23C 18/00", "C23C 18/22"], ["C23C", "C23C 18/00", "C23C 18/30"], ["C23C", "C23C 18/00", "C23C 18/32"], ["C09K", "C09K 15/00", "C09K 15/30"], ["C25D", "C25D 17/00", "C25D 17/08"]]],
["15519474", 0.93408203125, [["C25D", "C25D 21/00", "C25D 21/14"], ["C25D", "C25D 3/00", "C25D 3/58"], ["C22C", "C22C 19/00", "C22C 19/00"], ["C22C", "C22C 9/00", "C22C 9/06"], ["C22C", "C22C 19/00", "C22C 19/03"], ["C25D", "C25D 3/00", "C25D 3/56"], ["C25D", "C25D 17/00", "C25D 17/00"]]],
["15504225", 0.93408203125, [["C21D", "C21D 9/00", "C21D 9/56"], ["C23C", "C23C 4/00", "C23C 4/18"], ["C23C", "C23C 4/00", "C23C 4/11"], ["C23C", "C23C 22/00", "C23C 22/33"], ["C23C", "C23C 22/00", "C23C 22/30"]]],
["15555401", 0.93408203125, [["B23K", "B23K 11/00", "B23K 11/36"], ["B23K", "B23K 31/00", "B23K 31/10"], ["C23C", "C23C 22/00", "C23C 22/12"], ["B21C", "B21C 37/00", "B21C 37/08"], ["B21C", "B21C 37/00", "B21C 37/30"], ["B23K", "B23K 11/00", "B23K 11/087"], ["F16L", "F16L 9/00", "F16L 9/17"]]],
["15548514", 0.93408203125, [["B32B", "B32B 15/00", "B32B 15/14"], ["B32B", "B32B 15/00", "B32B 15/095"], ["B32B", "B32B 5/00", "B32B 5/02"], ["C09D", "C09D 175/00", "C09D 175/04"]]],
["15546203", 0.93359375, [["C25D", "C25D 5/00", "C25D 5/10"], ["C25D", "C25D 9/00", "C25D 9/10"], ["C25D", "C25D 5/00", "C25D 5/48"], ["C25D", "C25D 11/00", "C25D 11/36"], ["B32B", "B32B 9/00", "B32B 9/04"]]],
["15807577", 0.93359375, [["B22F", "B22F 1/00", "B22F 1/00"], ["B22F", "B22F 1/00", null], ["C23C", "C23C 26/00", "C23C 26/00"], ["C09B", "C09B 67/00", "C09B 67/02"], ["G06F", "G06F 3/00", "G06F 3/041"], ["B82Y", "B82Y 30/00", "B82Y 30/00"]]],
["15329809", 0.93359375, [["C09K", "C09K 9/00", "C09K 9/00"], ["C01G", "C01G 31/00", "C01G 31/02"], ["C09C", "C09C 3/00", "C09C 3/08"], ["C09C", "C09C 1/00", "C09C 1/00"], ["C09C", "C09C 3/00", "C09C 3/12"]]],
["15554903", 0.93359375, [["B32B", "B32B 15/00", "B32B 15/01"], ["C23C", "C23C 2/00", "C23C 2/06"], ["C23C", "C23C 2/00", "C23C 2/28"], ["B05D", "B05D 1/00", "B05D 1/28"], ["B05D", "B05D 3/00", "B05D 3/02"], ["B05D", "B05D 7/00", "B05D 7/14"], ["C22C", "C22C 38/00", "C22C 38/54"], ["C22C", "C22C 38/00", "C22C 38/50"], ["C22C", "C22C 38/00", "C22C 38/26"], ["C22C", "C22C 38/00", "C22C 38/22"], ["C22C", "C22C 38/00", "C22C 38/06"], ["C22C", "C22C 38/00", "C22C 38/04"], ["C22C", "C22C 38/00", "C22C 38/02"], ["C22C", "C22C 38/00", "C22C 38/00"]]],
["15446192", 0.93359375, [["B08B", "B08B 5/00", "B08B 5/00"], ["B08B", "B08B 9/00", "B08B 9/027"]]],
["15567484", 0.93359375, [["B32B", "B32B 15/00", "B32B 15/08"], ["C23F", "C23F 1/00", "C23F 1/28"], ["B29C", "B29C 45/00", "B29C 45/14"], ["C25D", "C25D 5/00", "C25D 5/48"]]],
["15473696", 0.93359375, [["G02B", "G02B 7/00", "G02B 7/02"], ["C22C", "C22C 9/00", "C22C 9/00"], ["G02B", "G02B 7/00", "G02B 7/10"], ["C25D", "C25D 7/00", "C25D 7/06"], ["B32B", "B32B 15/00", "B32B 15/01"], ["B23K", "B23K 1/00", "B23K 1/00"], ["C25D", "C25D 3/00", "C25D 3/30"]]],
["15546886", 0.93359375, [["C23C", "C23C 22/00", "C23C 22/42"], ["C25D", "C25D 5/00", "C25D 5/48"], ["C25D", "C25D 5/00", "C25D 5/12"], ["C23C", "C23C 28/00", "C23C 28/00"], ["B32B", "B32B 15/00", "B32B 15/18"], ["C25D", "C25D 7/00", "C25D 7/00"], ["B32B", "B32B 15/00", "B32B 15/01"]]],
["15736668", 0.93310546875, [["C23C", "C23C 14/00", "C23C 14/06"], ["B21D", "B21D 37/00", "B21D 37/01"], ["B21D", "B21D 37/00", "B21D 37/20"], ["B21D", "B21D 37/00", "B21D 37/10"]]],
["15599473", 0.93310546875, [["C23C", "C23C 18/00", "C23C 18/02"], ["C23C", "C23C 18/00", "C23C 18/06"], ["C23C", "C23C 18/00", "C23C 18/04"], ["C23C", "C23C 18/00", "C23C 18/12"], ["C09D", "C09D 11/00", "C09D 11/101"]]],
["15561155", 0.93310546875, [["C25D", "C25D 3/00", "C25D 3/32"], ["C25D", "C25D 3/00", "C25D 3/60"]]],
["15510466", 0.93310546875, [["C23C", "C23C 22/00", "C23C 22/77"], ["G01J", "G01J 3/00", "G01J 3/46"], ["G01J", "G01J 3/00", "G01J 3/50"]]],
["15739268", 0.93310546875, [["C23C", "C23C 18/00", "C23C 18/32"], ["C23C", "C23C 18/00", "C23C 18/16"], ["C22C", "C22C 23/00", "C22C 23/06"], ["F04D", "F04D 29/00", "F04D 29/26"], ["C23C", "C23C 28/00", "C23C 28/02"], ["C23C", "C23C 28/00", "C23C 28/00"], ["C23C", "C23C 22/00", "C23C 22/18"]]],
["15562826", 0.93310546875, [["C09D", "C09D 5/00", "C09D 5/08"], ["C09D", "C09D 5/00", "C09D 5/03"], ["C23F", "C23F 11/00", "C23F 11/16"], ["C23F", "C23F 11/00", "C23F 11/14"], ["C23F", "C23F 11/00", "C23F 11/18"], ["C09D", "C09D 163/00", "C09D 163/00"]]],
["15502137", 0.9326171875, [["C09C", "C09C 1/00", "C09C 1/28"], ["C09C", "C09C 3/00", "C09C 3/00"], ["F28F", "F28F 13/00", "F28F 13/18"], ["C09C", "C09C 3/00", "C09C 3/12"], ["C09K", "C09K 3/00", "C09K 3/00"], ["C09C", "C09C 1/00", "C09C 1/40"], ["C09C", "C09C 3/00", "C09C 3/10"]]],
["15598588", 0.9326171875, [["C25D", "C25D 5/00", "C25D 5/48"], ["H01B", "H01B 1/00", "H01B 1/08"], ["H01B", "H01B 13/00", "H01B 13/00"], ["C25D", "C25D 5/00", "C25D 5/34"], ["C25D", "C25D 3/00", "C25D 3/38"], ["C25D", "C25D 9/00", "C25D 9/04"]]],
["15497776", 0.9326171875, [["C23C", "C23C 22/00", "C23C 22/77"], ["C23C", "C23C 22/00", "C23C 22/33"], ["C23C", "C23C 22/00", "C23C 22/27"], ["C09D", "C09D 5/00", "C09D 5/08"], ["C23C", "C23C 22/00", "C23C 22/30"]]],
["15518130", 0.9326171875, [["C23F", "C23F 1/00", "C23F 1/18"], ["C23F", "C23F 1/00", "C23F 1/30"]]],
["15573758", 0.9326171875, [["C10M", "C10M 105/00", "C10M 105/26"], ["C10M", "C10M 173/00", "C10M 173/02"], ["C10M", "C10M 159/00", "C10M 159/06"], ["C10M", "C10M 125/00", "C10M 125/26"], ["C10M", "C10M 125/00", "C10M 125/02"], ["C10M", "C10M 125/00", "C10M 125/22"], ["C10M", "C10M 125/00", "C10M 125/18"], ["C10M", "C10M 125/00", "C10M 125/10"], ["C10M", "C10M 125/00", "C10M 125/24"]]],
["15532490", 0.9326171875, [["C09D", "C09D 5/00", "C09D 5/02"], ["B05D", "B05D 1/00", "B05D 1/02"], ["B05D", "B05D 7/00", "B05D 7/24"], ["C09D", "C09D 167/00", "C09D 167/08"]]],
["15691570", 0.9326171875, [["C09D", "C09D 5/00", "C09D 5/08"], ["C09D", "C09D 7/00", null], ["C08K", "C08K 5/00", "C08K 5/372"]]],
["15713000", 0.9326171875, [["C09D", "C09D 5/00", "C09D 5/08"], ["B05D", "B05D 3/00", "B05D 3/00"], ["C09D", "C09D 1/00", "C09D 1/00"], ["C01B", "C01B 25/00", "C01B 25/36"]]],
["15474054", 0.9326171875, [["C09D", "C09D 7/00", null], ["C09D", "C09D 167/00", "C09D 167/02"], ["C09D", "C09D 5/00", "C09D 5/36"], ["C09D", "C09D 133/00", "C09D 133/00"], ["C09D", "C09D 5/00", "C09D 5/08"], ["C25D", "C25D 3/00", "C25D 3/56"], ["C09D", "C09D 175/00", "C09D 175/04"]]],
["15562805", 0.9326171875, [["C23C", "C23C 28/00", "C23C 28/00"], ["C23C", "C23C 30/00", "C23C 30/00"], ["C25D", "C25D 9/00", "C25D 9/08"], ["C25D", "C25D 5/00", "C25D 5/48"]]],
["15588841", 0.9326171875, [["C25D", "C25D 5/00", "C25D 5/22"], ["A44B", "A44B 1/00", "A44B 1/02"], ["A44B", "A44B 1/00", "A44B 1/06"], ["A44B", "A44B 17/00", "A44B 17/00"], ["C25D", "C25D 21/00", "C25D 21/10"], ["A44B", "A44B 19/00", "A44B 19/42"]]],
["15417837", 0.9326171875, [["C23F", "C23F 11/00", "C23F 11/06"], ["C23C", "C23C 22/00", "C23C 22/62"], ["C23C", "C23C 22/00", "C23C 22/82"], ["C23C", "C23C 22/00", "C23C 22/80"]]],
["15448066", 0.9326171875, [["C23C", "C23C 18/00", "C23C 18/16"], ["C25D", "C25D 3/00", "C25D 3/12"], ["C09D", "C09D 11/00", "C09D 11/03"], ["C09D", "C09D 11/00", "C09D 11/52"], ["C09D", "C09D 177/00", "C09D 177/10"], ["C09D", "C09D 169/00", "C09D 169/00"], ["C09D", "C09D 167/00", "C09D 167/02"], ["C09D", "C09D 11/00", "C09D 11/38"], ["C23C", "C23C 18/00", "C23C 18/20"], ["C09D", null, null]]],
["15512458", 0.9326171875, [["C22F", "C22F 1/00", "C22F 1/10"], ["B21J", "B21J 5/00", "B21J 5/02"], ["C21D", "C21D 1/00", "C21D 1/68"], ["C22C", "C22C 19/00", "C22C 19/05"], ["B21J", "B21J 3/00", "B21J 3/00"]]],
["15520468", 0.9326171875, [["C09K", "C09K 13/00", "C09K 13/02"]]],
["15678884", 0.9326171875, [["C25D", "C25D 7/00", "C25D 7/00"], ["C25D", "C25D 3/00", "C25D 3/54"], ["C25D", "C25D 9/00", "C25D 9/10"], ["C25D", "C25D 21/00", "C25D 21/12"]]],
["15473721", 0.9326171875, [["G02B", "G02B 7/00", "G02B 7/02"], ["C22C", "C22C 9/00", "C22C 9/00"], ["G02B", "G02B 7/00", "G02B 7/10"], ["C25D", "C25D 7/00", "C25D 7/06"], ["B32B", "B32B 15/00", "B32B 15/01"], ["B23K", "B23K 1/00", "B23K 1/00"], ["C25D", "C25D 3/00", "C25D 3/30"]]]
]}
]
""")
# fmt: on
