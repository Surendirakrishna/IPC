"""Turn the nearest labelled patents into ranked IPC codes with vote shares (ADR-0005).

Port of ``vote`` in data/manifests/benchmarks/knn_hupd/knn_eval.py, kept numerically identical:

- neighbour n weighs ``exp((s_n - s_1) / T)``, with s_1 the highest similarity;
- its main label counts 1 and each secondary label ``beta``;
- at each level (subclass, main group, exact code) a neighbour counts once per code, with its best
  label weight for that code;
- a code's share is its weighted count over the total weight of all neighbours (labelled or not), so
  it reads as "the weighted share of similar patents that carry this code". Patents carry several
  codes, so shares of different codes do not sum to 1;
- the first code is chosen top-down: the best subclass, then the best main group inside it, then the
  best exact code inside that main group.

The "exact" level holds each label's exact code, which is a subgroup ("H04L 9/08") or, when a patent
is labelled with a main group itself, that main group ("H04L 9/00"). So the first code can be a main
group chosen at the exact level; ``first_code_level`` follows the code's shape, as the contract does.

Ties in share keep a deterministic order: the code first carried by the nearer neighbour (then by its
earlier label) ranks first. That is the offline behaviour (a stable sort over insertion order).

Shares are kept unrounded; the contract mapping (TASK-0010) rounds to 4 decimals. The class-level
share is not computed here (TASK-0010).

``vote`` is pure: no I/O, no globals, same inputs give the same result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

Level = Literal["subclass", "maingroup", "exact"]
FirstCodeLevel = Literal["subclass", "maingroup", "subgroup"]
LEVELS: tuple[Level, ...] = ("subclass", "maingroup", "exact")
DEFAULT_TOP_N = 10  # contract MAX_TOP_CODES_PER_LEVEL and MAX_NEIGHBOURS


class NeighbourLike(Protocol):
    """What the vote reads from a neighbour (``retrieval.patent_search.PatentNeighbour`` fits)."""

    @property
    def document_id(self) -> str: ...

    @property
    def title(self) -> str | None: ...

    @property
    def similarity(self) -> float: ...

    @property
    def labels(self) -> Sequence[Sequence[str | None]]: ...  # (subclass, main_group, exact), main label first


@dataclass(frozen=True)
class LevelShares:
    """Shares of one code path at each level; None where the level does not apply."""

    subclass: float | None = None
    maingroup: float | None = None
    exact: float | None = None


@dataclass(frozen=True)
class RankedCode:
    code: str  # catalogue display code
    share: float  # in [0, 1], unrounded


@dataclass(frozen=True)
class VotedNeighbour:
    document_id: str
    title: str | None
    similarity: float
    vote_weight: float  # normalised: the weights of all neighbours in the vote sum to 1
    labels: tuple[tuple[str, str | None, str | None], ...]  # main label first

    @property
    def ipc_codes(self) -> list[str]:
        """Deepest resolved display code per label, main label first, without duplicates."""
        return list(dict.fromkeys(ex or mg or sub for sub, mg, ex in self.labels))


@dataclass(frozen=True)
class NeighbourVote:
    """Result of ``vote``. ``first_code`` is None when no neighbour carries a resolvable label."""

    first_code: str | None
    first_code_level: FirstCodeLevel | None
    first_code_share: LevelShares
    ranked: Mapping[Level, tuple[RankedCode, ...]]  # every code with weight > 0, descending share
    neighbours: tuple[VotedNeighbour, ...]  # the neighbours in the vote, descending similarity
    k: int
    temperature: float
    beta: float
    _shares: Mapping[Level, Mapping[str, float]] = field(repr=False, compare=False, default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.first_code is None

    @property
    def neighbour_count(self) -> int:
        return len(self.neighbours)

    def top_codes(self, level: Level, n: int = DEFAULT_TOP_N) -> tuple[RankedCode, ...]:
        return self.ranked[level][:n]

    def nearest(self, n: int = DEFAULT_TOP_N) -> tuple[VotedNeighbour, ...]:
        return self.neighbours[:n]

    def share(self, level: Level, code: str) -> float:
        """Share of ``code`` at ``level``; 0.0 when no neighbour carries it."""
        return self._shares[level].get(code, 0.0)


def _empty(k: int, temperature: float, beta: float, neighbours: tuple[VotedNeighbour, ...] = ()) -> NeighbourVote:
    return NeighbourVote(None, None, LevelShares(), {lvl: () for lvl in LEVELS}, neighbours, k, temperature, beta,
                         {lvl: {} for lvl in LEVELS})  # fmt: skip


def _first_code_level(code: str) -> FirstCodeLevel:
    if "/" not in code:
        return "subclass"
    return "maingroup" if code.endswith("/00") else "subgroup"


def vote(neighbours: Sequence[NeighbourLike], k: int, temperature: float, beta: float) -> NeighbourVote:
    """Vote over the ``k`` most similar of ``neighbours``.

    Neighbours are ordered by descending similarity (a stable sort, so equal similarities keep their
    input order) and the first ``k`` vote. Fewer than ``k`` neighbours simply vote as they are: a short
    search result is an error for the caller to raise (``patent_search.NeighbourShortfall``), not here.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (temperature > 0 and math.isfinite(temperature)):
        raise ValueError("temperature must be a finite number > 0")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]: a secondary label weighing more than the main one breaks shares")
    ordered = sorted(neighbours, key=lambda n: -n.similarity)[:k]
    if not ordered:
        return _empty(k, temperature, beta)
    sims = [float(n.similarity) for n in ordered]
    if not all(math.isfinite(s) for s in sims):
        raise ValueError("neighbour similarities must be finite")
    s1 = sims[0]
    weights = [math.exp((s - s1) / temperature) for s in sims]
    total = sum(weights)

    score: dict[Level, dict[str, float]] = {lvl: {} for lvl in LEVELS}
    labels_of: list[tuple[tuple[str, str | None, str | None], ...]] = []
    for wn, n in zip(weights, ordered, strict=True):
        labels = tuple((lab[0], lab[1], lab[2]) for lab in n.labels)
        labels_of.append(labels)
        best: dict[Level, dict[str, float]] = {lvl: {} for lvl in LEVELS}
        for j, (sub, mg, ex) in enumerate(labels):
            lw = 1.0 if j == 0 else beta
            if lw == 0:
                continue
            for lvl, key in (("subclass", sub), ("maingroup", mg), ("exact", ex)):
                if key and lw > best[lvl].get(key, 0.0):
                    best[lvl][key] = lw
        for lvl in LEVELS:
            for key, lw in best[lvl].items():
                score[lvl][key] = score[lvl].get(key, 0.0) + wn * lw

    voted = tuple(
        VotedNeighbour(n.document_id, n.title, s, w / total, labels)
        for n, s, w, labels in zip(ordered, sims, weights, labels_of, strict=True)
    )
    # sorted() is stable, also with reverse=True: equal scores keep insertion order (nearest carrier first).
    order = {lvl: sorted(score[lvl], key=score[lvl].__getitem__, reverse=True) for lvl in LEVELS}
    shares = {lvl: {c: score[lvl][c] / total for c in order[lvl]} for lvl in LEVELS}
    ranked = {lvl: tuple(RankedCode(c, shares[lvl][c]) for c in order[lvl]) for lvl in LEVELS}
    if not order["subclass"]:
        return _empty(k, temperature, beta, voted)

    sub = order["subclass"][0]
    mg = next((c for c in order["maingroup"] if c.startswith(sub + " ")), None)
    ex = next((c for c in order["exact"] if c.split("/")[0] == mg.split("/")[0]), None) if mg else None
    first = ex or mg or sub
    return NeighbourVote(
        first_code=first,
        first_code_level=_first_code_level(first),
        first_code_share=LevelShares(
            subclass=shares["subclass"][sub],
            maingroup=shares["maingroup"][mg] if mg else None,
            exact=shares["exact"][ex] if ex else None,
        ),
        ranked=ranked,
        neighbours=voted,
        k=k,
        temperature=temperature,
        beta=beta,
        _shares=shares,
    )
