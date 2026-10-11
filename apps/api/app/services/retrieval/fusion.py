"""Reciprocal rank fusion, within a view and across source-labelled views.

RRF(candidate) = sum over lists of 1 / (k + rank), ranks starting at 1; absent => 0.
Component ranks/scores are preserved. Cosine similarity and ts_rank_cd are never added together.

Two-level fusion for optional documents:
  1. per view: fuse that view's keyword and vector lists;
  2. per source (abstract / claims / specification): fuse that source's view lists, then cap;
  3. across sources: fuse the capped source lists with equal per-source contribution.
With only the abstract view, the result equals the per-view fusion (abstract-only path unchanged).

Optional weights scale a list's RRF contribution (w / (k + rank)): a view's weight applies inside its
source, and a source weight applies across sources. All weights default to 1.0 (plain RRF).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

SOURCE_ORDER = ("abstract", "claims", "specification")


@dataclass
class ViewHits:
    view_id: str
    source: str
    keyword: list[tuple[str, int, float]]  # (code, rank, ts_rank_cd)
    vector: list[tuple[str, int, float]]  # (code, rank, cosine_similarity)
    weight: float = 1.0


@dataclass
class FusedCandidate:
    code: str
    rrf_score: float
    rank: int = 0
    keyword_rank: int | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    cosine_similarity: float | None = None
    sources: dict[str, float] = field(default_factory=dict)  # source -> source-level RRF
    views: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    origin: set[str] = field(default_factory=set)


def rrf(lists: list[list[str]], k: int, weights: list[float] | None = None) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for i, lst in enumerate(lists):
        w = 1.0 if weights is None else weights[i]
        for rank, code in enumerate(lst, start=1):
            scores[code] += w / (k + rank)
    return dict(scores)


def _ordered(scores: dict[str, float]) -> list[str]:
    return [c for c, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def fuse(view_hits: list[ViewHits], *, k: int, per_source_cap: int,
         source_weights: dict[str, float] | None = None) -> list[FusedCandidate]:
    cands: dict[str, FusedCandidate] = {}
    by_source: dict[str, list[list[str]]] = defaultdict(list)
    by_source_weights: dict[str, list[float]] = defaultdict(list)

    for vh in view_hits:
        kw_codes = [c for c, _, _ in vh.keyword]
        vec_codes = [c for c, _, _ in vh.vector]
        view_scores = rrf([kw_codes, vec_codes], k)
        by_source[vh.source].append(_ordered(view_scores))
        by_source_weights[vh.source].append(vh.weight)
        kw = {c: (r, s) for c, r, s in vh.keyword}
        vec = {c: (r, s) for c, r, s in vh.vector}
        for code, score in view_scores.items():
            fc = cands.setdefault(code, FusedCandidate(code=code, rrf_score=0.0))
            entry = {
                "rrf": score,
                "keyword_rank": kw[code][0] if code in kw else None,
                "keyword_score": kw[code][1] if code in kw else None,
                "vector_rank": vec[code][0] if code in vec else None,
                "cosine_similarity": vec[code][1] if code in vec else None,
            }
            fc.views[vh.view_id] = entry
            if code in kw:
                fc.origin.add(f"keyword:{vh.source}")
                if fc.keyword_rank is None or kw[code][0] < fc.keyword_rank:
                    fc.keyword_rank, fc.keyword_score = kw[code]
            if code in vec:
                fc.origin.add(f"vector:{vh.source}")
                if fc.vector_rank is None or vec[code][0] < fc.vector_rank:
                    fc.vector_rank, fc.cosine_similarity = vec[code]

    source_lists: list[list[str]] = []
    list_weights: list[float] = []
    for source in SOURCE_ORDER:
        lists = by_source.get(source)
        if not lists:
            continue
        source_scores = rrf(lists, k, by_source_weights[source])
        capped = _ordered(source_scores)[:per_source_cap]
        source_lists.append(capped)
        list_weights.append((source_weights or {}).get(source, 1.0))
        for code in capped:
            cands[code].sources[source] = source_scores[code]

    if len(source_lists) == 1 and len(view_hits) == 1:
        # Abstract-only: final score is exactly the per-view fusion.
        only = view_hits[0].view_id
        final = {c: cands[c].views[only]["rrf"] for c in source_lists[0]}  # type: ignore[misc]
    else:
        final = rrf(source_lists, k, list_weights)

    out = []
    for rank, code in enumerate(_ordered(final), start=1):
        fc = cands[code]
        fc.rrf_score = float(final[code])
        fc.rank = rank
        out.append(fc)
    return out


def vote_subclasses(fused: list[FusedCandidate], limit: int) -> list[str]:
    """Rank subclasses by the summed fused RRF score of their retrieved groups (IPC symbol prefix)."""
    votes: dict[str, float] = defaultdict(float)
    for c in fused:
        votes[c.code[:4]] += c.rrf_score
    return _ordered(votes)[:limit]
