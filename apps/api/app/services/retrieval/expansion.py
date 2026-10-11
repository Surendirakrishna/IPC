"""Expand strongest candidates with parents and bounded siblings.

- Parent main groups of top subgroups join the comparison set (a broader candidate must be possible).
- Siblings: for each top subgroup, at most N siblings under the same main group, chosen by exact
  cosine similarity to the abstract view (deterministic, bounded).
- Cross-references remain scope context; a reference link alone never creates an assignment candidate.
Each added candidate records its expansion origin; its missing first-stage scores stay None.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.db import repository as repo
from app.services.retrieval.fusion import FusedCandidate
from app.services.retrieval.vector import similarities_for_codes

GROUP_TYPES = {"maingroup", "subgroup"}


def expand(
    session: Session,
    *,
    version: str,
    table: str,
    fused: list[FusedCandidate],
    abstract_vector: np.ndarray | None,
    top_n: int,
    max_siblings: int,
    pool_cap: int,
    allowed_types: set[str] = GROUP_TYPES,
    lexical_reserve: int = 15,
) -> list[FusedCandidate]:
    pool: dict[str, FusedCandidate] = {c.code: c for c in fused}
    top = fused[:top_n]
    records = repo.get_records(session, version, [c.code for c in top])

    added: list[FusedCandidate] = []

    def add(code: str, origin: str) -> None:
        if code in pool:
            pool[code].origin.add(origin)
            return
        fc = FusedCandidate(code=code, rrf_score=0.0, origin={origin})
        pool[code] = fc
        added.append(fc)

    # parents
    for c in top:
        rec = records.get(c.code)
        if rec and rec.record_type == "subgroup" and rec.maingroup_code and "maingroup" in allowed_types:
            add(rec.maingroup_code, f"expansion:parent_of:{c.code}")

    # siblings, bounded and ranked by exact similarity
    if abstract_vector is not None and max_siblings > 0:
        parents = sorted({r.parent_code for r in records.values()
                          if r.record_type == "subgroup" and r.parent_code})
        sibs = repo.get_siblings(session, version, parents)
        sib_codes = sorted({s for lst in sibs.values() for s in lst} - set(pool))
        sims = similarities_for_codes(session, table=table, version=version, query_vector=abstract_vector,
                                      codes=sib_codes) if sib_codes else {}
        for c in top:
            rec = records.get(c.code)
            if not rec or rec.record_type != "subgroup" or not rec.parent_code:
                continue
            cands = [s for s in sibs.get(rec.parent_code, []) if s not in pool and s in sims]
            cands.sort(key=lambda s: (-sims[s], s))
            for s in cands[:max_siblings]:
                add(s, f"expansion:sibling_of:{c.code}")

    # Bounded pool: fused head, plus strong lexical-only candidates (exact technical terms must be
    # able to influence the result), plus expansions; trimmed to pool_cap.
    added = added[: max(0, pool_cap // 3)]
    base_n = max(0, pool_cap - len(added))
    head = list(fused[:base_n])
    head_codes = {c.code for c in head}
    lexical = [c for c in fused[base_n:] if c.keyword_rank is not None and c.keyword_rank <= lexical_reserve
               and c.vector_rank is None and c.code not in head_codes]
    if lexical:
        head = head[: max(0, base_n - len(lexical))] + lexical
    return (head + added)[:pool_cap]
