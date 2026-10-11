"""PostgreSQL native full-text retrieval (ts_rank_cd over a weighted tsvector). Not BM25.

Terms are extracted deterministically from each query view; the query is an OR of phrase
alternatives passed as a bound parameter to websearch_to_tsquery — never an AND of every word
and never string-concatenated SQL.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

STOPWORDS = frozenset(
    ["a", "about", "above", "after", "again", "against", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because", "been", "being", "below", "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing", "during", "each", "either", "few", "for", "from", "further", "had", "has", "have", "having", "here", "how", "if", "in", "into", "is", "it", "its", "itself", "may", "might", "more", "most", "must", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "same", "shall", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "upon", "very", "via", "was", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "within", "without", "would", "wherein", "whereby", "thereof", "therein", "thereby", "said"]
)
# Patent boilerplate and generic framing words that carry no classification signal on their own.
BOILERPLATE = frozenset(
    ["invention", "present", "disclosed", "discloses", "disclose", "discloses", "embodiment", "embodiments", "comprising", "comprises", "comprise", "include", "includes", "including", "configured", "provide", "provides", "provided", "providing", "accordingly", "plurality", "various", "further", "thereof", "according", "wherein", "whereby", "use", "used", "using", "based", "one", "two", "first", "second", "least", "portion", "part", "parts", "manner", "way", "means", "herein", "example", "examples", "like", "particularly", "particular", "certain", "respective", "respectively", "proposed", "improving", "improved", "improve", "integrated", "framework"]
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]")
SAFE_TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\- ]*$")


@dataclass
class Term:
    text: str
    weight: float
    first_pos: int


def _content_tokens(text: str) -> list[tuple[str, int, bool]]:
    """(lowercased token, position, is_content) with sentence/clause breaks marked as None tokens."""
    out: list[tuple[str, int, bool]] = []
    for piece in re.split(r"[.;:,()\[\]\n]+", text):
        for m in WORD_RE.finditer(piece):
            tok = m.group(0).lower()
            out.append((tok, len(out), tok not in STOPWORDS and tok not in BOILERPLATE and len(tok) > 1))
        out.append(("", len(out), False))
    return out


def extract_terms(text: str, max_terms: int) -> list[Term]:
    """Deterministic technical-term extraction: content-word n-grams (n<=3) within clauses.

    Score = frequency * length bonus; ties broken by first position then text.
    """
    toks = _content_tokens(text)
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    run: list[tuple[str, int]] = []

    def flush() -> None:
        for n in (3, 2, 1):
            for i in range(len(run) - n + 1):
                gram = " ".join(t for t, _ in run[i : i + n])
                counts[gram] += 1
                first.setdefault(gram, run[i][1])

    for tok, pos, is_content in toks:
        if is_content:
            run.append((tok, pos))
        else:
            flush()
            run = []
    flush()

    scored = []
    for gram, c in counts.items():
        n = gram.count(" ") + 1
        if n == 1 and len(gram) < 3:
            continue
        scored.append(Term(gram, c * (1.0 + 0.6 * (n - 1)), first[gram]))
    scored.sort(key=lambda t: (-t.weight, t.first_pos, t.text))
    # Phrases first: single generic words (e.g. "key") match unrelated fields (lock keys, on/off keying),
    # so unigrams are capped and ranked after multi-word technical phrases of equal weight.
    max_unigrams = max(2, max_terms // 6)
    scored.sort(key=lambda t: (-(t.weight * (1.0 if " " in t.text else 0.5)), t.first_pos, t.text))
    chosen: list[Term] = []
    unigrams = 0
    for t in scored:
        is_unigram = " " not in t.text
        if is_unigram and (unigrams >= max_unigrams
                           or any(t.text in c.text.split() for c in chosen if " " in c.text)):
            continue
        chosen.append(t)
        unigrams += is_unigram
        if len(chosen) >= max_terms:
            break
    return chosen


def build_websearch_query(terms: list[Term]) -> str:
    parts = []
    for t in terms:
        if not SAFE_TERM_RE.match(t.text):
            continue
        parts.append(f'"{t.text}"' if " " in t.text else t.text)
    return " or ".join(parts)


@dataclass
class KeywordHit:
    code: str
    rank: int
    score: float


GROUP_TYPES = ("maingroup", "subgroup")


def keyword_search(
    session: Session,
    *,
    version: str,
    query: str,
    limit: int,
    record_types: tuple[str, ...] = GROUP_TYPES,
    section: str | None = None,
    subclass: str | None = None,
    subclasses: list[str] | None = None,
) -> list[KeywordHit]:
    if not query.strip():
        return []
    rows = session.execute(
        text(
            """
            WITH q AS (SELECT websearch_to_tsquery('english', :query) AS tsq)
            SELECT r.code, ts_rank_cd(r.search_vector, q.tsq, 1) AS score
            FROM ipc_records r
            JOIN ipc_official_scheme_entries official
              ON official.taxonomy_version = r.taxonomy_version AND official.code = r.code
            CROSS JOIN q
            WHERE r.taxonomy_version = :version
              AND official.entry_type = 'K'
              AND r.search_vector @@ q.tsq
              AND r.record_type = ANY(:types)
              AND (CAST(:section AS text) IS NULL OR r.section_code = :section)
              AND (CAST(:subclass AS text) IS NULL OR r.subclass_code = :subclass)
              AND (CAST(:subclasses AS text[]) IS NULL OR r.subclass_code = ANY(:subclasses))
            ORDER BY score DESC, r.code
            LIMIT :limit
            """
        ),
        {"query": query, "version": version, "types": list(record_types), "section": section,
         "subclass": subclass, "subclasses": subclasses, "limit": limit},
    ).all()
    return [KeywordHit(code=r[0], rank=i + 1, score=float(r[1])) for i, r in enumerate(rows)]
