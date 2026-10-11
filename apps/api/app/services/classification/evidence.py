"""Extractive evidence: exact passages from the original input fields with verified offsets.

Offsets are Python str (Unicode code point) indices, start-inclusive, end-exclusive, relative to
the original field named by `source`. Every passage satisfies field_text[start:end] == quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.retrieval.keyword import BOILERPLATE, STOPWORDS

SENTENCE_RE = re.compile(r"[^.!?;\n]+(?:[.!?;]+|$)")
CLAUSE_SPLIT_RE = re.compile(r",\s+(?=(?:while|whereas|wherein|and wherein|such that)\b)|:\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
MAX_PASSAGE_CHARS = 420


@dataclass(frozen=True)
class Passage:
    source: str
    quote: str
    start: int
    end: int
    claim_number: int | None = None
    section: str | None = None
    is_background: bool = False


class EvidenceError(ValueError):
    pass


def verify(passage: Passage, fields: dict[str, str]) -> None:
    text = fields.get(passage.source)
    if text is None:
        raise EvidenceError(f"source field {passage.source!r} not supplied")
    if not (0 <= passage.start < passage.end <= len(text)) or text[passage.start : passage.end] != passage.quote:
        raise EvidenceError(f"quote does not match {passage.source}[{passage.start}:{passage.end}]")


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and (text[s].isspace() or text[s] in ",;"):
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def split_passages(text: str, base: int = 0, *, source: str, **meta) -> list[Passage]:
    """Sentences, with long sentences split further at clause boundaries; exact offsets."""
    out: list[Passage] = []
    for m in SENTENCE_RE.finditer(text):
        s, e = _trim(text, m.start(), m.end())
        if e - s < 12:
            continue
        spans = [(s, e)]
        if e - s > MAX_PASSAGE_CHARS:
            spans, cur = [], s
            for c in CLAUSE_SPLIT_RE.finditer(text, s, e):
                spans.append((cur, c.start()))
                cur = c.end()
            spans.append((cur, e))
        for a, b in spans:
            a, b = _trim(text, a, b)
            if b - a >= 12:
                out.append(Passage(source=source, quote=text[a:b], start=base + a, end=base + b, **meta))
    return out


def light_stem(word: str) -> str:
    w = word.lower()
    for suf in ("ations", "ation", "ings", "ing", "ions", "ion", "ers", "er", "ed", "es", "s"):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_stems(text: str) -> set[str]:
    return {
        light_stem(w)
        for w in WORD_RE.findall(text)
        if w.lower() not in STOPWORDS and w.lower() not in BOILERPLATE and len(w) > 2
    }


def _matches(stem: str, pool: set[str]) -> bool:
    if stem in pool:
        return True
    k = min(len(stem), 6)
    return any(p[:k] == stem[:k] for p in pool if len(p) >= k)


def defining_feature_check(candidate_title: str, input_text: str) -> tuple[list[str], list[str]]:
    """Heuristic: which content terms of the candidate title appear in the input (stem-prefix match).

    Returns (present_terms, missing_terms). This is a lexical heuristic, not proof of scope.
    """
    title = re.sub(r"\([^()]*\)", " ", candidate_title)
    pool = content_stems(input_text)
    present, missing = [], []
    seen: set[str] = set()
    for w in WORD_RE.findall(title):
        lw = w.lower()
        if lw in STOPWORDS or lw in BOILERPLATE or len(lw) <= 2 or lw in seen:
            continue
        if lw in {"e.g", "eg", "i.e", "ie", "etc", "specially", "adapted", "therefor", "details", "general",
                  "arrangements", "arrangement", "systems", "system", "methods", "method", "apparatus",
                  "devices", "device", "means", "other", "kinds"}:
            continue
        seen.add(lw)
        (present if _matches(light_stem(lw), pool) else missing).append(lw)
    return present, missing


# Generic words that appear in many subclass titles and carry no field-specific scope.
GENERIC_SCOPE_TERMS = frozenset(
    ["communication", "communications", "network", "networks", "system", "systems", "information", "data", "apparatus", "apparatuses", "method", "methods", "device", "devices", "general", "arrangement", "arrangements", "means", "element", "elements", "processing", "process", "processes", "equipment", "technique", "techniques", "electric", "electrical", "other", "kind", "kinds", "specially", "adapted", "therefor", "thereof", "details", "parts", "accessories"]
)


def subclass_scope_check(subclass_title: str, input_text: str) -> tuple[list[str], list[str]]:
    """Heuristic: do any *distinctive* subclass-title terms appear in the input?

    Returns (present, missing) over distinctive terms only. A candidate whose subclass scope has
    distinctive terms but none present (e.g. H04W 'WIRELESS ...' for wired/unspecified networks)
    is not supported as-is. Lexical heuristic; not proof of scope.
    """
    title = re.sub(r"\([^()]*\)", " ", subclass_title)
    title = re.split(r"\be\.g\.|\bi\.e\.", title, flags=re.I)[0]
    pool = content_stems(input_text)
    present, missing, seen = [], [], set()
    for w in WORD_RE.findall(title):
        lw = w.lower()
        if lw in seen or lw in STOPWORDS or lw in BOILERPLATE or lw in GENERIC_SCOPE_TERMS or len(lw) <= 2:
            continue
        seen.add(lw)
        (present if _matches(light_stem(lw), pool) else missing).append(lw)
    return present, missing
