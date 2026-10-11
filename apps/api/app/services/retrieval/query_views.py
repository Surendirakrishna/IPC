"""Source-labelled query views over the request documents.

- The abstract (+ title) view is always present and is the only view for an abstract-only request.
- Claims are segmented at reliable supplied claim boundaries ("1. ...", "2. ..."), preserving
  numbering, dependency references and original code-point offsets. If numbering cannot be parsed
  reliably, claims are kept as labelled chunks with claim_number=None.
- The specification is split at detected section headings, then into overlapping token-bounded
  chunks with original offsets. Background/prior-art sections are marked.
- A deterministic budget policy bounds the number of views; omissions are reported, never silent.

All offsets are Python str indices (Unicode code points), start-inclusive, end-exclusive,
relative to the original field.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

Source = Literal["abstract", "title", "claims", "specification"]

CLAIM_START_RE = re.compile(r"(?m)^[ \t]*(?:claim[ \t]+)?(\d{1,3})[ \t]*[.)][ \t]+(?=\S)", re.IGNORECASE)
CLAIM_DEP_RE = re.compile(r"\b(?:according to|as claimed in|of|in|as defined in|as recited in)\s+claims?\s+(\d{1,3})",
                          re.IGNORECASE)
HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:\[\d{4}\][ \t]*)?"
    r"(TECHNICAL FIELD|FIELD OF (?:THE )?INVENTION|BACKGROUND(?: OF (?:THE )?INVENTION| ART)?|PRIOR ART|"
    r"DESCRIPTION OF (?:THE )?RELATED ART|SUMMARY(?: OF (?:THE )?INVENTION)?|"
    r"BRIEF DESCRIPTION OF (?:THE )?DRAWINGS|DETAILED DESCRIPTION(?: OF [A-Z ]+)?|"
    r"DESCRIPTION OF (?:THE )?(?:PREFERRED )?EMBODIMENTS?|OBJECTS? OF (?:THE )?INVENTION)[ \t:]*$",
    re.IGNORECASE,
)
BACKGROUND_KEYS = ("BACKGROUND", "PRIOR ART", "RELATED ART")


@dataclass
class QueryView:
    view_id: str
    source: Source
    text: str
    start: int
    end: int
    claim_number: int | None = None
    claim_depends_on: list[int] = field(default_factory=list)
    is_independent_claim: bool | None = None
    section: str | None = None
    is_background: bool = False

    def label(self) -> str:
        if self.source == "claims" and self.claim_number is not None:
            return f"claims[{self.claim_number}]"
        return self.view_id


@dataclass
class ViewPlan:
    views: list[QueryView]
    coverage: dict[str, str]
    counts: dict[str, dict[str, int]]
    warnings: list[str]
    claims_parse: str | None = None


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def segment_claims(claims: str) -> tuple[list[QueryView], str]:
    """Return per-claim views and a parse status: 'numbered' or 'unnumbered'."""
    matches = list(CLAIM_START_RE.finditer(claims))
    numbers = [int(m.group(1)) for m in matches]
    reliable = bool(matches) and numbers[0] == 1 and all(b == a + 1 for a, b in zip(numbers, numbers[1:], strict=False))
    if not reliable:
        return [], "unnumbered"
    views: list[QueryView] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(claims)
        s, e = _strip_span(claims, start, end)
        body = claims[s:e]
        deps = sorted({int(d) for d in CLAIM_DEP_RE.findall(body) if int(d) < numbers[i]})
        views.append(
            QueryView(
                view_id=f"claims[{numbers[i]}]",
                source="claims",
                text=body,
                start=s,
                end=e,
                claim_number=numbers[i],
                claim_depends_on=deps,
                is_independent_claim=not deps,
            )
        )
    return views, "numbered"


def specification_sections(spec: str) -> list[tuple[str | None, int, int]]:
    heads = list(HEADING_RE.finditer(spec))
    if not heads:
        return [(None, 0, len(spec))]
    sections: list[tuple[str | None, int, int]] = []
    if heads[0].start() > 0:
        sections.append((None, 0, heads[0].start()))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(spec)
        sections.append((h.group(1).upper(), h.end(), end))
    return sections


TokenOffsets = Callable[[str], list[tuple[int, int]]]


def regex_token_offsets(text: str) -> list[tuple[int, int]]:
    """Fallback tokenizer: word/punctuation pieces with code-point offsets (conservative estimate)."""
    return [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def chunk_span(
    text: str, start: int, end: int, *, max_tokens: int, overlap: int, token_offsets: TokenOffsets
) -> list[tuple[int, int]]:
    """Split text[start:end] into overlapping chunks of at most max_tokens tokens.

    Boundaries fall on token offsets (code-point indices), so no code point is split.
    """
    segment = text[start:end]
    offs = token_offsets(segment)
    if not offs:
        return []
    budget = max(8, max_tokens - 2)  # room for special tokens
    step = max(1, budget - overlap)
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < len(offs):
        j = min(len(offs), i + budget)
        cs, ce = start + offs[i][0], start + offs[j - 1][1]
        chunks.append((cs, ce))
        if j == len(offs):
            break
        i += step
    return chunks


def build_views(
    abstract: str,
    title: str | None,
    claims: str | None,
    specification: str | None,
    *,
    max_views: int,
    chunk_tokens: int,
    chunk_overlap: int,
    token_offsets: TokenOffsets = regex_token_offsets,
) -> ViewPlan:
    warnings: list[str] = []
    abstract_text = f"{title}\n{abstract}" if title else abstract
    abstract_view = QueryView("abstract", "abstract", abstract_text, 0, len(abstract))
    views: list[QueryView] = [abstract_view]
    coverage = {"abstract": "complete", "specification": "not_supplied", "claims": "not_supplied"}
    counts: dict[str, dict[str, int]] = {}
    claims_parse = None

    claim_views: list[QueryView] = []
    if claims:
        claim_views, claims_parse = segment_claims(claims)
        if claims_parse == "unnumbered":
            warnings.append("Claim numbering could not be parsed reliably; claims kept as unnumbered chunks.")
            claim_views = [
                QueryView(f"claims#{k}", "claims", claims[s:e], s, e)
                for k, (s, e) in enumerate(
                    chunk_span(claims, 0, len(claims), max_tokens=chunk_tokens, overlap=chunk_overlap,
                               token_offsets=token_offsets)
                )
            ]
        else:
            # Split any over-long single claim into labelled sub-chunks keeping its claim number.
            expanded: list[QueryView] = []
            for v in claim_views:
                pieces = chunk_span(claims, v.start, v.end, max_tokens=chunk_tokens, overlap=chunk_overlap,
                                    token_offsets=token_offsets)
                if len(pieces) <= 1:
                    expanded.append(v)
                    continue
                for k, (s, e) in enumerate(pieces):
                    expanded.append(QueryView(f"{v.view_id}#{k}", "claims", claims[s:e], s, e, v.claim_number,
                                              v.claim_depends_on, v.is_independent_claim))
            claim_views = expanded

    spec_views: list[QueryView] = []
    if specification:
        for sec_name, s0, e0 in specification_sections(specification):
            is_bg = bool(sec_name and any(k in sec_name for k in BACKGROUND_KEYS))
            for k, (s, e) in enumerate(
                chunk_span(specification, s0, e0, max_tokens=chunk_tokens, overlap=chunk_overlap,
                           token_offsets=token_offsets)
            ):
                spec_views.append(
                    QueryView(f"specification:{sec_name or 'body'}#{k}", "specification",
                              specification[s:e], s, e, section=sec_name, is_background=is_bg)
                )

    # Deterministic budget: abstract always; then independent claims, dependent claims in order;
    # then non-background spec chunks in document order; background chunks last.
    remaining = max_views - 1
    claim_order = sorted(claim_views, key=lambda v: (v.is_independent_claim is False, v.start))
    spec_order = sorted(spec_views, key=lambda v: (v.is_background, v.start))
    claim_quota = min(len(claim_order), remaining // 2 if spec_order else remaining)
    spec_quota = min(len(spec_order), remaining - claim_quota)
    claim_quota = min(len(claim_order), remaining - spec_quota)
    chosen_claims = sorted(claim_order[:claim_quota], key=lambda v: v.start)
    chosen_spec = sorted(spec_order[:spec_quota], key=lambda v: v.start)
    views += chosen_claims + chosen_spec

    for name, all_v, chosen in (("claims", claim_views, chosen_claims), ("specification", spec_views, chosen_spec)):
        text = claims if name == "claims" else specification
        if not text:
            continue
        counts[name] = {"chunks_total": len(all_v), "chunks_reviewed": len(chosen)}
        if not all_v:
            coverage[name] = "empty"
        elif len(chosen) == len(all_v):
            coverage[name] = "complete"
        else:
            coverage[name] = "partial"
            warnings.append(
                f"{name}: {len(chosen)} of {len(all_v)} chunks reviewed within the query-view budget; "
                "omitted text was not used as evidence."
            )
    if any(v.is_background for v in chosen_spec):
        warnings.append("Background/prior-art specification sections are marked and not treated as claimed features.")
    return ViewPlan(views=views, coverage=coverage, counts=counts, warnings=warnings, claims_parse=claims_parse)
