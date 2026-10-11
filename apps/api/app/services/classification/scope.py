"""Deterministic candidate scope documents for reranking and comparison.

Order of content (most important first, so any truncation drops optional context last):
candidate code/title -> original wording (if it differs) -> parent path -> duplicate context
-> reference target titles. The version string is part of the confidence compatibility key.
"""

from __future__ import annotations

from app.db.repository import Record
from app.services.ingestion.wipo import display_symbol

SCOPE_FORMATTER_VERSION = "scope-v3"
RULE_LABELS = {
    "precedence": "precedence rule",
    "place_last": "last-place priority rule",
    "place_first": "first-place priority rule",
    "limiting_reference": "limiting reference",
    "informative_reference": "informative reference",
    "indexing": "indexing note",
    "definition": "definition",
    "scope": "scope statement",
}


def parent_path(rec: Record, records: dict[str, Record]) -> list[dict[str, str]]:
    """Return the complete WIPO ancestor chain, including nested subgroup parents."""
    path = []
    for code in rec.ancestor_codes:
        ancestor = records.get(code)
        if not ancestor:
            continue
        path.append({"code": code, "display_code": ancestor.display_code, "title": ancestor.title,
                     "level": ancestor.record_type})
    if path:
        return path
    # Compatibility for an unmigrated record while an old release is being inspected.
    path = [{"code": rec.section_code, "display_code": rec.section_code,
             "title": rec.section_title, "level": "section"}]
    if rec.class_code:
        path.append({"code": rec.class_code, "display_code": rec.class_code,
                     "title": rec.class_title or "", "level": "class"})
    if rec.subclass_code and rec.record_type not in ("section", "class", "subclass"):
        path.append({"code": rec.subclass_code, "display_code": rec.subclass_code,
                     "title": rec.subclass_title or "", "level": "subclass"})
    if rec.maingroup_code and rec.record_type == "subgroup":
        mg = records.get(rec.maingroup_code)
        path.append({"code": rec.maingroup_code, "display_code": mg.display_code if mg else rec.maingroup_code,
                     "title": rec.maingroup_title or "", "level": "maingroup"})
    return path


def _official_context_lines(rec: Record, max_chars: int = 6_000) -> list[str]:
    labels = {
        "DEFINITION-STATEMENT": "WIPO definition",
        "SPECIALRULES": "WIPO special rule",
        "LIMITINGREFERENCES": "WIPO limiting reference",
        "APPLICATIONORIENTEDREFERENCES": "WIPO application-oriented reference",
        "INFORMATIVEREFERENCES": "WIPO informative reference",
        "REFERENCESOUTOFARESIDUALPLACE": "WIPO residual-place reference",
        "GLOSSARYOFTERMS": "WIPO glossary",
        "SYNONYMSANDKEYWORDS": "WIPO synonyms and keywords",
        "SYNONYMS": "WIPO synonyms",
        "LARGESUBJECTS": "WIPO large-subject note",
        "ABBREVIATIONS": "WIPO abbreviations",
        "SPECIALMEANINGS": "WIPO special meanings",
        "INSTEADOFWORDS": "WIPO terminology",
        "note": "WIPO scheme note",
        "entryReference": "WIPO scheme reference",
        "references": "WIPO scheme reference",
    }
    owner_order = {rec.code: 0, **{code: i + 1 for i, code in enumerate(reversed(rec.ancestor_codes))}}
    priority = {"SPECIALRULES": 0, "note": 1, "DEFINITION-STATEMENT": 2,
                "LIMITINGREFERENCES": 3, "APPLICATIONORIENTEDREFERENCES": 4,
                "INFORMATIVEREFERENCES": 5}
    rule_priority = {"precedence": 0, "place_last": 0, "place_first": 0}
    context = sorted(
        rec.official_context,
        key=lambda item: (owner_order.get(item.get("applies_to") or item["owner_symbol"], 999),
                          rule_priority.get(item.get("rule_kind") or "", 1),
                          priority.get(item["part_type"], 10), item["ordinal"]),
    )
    lines: list[str] = []
    used = 0
    for item in context:
        text = " ".join(str(item.get("text") or "").split())
        if not text:
            continue
        part_type = item["part_type"]
        label = labels.get(part_type, f"WIPO {part_type.replace('_', ' ').lower()}")
        details = []
        if item["source_kind"] == "scheme_note":
            details.append(f"type={item.get('note_type') or 'none'}")
        if item.get("rule_kind") and item["rule_kind"] != "other":
            details.append(RULE_LABELS.get(item["rule_kind"], item["rule_kind"]))
        if item.get("editions"):
            details.append(f"edition {item['editions']}")
        if details:
            label += f" ({'; '.join(details)})"
        if item.get("rule_kind") in ("limiting_reference", "informative_reference") or item["source_kind"] == "scheme_reference":
            label += "; reference context does not itself assign a target code"
        owner = display_symbol(item["owner_symbol"])
        if item.get("scope_end_symbol"):
            owner += f"-{display_symbol(item['scope_end_symbol'])}"
        line = f"{label} for {owner}: {text}"
        symbols = item.get("related_symbols") or []
        if symbols:
            line += "; symbols cited under the stated conditions: " + ", ".join(symbols[:16])
            if len(symbols) > 16:
                line += ", …"
        remain = max_chars - used
        if remain <= 0:
            break
        if len(line) > remain:
            line = line[:remain].rstrip() + "…"
        lines.append(line)
        used += len(line)
    return lines


def format_scope(rec: Record, records: dict[str, Record], *, max_refs: int = 6, official: bool = True) -> str:
    lines = [f"{rec.display_code}: {rec.title}"]
    if rec.original_title and rec.original_title.strip() != rec.title.strip():
        lines.append(f"Original wording: {rec.original_title.strip()}")
    for p in reversed(parent_path(rec, records)):
        lines.append(f"Within {p['level']} {p['display_code']}: {p['title']}")
    if rec.context_heading:
        lines.append(f"Context heading: {rec.context_heading}")
    for ref in rec.references[:max_refs]:
        if ref.get("target_title"):
            lines.append(
                f"Reference context ({ref['kind'].replace('_', ' ')}; cited code is not an assignment instruction): "
                f"{ref.get('target_display_code') or ref['target_code']} {ref['target_title']}"
            )
    if official:
        lines.extend(_official_context_lines(rec))
    return "\n".join(lines)
