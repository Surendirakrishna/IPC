"""Read-side queries for the taxonomy. All parameters are bound; no SQL string interpolation of input."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

RECORD_FIELDS = (
    "code, display_code, record_type, COALESCE(official_title, title) AS title, title_raw, semantic_text, "
    "section_code, section_title,"
    " class_code, class_title, subclass_code, subclass_title, maingroup_code, maingroup_title,"
    " parent_code, ancestor_codes, child_codes, depth, context_heading, titles, canonical_title,"
    " is_duplicate_symbol, original_title, original_title_source, official_title"
)


@dataclass
class Record:
    code: str
    display_code: str
    record_type: str
    title: str
    title_raw: str | None
    semantic_text: str
    section_code: str
    section_title: str
    class_code: str | None
    class_title: str | None
    subclass_code: str | None
    subclass_title: str | None
    maingroup_code: str | None
    maingroup_title: str | None
    parent_code: str | None
    ancestor_codes: list[str]
    child_codes: list[str]
    depth: int
    context_heading: str | None
    titles: list[str] | None
    canonical_title: str | None
    is_duplicate_symbol: bool | None
    original_title: str | None
    original_title_source: dict[str, Any] | None
    official_title: str | None
    references: list[dict[str, Any]] = field(default_factory=list)
    official_context: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Release:
    taxonomy_version: str
    source_hashes: dict[str, str]
    source_root: dict[str, Any]
    validation_report: dict[str, Any] | None
    import_state: str
    is_active: bool


def active_release(session: Session) -> Release | None:
    row = session.execute(
        text(
            "SELECT taxonomy_version, source_hashes || official_source_hashes, source_root, validation_report, "
            "import_state, is_active"
            " FROM taxonomy_releases WHERE is_active"
        )
    ).first()
    return Release(*row) if row else None


def get_release(session: Session, version: str) -> Release | None:
    row = session.execute(
        text(
            "SELECT taxonomy_version, source_hashes || official_source_hashes, source_root, validation_report, "
            "import_state, is_active"
            " FROM taxonomy_releases WHERE taxonomy_version = :v"
        ),
        {"v": version},
    ).first()
    return Release(*row) if row else None


def get_records(session: Session, version: str, codes: list[str]) -> dict[str, Record]:
    if not codes:
        return {}
    rows = session.execute(
        text(f"SELECT {RECORD_FIELDS} FROM ipc_records WHERE taxonomy_version = :v AND code = ANY(:codes)"),
        {"v": version, "codes": list(dict.fromkeys(codes))},
    ).all()
    return {r[0]: Record(*r) for r in rows}


def get_record_by_any_code(session: Session, version: str, code: str) -> Record | None:
    row = session.execute(
        text(
            f"SELECT {RECORD_FIELDS} FROM ipc_records WHERE taxonomy_version = :v"
            " AND (code = :c OR display_code = :c) ORDER BY (code = :c) DESC LIMIT 1"
        ),
        {"v": version, "c": code.strip()},
    ).first()
    return Record(*row) if row else None


def get_references(session: Session, version: str, codes: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not codes:
        return {}
    rows = session.execute(
        text(
            "SELECT e.source_code, e.target_code, e.kind, e.original_wording, e.target_exists,"
            " t.display_code, t.title, t.record_type"
            " FROM ipc_reference_edges e LEFT JOIN ipc_records t"
            "   ON t.taxonomy_version = e.taxonomy_version AND t.code = e.target_code"
            " WHERE e.taxonomy_version = :v AND e.source_code = ANY(:codes)"
            " ORDER BY e.source_code, e.kind, e.target_code"
        ),
        {"v": version, "codes": codes},
    ).all()
    out: dict[str, list[dict[str, Any]]] = {}
    for src, tgt, kind, wording, exists, disp, title, rtype in rows:
        out.setdefault(src, []).append(
            {"target_code": tgt, "target_display_code": disp, "target_title": title,
             "target_record_type": rtype, "kind": kind, "original_wording": wording,
             "target_exists": exists}
        )  # fmt: skip
    return out


def get_official_texts(session: Session, version: str, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
    """WIPO notes, references and definitions that apply to each requested symbol.

    A text applies to its owner symbol and to every entry in its scope (a note's symbol..endSymbol
    range), so a code in the middle of a range note's range is keyed to that note too. `applies_to`
    records which requested symbol the text was found through.
    """
    if not symbols:
        return {}
    wanted = list(dict.fromkeys(symbols))
    rows = session.execute(
        text(
            "SELECT owner_symbol, source_kind, part_type, ordinal, note_type, content_text, related_symbols,"
            " container_symbol, scope_end_symbol, scope_symbols, editions, rule_kind, rules"
            " FROM ipc_official_texts WHERE taxonomy_version = :v"
            " AND (owner_symbol = ANY(:symbols) OR scope_symbols && CAST(:symbols AS text[]))"
            " ORDER BY owner_symbol, source_kind, part_type, ordinal"
        ),
        {"v": version, "symbols": wanted},
    ).all()
    wanted_set = set(wanted)
    out: dict[str, list[dict[str, Any]]] = {}
    for (owner, source_kind, part_type, ordinal, note_type, content, related, container, scope_end, scope,
         editions, rule_kind, rules) in rows:
        item = {
            "owner_symbol": owner,
            "source_kind": source_kind,
            "part_type": part_type,
            "ordinal": ordinal,
            "note_type": note_type,
            "text": content,
            "related_symbols": related,
            "container_symbol": container,
            "scope_end_symbol": scope_end,
            "scope_symbols": scope,
            "editions": editions,
            "rule_kind": rule_kind,
            "rules": rules,
        }
        for symbol in dict.fromkeys([owner, *(scope or [])]):
            if symbol in wanted_set:
                out.setdefault(symbol, []).append({**item, "applies_to": symbol})
    return out


def get_incoming_references(session: Session, version: str, code: str, limit: int = 50) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            "SELECT e.source_code, s.display_code, s.title, e.kind, e.original_wording"
            " FROM ipc_reference_edges e JOIN ipc_records s"
            "   ON s.taxonomy_version = e.taxonomy_version AND s.code = e.source_code"
            " WHERE e.taxonomy_version = :v AND e.target_code = :c ORDER BY e.source_code LIMIT :lim"
        ),
        {"v": version, "c": code, "lim": limit},
    ).all()
    return [
        {"source_code": r[0], "source_display_code": r[1], "source_title": r[2], "kind": r[3],
         "original_wording": r[4]}
        for r in rows
    ]  # fmt: skip


def get_occurrences(session: Session, version: str, code: str) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            "SELECT occurrence_id, source_path, original_title, direct_child_symbols, parent_symbol,"
            " sibling_index, node_type FROM ipc_source_occurrences"
            " WHERE taxonomy_version = :v AND code = :c ORDER BY occurrence_id"
        ),
        {"v": version, "c": code},
    ).all()
    return [
        {"occurrence_id": r[0], "source_path": r[1], "original_title": r[2], "direct_child_symbols": r[3],
         "parent_symbol": r[4], "sibling_index": r[5], "node_type": r[6]}
        for r in rows
    ]  # fmt: skip


def get_siblings(session: Session, version: str, parent_codes: list[str]) -> dict[str, list[str]]:
    if not parent_codes:
        return {}
    rows = session.execute(
        text(
            "SELECT r.parent_code, r.code FROM ipc_records r"
            " JOIN ipc_official_scheme_entries o ON o.taxonomy_version = r.taxonomy_version AND o.code = r.code"
            " WHERE r.taxonomy_version = :v AND o.entry_type = 'K'"
            " AND r.parent_code = ANY(:p) ORDER BY r.parent_code, r.code"
        ),
        {"v": version, "p": parent_codes},
    ).all()
    out: dict[str, list[str]] = {}
    for parent, code in rows:
        out.setdefault(parent, []).append(code)
    return out


def search_records(
    session: Session,
    version: str,
    *,
    q: str | None,
    record_type: str | None,
    section: str | None,
    subclass: str | None,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    """Exact/prefix code lookup is separate from full-text ranking."""
    params = {"v": version, "rt": record_type, "sec": section, "sub": subclass, "lim": limit, "off": offset}
    filters = (
        " AND (CAST(:rt AS text) IS NULL OR record_type = :rt)"
        " AND (CAST(:sec AS text) IS NULL OR section_code = :sec)"
        " AND (CAST(:sub AS text) IS NULL OR subclass_code = :sub)"
    )
    q = (q or "").strip()
    code_like = bool(q) and len(q) <= 20 and q[:1].isalpha() and q[:1].isupper() and any(ch.isdigit() for ch in q)
    if not q:
        where, order, mode = "", "code", "browse"
    elif code_like:
        params["q"] = q
        params["qp"] = q.replace("%", "").replace("_", "") + "%"
        where = " AND (code = :q OR display_code = :q OR display_code LIKE :qp OR code LIKE :qp)"
        order = "(code = :q OR display_code = :q) DESC, code"
        mode = "code"
    else:
        params["q"] = q
        where = " AND search_vector @@ websearch_to_tsquery('english', :q)"
        order = "ts_rank_cd(search_vector, websearch_to_tsquery('english', :q), 1) DESC, code"
        mode = "fulltext"
    total = session.execute(
        text(f"SELECT count(*) FROM ipc_records WHERE taxonomy_version = :v{filters}{where}"), params
    ).scalar_one()
    rows = session.execute(
        text(
            "SELECT code, display_code, record_type, title, parent_code, subclass_code FROM ipc_records"
            f" WHERE taxonomy_version = :v{filters}{where} ORDER BY {order} LIMIT :lim OFFSET :off"
        ),
        params,
    ).all()
    items = [
        {"code": r[0], "display_code": r[1], "record_type": r[2], "title": r[3], "parent_code": r[4],
         "subclass_code": r[5], "match_mode": mode}
        for r in rows
    ]  # fmt: skip
    return items, int(total)
