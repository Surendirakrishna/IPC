"""Import a validated source snapshot into PostgreSQL.

Properties:
- Streams and validates the supplied files; never modifies them.
- Refuses to import into an existing release whose source hashes differ.
- Deterministic order (depth, source line) so parents precede children across batches.
- Bounded COPY batches into a temporary staging table, upserted by (release, code);
  each batch commits with a checkpoint in ingestion_jobs so an interrupted run resumes.
- Re-running an unchanged import leaves canonical counts unchanged.
- Activation is a separate step that requires import_state = 'validated'.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from app.config import Settings
from app.services.ingestion.sources import (
    IndexLine,
    read_duplicates,
    read_index,
    read_tree,
    sha256_file,
    sha256_text,
)
from app.services.ingestion.tree import (
    Occurrence,
    reference_wording,
    select_canonical_occurrence,
    traverse,
)
from app.services.ingestion.validate import ValidationReport, validate_sources
from app.services.ingestion.wipo import (
    OfficialCatchword,
    OfficialEntry,
    OfficialText,
    parse_catchwords,
    parse_definitions,
    parse_scheme,
    semantic_text,
)

log = logging.getLogger(__name__)

RECORD_COLUMNS = [
    "taxonomy_version", "code", "display_code", "record_type", "title", "title_raw", "semantic_text",
    "section_code", "section_title", "class_code", "class_title", "subclass_code", "subclass_title",
    "maingroup_code", "maingroup_title", "parent_code", "ancestor_codes", "child_codes", "depth",
    "aliases", "heading_text", "context_heading", "titles", "canonical_title", "is_duplicate_symbol",
    "original_title", "original_title_source", "source_line", "source_payload", "semantic_text_sha256",
]  # fmt: skip


class ImportRefused(RuntimeError):
    pass


@dataclass
class LoadedSources:
    hashes: dict[str, str]
    legacy_hashes: dict[str, str]
    official_hashes: dict[str, str]
    files: dict[str, str]
    root_meta: dict[str, Any]
    lines: list[IndexLine]
    occurrences: list[Occurrence]
    duplicates: list[dict[str, Any]]
    official_entries: list[OfficialEntry]
    official_texts: list[OfficialText]
    official_catchwords: list[OfficialCatchword]
    semantic_texts: dict[str, str]
    hierarchy: dict[str, tuple]
    report: ValidationReport
    taxonomy_version: str
    expected_manifest: str | None


def load_and_validate(settings: Settings) -> LoadedSources:
    legacy_paths = {k: settings.source_path(k) for k in ("index", "tree", "duplicates")}
    official_paths = {k: settings.source_path(k) for k in ("wipo_scheme", "wipo_definitions", "wipo_catchwords")}
    paths = {**legacy_paths, **official_paths}
    for kind, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"source file missing ({kind}): {path}")
    hashes = {k: sha256_file(p) for k, p in paths.items()}
    legacy_hashes = {k: hashes[k] for k in legacy_paths}
    official_hashes = {k: hashes[k] for k in official_paths}

    idx = read_index(legacy_paths["index"])
    root_meta, occurrences = traverse(read_tree(legacy_paths["tree"]))
    duplicates = read_duplicates(legacy_paths["duplicates"])
    versions = {ln.record.taxonomy_version for ln in idx.lines}
    version = next(iter(versions)) if len(versions) == 1 else str(root_meta.get("version"))

    scheme_version, official_entries, scheme_texts = parse_scheme(official_paths["wipo_scheme"])
    scheme_notes = [item for item in scheme_texts if item.source_kind == "scheme_note"]
    scheme_references = [item for item in scheme_texts if item.source_kind == "scheme_reference"]
    definitions_version, definitions = parse_definitions(official_paths["wipo_definitions"], official_entries)
    catchwords_version, official_catchwords = parse_catchwords(official_paths["wipo_catchwords"])
    official_texts = [*scheme_texts, *definitions]
    # Embedded and searched text follows the WIPO path, including every dotted subgroup parent.
    by_code = {entry.code: entry for entry in official_entries}
    semantic_texts = {entry.code: semantic_text(entry, by_code) for entry in official_entries}

    def title(code: str | None) -> str | None:
        return by_code[code].title if code in by_code else None

    # Written with each record so a re-import never passes through the supplied (flattened) hierarchy;
    # mirrors the UPDATE in _upsert_official_data.
    hierarchy = {e.code: (e.record_type, e.section_code, title(e.section_code), e.class_code, title(e.class_code),
                          e.subclass_code, title(e.subclass_code), e.maingroup_code, title(e.maingroup_code),
                          e.parent_code, e.ancestor_codes, e.child_codes, e.depth)
                 for e in official_entries}

    expected, expected_name = None, None
    manifest = settings.manifests_dir / f"expected_snapshot_{version}.json"
    if manifest.exists():
        candidate = json.loads(manifest.read_text(encoding="utf-8"))
        if candidate.get("source_hashes") == legacy_hashes:
            expected, expected_name = candidate, manifest.name

    report = validate_sources(
        idx.lines, idx.errors, idx.duplicate_codes, root_meta, occurrences, duplicates, expected,
        official_hierarchy=True,
    )
    record_codes = {ln.record.code for ln in idx.lines}
    official_codes = {entry.code for entry in official_entries}
    scheme_diffs = sorted(record_codes ^ official_codes)
    edition_diffs = [
        {"source": name, "expected": version, "actual": actual}
        for name, actual in (("scheme", scheme_version), ("definitions", definitions_version),
                             ("catchwords", catchwords_version))
        if actual != version
    ]
    report.add(
        "wipo_scheme_code_mismatch",
        [{"code": code, "in_index": code in record_codes, "in_wipo_scheme": code in official_codes}
         for code in scheme_diffs],
        blocking=True,
        note="The WIPO master scheme must account for every supplied catalogue symbol.",
    )
    report.add("wipo_edition_mismatch", edition_diffs, blocking=True)
    if len(official_codes) != len(official_entries):
        report.add("duplicate_wipo_scheme_symbols", [len(official_entries) - len(official_codes)], blocking=True)
    report.counts.update({
        "wipo_scheme_entries": len(official_entries),
        "wipo_classification_entries": sum(e.entry_type == "K" for e in official_entries),
        "wipo_indexing_entries": sum(e.entry_type == "I" for e in official_entries),
        "wipo_scheme_notes": len(scheme_notes),
        "wipo_scheme_references": len(scheme_references),
        "wipo_definition_parts": len(definitions),
        "wipo_catchwords": len(official_catchwords),
        "wipo_range_scoped_notes": sum(1 for item in scheme_notes if item.scope_end_symbol),
        **{f"wipo_rule_kind_{kind}": sum(1 for item in official_texts if item.rule_kind == kind)
           for kind in sorted({item.rule_kind for item in official_texts})},
        "wipo_precedence_rules": sum(1 for item in official_texts for rule in item.rules
                                     if rule["kind"] == "precedence"),
        "wipo_conditional_precedence_rules": sum(1 for item in official_texts for rule in item.rules
                                                 if rule["kind"] == "precedence" and rule["conditional"]),
    })
    root_meta = {**root_meta, "wipo_master_edition": scheme_version,
                 "wipo_source_root": "IPCScheme", "wipo_classification_entries": report.counts["wipo_classification_entries"]}
    report.source_quality.append(
        f"WIPO IPC master files for {scheme_version} supply the authoritative hierarchy, classification (K) vs indexing (I) entry types, "
        f"status, {len(scheme_notes)} scheme-note paragraphs, and {len(definitions)} definition parts."
    )
    if expected is None:
        report.source_quality.append(
            "Source hashes do not match a known validated snapshot manifest; expected counts were not checked."
        )
    return LoadedSources(
        hashes=hashes,
        legacy_hashes=legacy_hashes,
        official_hashes=official_hashes,
        files={k: p.name for k, p in paths.items()},
        root_meta=root_meta,
        lines=idx.lines,
        occurrences=occurrences,
        duplicates=duplicates,
        official_entries=official_entries,
        official_texts=official_texts,
        official_catchwords=official_catchwords,
        semantic_texts=semantic_texts,
        hierarchy=hierarchy,
        report=report,
        taxonomy_version=version,
        expected_manifest=expected_name,
    )


def _record_row(ln: IndexLine, version: str, occ_by_code: dict[str, list[Occurrence]],
                semantic_texts: dict[str, str], hierarchy: dict[str, tuple]) -> tuple:
    r = ln.record
    chosen, reason = select_canonical_occurrence(r.title, r.canonical_title, occ_by_code.get(r.code, []))
    text = semantic_texts.get(r.code, r.semantic_text)
    (rtype, section, section_title, class_, class_title, subclass, subclass_title, maingroup, maingroup_title,
     parent, ancestors, children, depth) = hierarchy.get(r.code) or (
        r.type, r.section, r.section_title, r.class_, r.class_title, r.subclass, r.subclass_title,
        r.maingroup, r.maingroup_title, r.parent_code, r.ancestor_codes, r.child_codes, r.depth)
    return (
        version, r.code, r.display_code, rtype, r.title, r.title_raw, text,
        section or r.section, section_title or r.section_title, class_, class_title, subclass, subclass_title,
        maingroup, maingroup_title, parent, ancestors, children, depth,
        Jsonb(r.aliases), r.heading_text, r.context_heading, r.titles, r.canonical_title,
        r.is_duplicate_symbol,
        chosen.original_title if chosen else None,
        Jsonb({"occurrence_id": chosen.occurrence_id if chosen else None, "reason": reason,
               "occurrence_count": len(occ_by_code.get(r.code, []))}),
        ln.line_no, Jsonb(ln.raw), sha256_text(text),
    )  # fmt: skip


def _upsert_records(conn: Connection, rows: list[tuple]) -> int:
    cols = ", ".join(RECORD_COLUMNS)
    updatable = [c for c in RECORD_COLUMNS if c not in ("taxonomy_version", "code")]
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS stage_records "
            "(LIKE ipc_records INCLUDING DEFAULTS EXCLUDING GENERATED) ON COMMIT DELETE ROWS"
        )
        with cur.copy(f"COPY stage_records ({cols}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)
        distinct = " OR ".join(f"ipc_records.{c} IS DISTINCT FROM EXCLUDED.{c}" for c in updatable)
        cur.execute(
            f"INSERT INTO ipc_records ({cols}) SELECT {cols} FROM stage_records "
            f"ON CONFLICT (taxonomy_version, code) DO UPDATE SET {set_clause} WHERE {distinct}"
        )
        return cur.rowcount


def _job_update(conn: Connection, job_id: str, **fields: Any) -> None:
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k} = %s")
        vals.append(Jsonb(v) if isinstance(v, dict | list) else v)
    sets.append("updated_at = now()")
    conn.execute(f"UPDATE ingestion_jobs SET {', '.join(sets)} WHERE job_id = %s", [*vals, job_id])


def _ensure_release(conn: Connection, src: LoadedSources) -> str:
    row = conn.execute(
        "SELECT source_hashes, official_source_hashes, import_state FROM taxonomy_releases "
        "WHERE taxonomy_version = %s",
        [src.taxonomy_version],
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO taxonomy_releases (taxonomy_version, source_hashes, official_source_hashes, "
            "source_files, source_root, import_state) VALUES (%s, %s, %s, %s, %s, 'importing')",
            [src.taxonomy_version, Jsonb(src.legacy_hashes), Jsonb(src.official_hashes),
             Jsonb(src.files), Jsonb(src.root_meta)],
        )
        return "created"
    if row[0] != src.legacy_hashes:
        raise ImportRefused(
            f"release {src.taxonomy_version} already exists with different source hashes; "
            "refusing to merge different snapshots. Use a new taxonomy_version or remove the old release."
        )
    if row[1] and row[1] != src.official_hashes:
        raise ImportRefused(
            f"release {src.taxonomy_version} already exists with different WIPO master-file hashes; "
            "refusing to merge different official snapshots."
        )
    conn.execute(
        "UPDATE taxonomy_releases SET official_source_hashes = %s, source_files = %s, source_root = %s, "
        "import_state = CASE WHEN import_state = 'validated' THEN 'validated' ELSE 'importing' END, "
        "updated_at = now() WHERE taxonomy_version = %s",
        [Jsonb(src.official_hashes), Jsonb(src.files), Jsonb(src.root_meta), src.taxonomy_version],
    )
    return "resumed"


def _upsert_official_data(conn: Connection, src: LoadedSources) -> None:
    """Replace official IPC hierarchy/text atomically; supplied-tree occurrences stay as provenance."""
    version = src.taxonomy_version
    with conn.cursor() as cur:
        cur.execute("SET CONSTRAINTS ALL DEFERRED")
        cur.execute("DELETE FROM ipc_official_catchwords WHERE taxonomy_version = %s", [version])
        cur.execute("DELETE FROM ipc_official_texts WHERE taxonomy_version = %s", [version])
        cur.execute("DELETE FROM ipc_official_scheme_entries WHERE taxonomy_version = %s", [version])
        with cur.copy(
            "COPY ipc_official_scheme_entries (taxonomy_version, code, kind, entry_type, editions, title, "
            "parent_code, ancestor_codes, child_codes, depth, record_type, section_code, class_code, "
            "subclass_code, maingroup_code, source_order) FROM STDIN"
        ) as copy:
            for entry in src.official_entries:
                copy.write_row((version, entry.code, entry.kind, entry.entry_type, entry.editions, entry.title,
                                entry.parent_code, entry.ancestor_codes, entry.child_codes, entry.depth,
                                entry.record_type, entry.section_code, entry.class_code, entry.subclass_code,
                                entry.maingroup_code, entry.source_order))
        with cur.copy(
            "COPY ipc_official_texts (taxonomy_version, owner_symbol, source_kind, part_type, ordinal, "
            "note_type, content_text, content_xml, related_symbols, container_symbol, scope_end_symbol, "
            "scope_symbols, editions, rule_kind, rules) FROM STDIN"
        ) as copy:
            for item in src.official_texts:
                copy.write_row((version, item.owner_symbol, item.source_kind, item.part_type, item.ordinal,
                                item.note_type, item.text, item.content_xml, item.related_symbols,
                                item.container_symbol, item.scope_end_symbol, item.scope_symbols, item.editions,
                                item.rule_kind, Jsonb(item.rules)))
        with cur.copy(
            "COPY ipc_official_catchwords (taxonomy_version, catchword_id, target_ordinal, phrase, "
            "parent_phrase, target_symbol) FROM STDIN"
        ) as copy:
            for row in src.official_catchwords:
                targets = row.target_symbols or [None]
                for ordinal, target in enumerate(targets):
                    copy.write_row((version, row.catchword_id, ordinal, row.phrase, row.parent_phrase, target))

        # Both K (classification) and I (indexing code) entries remain stored. The canonical
        # parent path follows WIPO, and retrieval joins the official table to admit only K entries.
        cur.execute(
            """
            UPDATE ipc_records AS r SET
                record_type = o.record_type,
                parent_code = o.parent_code,
                ancestor_codes = o.ancestor_codes,
                child_codes = o.child_codes,
                depth = o.depth,
                section_code = COALESCE(o.section_code, r.section_code),
                section_title = COALESCE(sec.title, r.section_title),
                class_code = o.class_code,
                class_title = cls.title,
                subclass_code = o.subclass_code,
                subclass_title = sub.title,
                maingroup_code = o.maingroup_code,
                maingroup_title = mg.title,
                official_title = o.title,
                official_keywords = COALESCE(k.words, '')
            FROM ipc_official_scheme_entries AS o
            LEFT JOIN ipc_official_scheme_entries AS sec
              ON sec.taxonomy_version = o.taxonomy_version AND sec.code = o.section_code
            LEFT JOIN ipc_official_scheme_entries AS cls
              ON cls.taxonomy_version = o.taxonomy_version AND cls.code = o.class_code
            LEFT JOIN ipc_official_scheme_entries AS sub
              ON sub.taxonomy_version = o.taxonomy_version AND sub.code = o.subclass_code
            LEFT JOIN ipc_official_scheme_entries AS mg
              ON mg.taxonomy_version = o.taxonomy_version AND mg.code = o.maingroup_code
            LEFT JOIN LATERAL (
                SELECT string_agg(DISTINCT phrase, ' ' ORDER BY phrase) AS words
                FROM ipc_official_catchwords cw
                WHERE cw.taxonomy_version = o.taxonomy_version AND cw.target_symbol = o.code
            ) AS k ON true
            WHERE r.taxonomy_version = o.taxonomy_version AND r.code = o.code
              AND o.taxonomy_version = %s
            """,
            [version],
        )
    conn.commit()


def _resume_point(conn: Connection, version: str, hashes: dict[str, str]) -> tuple[str, int]:
    row = conn.execute(
        "SELECT job_id, last_checkpoint FROM ingestion_jobs WHERE kind = 'import' AND taxonomy_version = %s"
        " AND state IN ('running','failed') ORDER BY started_at DESC LIMIT 1",
        [version],
    ).fetchone()
    if row and row[1] and row[1].get("source_hashes") == hashes:
        return row[0], int(row[1].get("records_done", 0))
    job_id = f"import-{uuid.uuid4()}"
    conn.execute(
        "INSERT INTO ingestion_jobs (job_id, kind, taxonomy_version, state) VALUES (%s, 'import', %s, 'running')",
        [job_id, version],
    )
    return job_id, 0


def run_import(
    conn: Connection, settings: Settings, src: LoadedSources, *, activate: bool = False
) -> dict[str, Any]:
    if not src.report.ok:
        raise ImportRefused(f"source validation failed: {src.report.blocking}")

    version = src.taxonomy_version
    release_state = _ensure_release(conn, src)
    job_id, done = _resume_point(conn, version, src.hashes)
    _job_update(conn, job_id, state="running")
    conn.commit()

    occ_by_code: dict[str, list[Occurrence]] = defaultdict(list)
    for o in src.occurrences:
        occ_by_code[o.code].append(o)

    ordered = sorted(src.lines, key=lambda ln: (ln.record.depth, ln.line_no))
    batch = settings.import_batch_size
    changed = 0
    try:
        for start in range(done, len(ordered), batch):
            rows = [_record_row(ln, version, occ_by_code, src.semantic_texts, src.hierarchy) for ln in ordered[start : start + batch]]
            changed += _upsert_records(conn, rows)
            _job_update(
                conn,
                job_id,
                progress={"records_done": start + len(rows), "records_total": len(ordered)},
                last_checkpoint={"records_done": start + len(rows), "source_hashes": src.hashes},
            )
            conn.commit()
            log.info("records %d/%d", start + len(rows), len(ordered))

        _upsert_official_data(conn, src)

        # Supplied-tree occurrences and references are supplementary provenance.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ipc_source_occurrences WHERE taxonomy_version = %s", [version])
            with cur.copy(
                "COPY ipc_source_occurrences (taxonomy_version, occurrence_id, code, node_type, source_path,"
                " original_title, direct_child_symbols, parent_symbol, sibling_index, context) FROM STDIN"
            ) as copy:
                for o in src.occurrences:
                    copy.write_row(
                        (version, o.occurrence_id, o.code, o.node_type, o.source_path, o.original_title,
                         o.direct_child_symbols, o.parent_symbol, o.sibling_index, Jsonb(o.context))
                    )  # fmt: skip
            cur.execute("DELETE FROM ipc_reference_edges WHERE taxonomy_version = %s", [version])
            codes = {ln.record.code for ln in src.lines}
            with cur.copy(
                "COPY ipc_reference_edges (taxonomy_version, source_code, target_code, kind,"
                " original_wording, target_exists) FROM STDIN"
            ) as copy:
                for ln in src.lines:
                    r = ln.record
                    wording_src = next(
                        (o.original_title for o in occ_by_code.get(r.code, []) if "(" in o.original_title),
                        r.title_raw or "",
                    )
                    seen: set[tuple[str, str]] = set()
                    for kind, targets in (
                        ("cross_reference", r.cross_references),
                        ("precedence_reference", r.precedence_references),
                    ):
                        for t in targets:
                            if (t, kind) in seen:
                                continue
                            seen.add((t, kind))
                            copy.write_row(
                                (version, r.code, t, kind, reference_wording(wording_src, t), t in codes)
                            )
        counts = _db_counts(conn, version)
        expected_records = len(src.lines)
        expected_catchwords = sum(max(1, len(row.target_symbols)) for row in src.official_catchwords)
        complete = (
            counts["records"] == expected_records
            and counts["occurrences"] == len(src.occurrences)
            and counts["official_scheme_entries"] == len(src.official_entries)
            and counts["official_texts"] == len(src.official_texts)
            and counts["official_catchwords"] == expected_catchwords
        )
        report = src.report.as_dict()
        report["database_counts"] = counts
        conn.execute(
            "UPDATE taxonomy_releases SET validation_report = %s, import_state = %s, updated_at = now()"
            " WHERE taxonomy_version = %s",
            [Jsonb(report), "validated" if complete else "imported", version],
        )
        _job_update(conn, job_id, state="succeeded", finished_at=datetime.now(UTC))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        _job_update(conn, job_id, state="failed", errors=[{"error": str(exc)[:2000]}])
        conn.commit()
        raise

    if activate and complete:
        activate_release(conn, version)

    manifest = {
        "kind": "import",
        "job_id": job_id,
        "taxonomy_version": version,
        "release": release_state,
        "source_files": src.files,
        "source_hashes": src.hashes,
        "source_root": src.root_meta,
        "expected_manifest": src.expected_manifest,
        "rows_changed": changed,
        "database_counts": counts,
        "complete": complete,
        "activated": bool(activate and complete),
        "validation": src.report.as_dict(),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_manifest(settings.manifests_dir, f"import_{version}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}", manifest)
    return manifest


def _db_counts(conn: Connection, version: str) -> dict[str, int]:
    q = {
        "records": "SELECT count(*) FROM ipc_records WHERE taxonomy_version = %s",
        "occurrences": "SELECT count(*) FROM ipc_source_occurrences WHERE taxonomy_version = %s",
        "reference_edges": "SELECT count(*) FROM ipc_reference_edges WHERE taxonomy_version = %s",
        "official_scheme_entries": "SELECT count(*) FROM ipc_official_scheme_entries WHERE taxonomy_version = %s",
        "official_texts": "SELECT count(*) FROM ipc_official_texts WHERE taxonomy_version = %s",
        "official_catchwords": "SELECT count(*) FROM ipc_official_catchwords WHERE taxonomy_version = %s",
    }
    return {k: conn.execute(sql, [version]).fetchone()[0] for k, sql in q.items()}


def activate_release(conn: Connection, version: str) -> None:
    row = conn.execute(
        "SELECT import_state FROM taxonomy_releases WHERE taxonomy_version = %s", [version]
    ).fetchone()
    if row is None:
        raise ImportRefused(f"release {version} not found")
    if row[0] != "validated":
        raise ImportRefused(f"release {version} is {row[0]!r}; only a validated release can be activated")
    conn.execute("UPDATE taxonomy_releases SET is_active = false, updated_at = now() WHERE is_active")
    conn.execute(
        "UPDATE taxonomy_releases SET is_active = true, activated_at = now(), updated_at = now()"
        " WHERE taxonomy_version = %s",
        [version],
    )
    conn.commit()


def write_manifest(directory: Path, stem: str, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path
