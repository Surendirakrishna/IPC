"""Read and type-check the supplied source files without modifying them."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RECORD_TYPES = ("section", "class", "subclass", "maingroup", "subgroup")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IndexRecord(BaseModel):
    """One line of ipc_classification_index.jsonl. Unknown fields are kept (extra='allow')."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    code: str = Field(min_length=1)
    display_code: str = Field(min_length=1)
    type: str
    title: str
    title_raw: str | None = None
    section: str
    section_title: str
    class_: str | None = Field(default=None, alias="class")
    class_title: str | None = None
    subclass: str | None = None
    subclass_title: str | None = None
    maingroup: str | None = None
    maingroup_title: str | None = None
    parent_code: str | None = None
    ancestor_codes: list[str]
    child_codes: list[str]
    depth: int
    cross_references: list[str] = []
    precedence_references: list[str] = []
    aliases: list[Any] = []
    heading_text: list[str] = []
    taxonomy_version: str
    semantic_text: str = Field(min_length=1)
    is_duplicate_symbol: bool | None = None
    titles: list[str] | None = None
    canonical_title: str | None = None
    context_heading: str | None = None


@dataclass
class IndexLine:
    line_no: int
    raw: dict[str, Any]
    record: IndexRecord


@dataclass
class IndexReadResult:
    lines: list[IndexLine] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    duplicate_codes: dict[str, list[int]] = field(default_factory=dict)


def read_index(path: Path) -> IndexReadResult:
    """Stream the JSONL, validating each line. Duplicate canonical codes are reported, not merged."""
    result = IndexReadResult()
    first_seen: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rec = IndexRecord.model_validate(raw)
            except Exception as exc:  # noqa: BLE001 - report every malformed line
                result.errors.append({"line": line_no, "error": str(exc)[:500]})
                continue
            if rec.type not in RECORD_TYPES:
                result.errors.append({"line": line_no, "error": f"unknown type {rec.type!r}"})
                continue
            if rec.code in first_seen:
                result.duplicate_codes.setdefault(rec.code, [first_seen[rec.code]]).append(line_no)
                continue
            first_seen[rec.code] = line_no
            result.lines.append(IndexLine(line_no=line_no, raw=raw, record=rec))
    return result


def iter_index_raw(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def read_tree(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_duplicates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("duplicates file must be a JSON list")
    return data
