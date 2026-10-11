"""Controlled, versioned field-of-invention label mapping (an application display category).

Lookup for the primary supported candidate: subclass override -> class alias -> section fallback ->
catalogue class_title in uppercase. Version 2 targets IP India's 'Field Of Invention' vocabulary and was
learned from a dev set (see provenance in field_mapping.json). No candidates => UNDETERMINED with a reason.
The label never comes from the abstract's wording and has no confidence of its own here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.db.repository import Record


@dataclass(frozen=True)
class FieldMapping:
    version: str
    class_aliases: dict[str, str]
    subclass_overrides: dict[str, str]
    section_fallback: dict[str, str] = field(default_factory=dict)

    def label_for(self, rec: Record) -> tuple[str, str]:
        """Return (label, mapping_key)."""
        if rec.subclass_code and rec.subclass_code in self.subclass_overrides:
            return self.subclass_overrides[rec.subclass_code], rec.subclass_code
        key = rec.class_code or rec.section_code
        if key in self.class_aliases:
            return self.class_aliases[key], key
        if rec.section_code in self.section_fallback:
            return self.section_fallback[rec.section_code], rec.section_code
        title = rec.class_title or rec.section_title
        return title.upper(), key


@lru_cache
def load_mapping(path: Path) -> FieldMapping:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FieldMapping(
        version=str(data["mapping_version"]),
        class_aliases=dict(data.get("class_aliases", {})),
        subclass_overrides=dict(data.get("subclass_overrides", {})),
        section_fallback=dict(data.get("section_fallback", {})),
    )


def field_of_invention(mapping: FieldMapping, supported: list[Record], reason_if_empty: str) -> dict[str, Any]:
    if not supported:
        return {
            "label": "UNDETERMINED",
            "mapping_key": None,
            "mapping_version": mapping.version,
            "additional_labels": [],
            "rationale": reason_if_empty,
        }
    primary = supported[0]
    label, key = mapping.label_for(primary)
    additional: list[dict[str, str]] = []
    seen = {label}
    for rec in supported[1:]:
        lab, k = mapping.label_for(rec)
        if lab not in seen:
            seen.add(lab)
            additional.append({"label": lab, "mapping_key": k, "from_code": rec.display_code})
    return {
        "label": label,
        "mapping_key": key,
        "mapping_version": mapping.version,
        "additional_labels": additional,
        "rationale": (
            f"Primary supported candidate {primary.display_code} ({primary.title}) is in class {primary.class_code}"
            f" ({primary.class_title}); mapped via field mapping v{mapping.version}."
        ),
    }
