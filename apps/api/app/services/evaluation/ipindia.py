"""Parse saved IP India Public Search 'PatentDetails' pages into benchmark documents.

Extracted per application: number, title, publication number/date, priority claim, the office's
'Field Of Invention' label, the recorded 'Classification (IPC)' symbols, abstract, and the complete
specification split into description and claims when a 'Claims:' section is present.
"""

from __future__ import annotations

import html as htmllib
import re
from pathlib import Path
from typing import Any

from app.services.confidence.ipc_codes import parse_symbols

ROW_RE = re.compile(r'<td style="width: 200px">\s*([^<]+?)\s*</td>\s*<td>(.*?)</td>', re.S)
ABSTRACT_RE = re.compile(r"<strong>\s*Abstract:\s*</strong>(.*?)</td>", re.S)
SPEC_RE = re.compile(r'<textarea[^>]*id="COMPLETE_SPECIFICATION"[^>]*>(.*?)</textarea>', re.S)
CLAIMS_SPLIT_RE = re.compile(r"(?:,\s*C\s*,\s*)?\bClaims\s*:", re.I)


def _clean(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", fragment)
    text = re.sub(r"<[^>]+>", " ", text)
    text = htmllib.unescape(text).replace("\r\n", "\n").replace("�", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def parse_detail(html: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for label, value in ROW_RE.findall(html):
        fields.setdefault(label.strip(), _clean(value))
    abstract_m = ABSTRACT_RE.search(html)
    spec_m = SPEC_RE.search(html)
    spec = htmllib.unescape(spec_m.group(1)).replace("\r\n", "\n").replace("�", " ").strip() if spec_m else ""
    description, claims = spec, None
    parts = list(CLAIMS_SPLIT_RE.finditer(spec))
    if parts:
        cut = parts[-1]
        description, claims = spec[: cut.start()].strip(), spec[cut.end():].strip() or None
    description = re.sub(r"^\s*Description\s*:\s*", "", description).strip() or None
    ipc_raw = fields.get("Classification (IPC)", "")
    return {
        "application_number": fields.get("Application Number", "").strip(),
        "title": fields.get("Invention Title") or None,
        "publication_number": fields.get("Publication Number") or None,
        "publication_date": fields.get("Publication Date") or None,
        "filing_date": fields.get("Application Filing Date") or None,
        # Priority claims group applications into families (see services/evaluation/families.py).
        "priority_number": fields.get("Priority Number") or None,
        "priority_country": fields.get("Priority Country") or None,
        "priority_date": fields.get("Priority Date") or None,
        "field_of_invention": fields.get("Field Of Invention") or None,
        "ipc_raw": ipc_raw,
        "ground_truth": [s.display for s in parse_symbols(ipc_raw)],
        "abstract": _clean(abstract_m.group(1)) if abstract_m else None,
        "specification": description,
        "claims": claims,
    }


def parse_dir(details_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    docs, skipped = [], []
    for f in sorted(details_dir.glob("*.html")):
        d = parse_detail(f.read_text(encoding="utf-8", errors="replace"))
        reason = None
        if not d["abstract"] or len(d["abstract"].strip()) < 50:
            reason = "no abstract"
        elif not d["ground_truth"]:
            reason = f"no parseable IPC symbol in {d['ipc_raw']!r}"
        if reason:
            skipped.append({"file": f.name, "reason": reason})
            continue
        d["source_file"] = f.name
        docs.append(d)
    return docs, skipped
