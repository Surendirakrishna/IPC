"""Parse the versioned WIPO IPC master files without flattening their hierarchy or rules.

Each WIPO package holds one XML per language (EN_*.xml, FR_*.xml); the English file is parsed.

Every note, title reference and definition part keeps the symbol that owns it, the range it applies
to (a note's symbol..endSymbol), its note type, and the edition(s) recorded for it. Its wording is
also read for what it means, because a reference is not an instruction to assign the cited code:

- precedence            "Y takes precedence [over X]": when both places apply, Y is used instead of X.
- place_last/place_first priority rule for the stated scope.
- limiting_reference    the subject matter named in the reference is classified in the cited place.
- informative_reference, indexing, definition, scope, other: context for review only.
"""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

WIPO_NS = "http://www.wipo.int/classifications/ipc/masterfiles"
XHTML_NS = "http://www.w3.org/1999/xhtml"
NS = {"i": WIPO_NS, "x": XHTML_NS}
TAG = f"{{{WIPO_NS}}}"
CLASSIFICATION_KINDS = {
    "s": "section",
    "c": "class",
    "u": "subclass",
    "m": "maingroup",
    **{str(n): "subgroup" for n in range(1, 10)},
}
LEVEL_LABELS = {"section": "Section", "class": "Class", "subclass": "Subclass", "maingroup": "Main group"}
RULE_KINDS = ("precedence", "place_last", "place_first", "indexing", "definition", "limiting_reference",
              "informative_reference", "scope", "other")
SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])|\n+")
BLOCK_TAGS = {"p", "li", "div", "dd", "dt", "td", "tr", "br", "noteParagraph", "subnote"}
PRECEDENCE_RE = re.compile(r"\btakes?\s+precedence\b", re.IGNORECASE)
OVER_RE = re.compile(r"^\W*over\b", re.IGNORECASE)
OTHER_RE = re.compile(r"\bother\b", re.IGNORECASE)
INTRO_RE = re.compile(r"^\s*in\b", re.IGNORECASE)
NEGATION_RE = re.compile(r"\bnot\s+(?:be\s+)?applied\b|\bdoes\s+not\s+take\s+precedence\b", re.IGNORECASE)
# "must also be classified in the last appropriate place" asks for an additional code, not a choice of place.
ALSO_CLASSIFIED_RE = re.compile(r"\balso\s+(?:be\s+)?classified\b", re.IGNORECASE)
# A rule that only holds under a stated condition needs review; it is not applied automatically.
CONDITION_RE = re.compile(r"^\s*(?:for|if|when|where|insofar|in\s+the\s+case)\b|\be\.g\.|\bunless\b", re.IGNORECASE)
LAST_PLACE_RE = re.compile(r"\blast(?:\s+appropriate)?\s+place\b", re.IGNORECASE)
FIRST_PLACE_RE = re.compile(r"\bfirst(?:\s+appropriate)?\s+place\b", re.IGNORECASE)
INDEXING_RE = re.compile(r"\bindexing\s+(?:codes?|schemes?)\b", re.IGNORECASE)
DEFINITION_RE = re.compile(
    r"\bmeaning\s+indicated\b|\bfollowing\s+(?:terms?|expressions?)\b|\b(?:is|are)\s+used\s+with\s+the\s+meaning\b",
    re.IGNORECASE,
)
LIMITING_RE = re.compile(r"\bcovered\s+by\b|\b(?:is|are)\s+classified\s+in\b|\bprovided\s+for\s+in\b|\bsee\b",
                         re.IGNORECASE)
INFORMATIVE_RE = re.compile(r"\battention\s+is\s+drawn\b", re.IGNORECASE)
SCOPE_RE = re.compile(r"\bcovers\b", re.IGNORECASE)
RANGE_JOINERS = {"-", "–", "—", "to"}
DEFINITION_PART_KINDS = {
    "LIMITINGREFERENCES": "limiting_reference",
    "REFERENCESOUTOFARESIDUALPLACE": "limiting_reference",
    "INFORMATIVEREFERENCES": "informative_reference",
    "APPLICATIONORIENTEDREFERENCES": "informative_reference",
    "GLOSSARYOFTERMS": "definition",
    "SYNONYMSANDKEYWORDS": "definition",
    "SYNONYMS": "definition",
    "ABBREVIATIONS": "definition",
    "SPECIALMEANINGS": "definition",
    "INSTEADOFWORDS": "definition",
    "DEFINITION-STATEMENT": "scope",
}

# A symbol reference in document order: (start, end); end is None for a single symbol.
Ref = tuple[str, str | None]
Sequence = list[str | Ref]


@dataclass
class OfficialEntry:
    code: str
    kind: str
    entry_type: str
    editions: str | None
    title: str
    parent_code: str | None
    ancestor_codes: list[str]
    child_codes: list[str] = field(default_factory=list)
    depth: int = 1
    record_type: str = ""
    section_code: str | None = None
    class_code: str | None = None
    subclass_code: str | None = None
    maingroup_code: str | None = None
    source_order: int = 0


@dataclass
class OfficialText:
    owner_symbol: str
    source_kind: str
    part_type: str
    ordinal: int
    note_type: str | None
    text: str
    content_xml: str
    related_symbols: list[str]
    container_symbol: str | None = None
    scope_end_symbol: str | None = None
    scope_symbols: list[str] = field(default_factory=list)
    editions: str | None = None
    rule_kind: str = "other"
    rules: list[dict[str, Any]] = field(default_factory=list)
    # Unresolved parse of the wording; resolved into rules once the whole scheme is known.
    sequence: Sequence = field(default_factory=list, repr=False)
    default_kind: str | None = field(default=None, repr=False)


@dataclass
class OfficialCatchword:
    catchword_id: str
    phrase: str
    target_symbols: list[str]
    parent_phrase: str | None = None


def display_symbol(code: str) -> str:
    """A01B0003040000 -> A01B 3/04; sections, classes and subclasses are unchanged."""
    if len(code) != 14 or not code[4:].isdigit():
        return code
    return f"{code[:4]} {int(code[4:8])}/{code[8:].rstrip('0').ljust(2, '0')}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_package(path: Path, expected_root: str) -> tuple[ET.Element, str]:
    with zipfile.ZipFile(path) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if len(xml_names) > 1:
            xml_names = [name for name in xml_names if Path(name).name.upper().startswith("EN_")]
        if len(xml_names) != 1:
            raise ValueError(f"Expected one English XML file in {path.name}, found {len(xml_names)}")
        root = ET.fromstring(archive.read(xml_names[0]))
    if _local(root.tag) != expected_root:
        raise ValueError(f"{path.name} has root {_local(root.tag)!r}, expected {expected_root!r}")
    if root.attrib.get("lang", "EN").upper() != "EN":
        raise ValueError(f"{path.name} is not the English edition")
    edition = root.attrib.get("edition")
    if not edition:
        raise ValueError(f"{path.name} does not declare an edition")
    return root, edition


def _sequence(element: ET.Element, *, skip_tags: set[str] | None = None) -> Sequence:
    """Text and symbol references in document order (sref, mref, and xhtml a rel=symbol)."""
    skip_tags = skip_tags or set()
    out: Sequence = []

    def visit(node: ET.Element) -> None:
        name = _local(node.tag)
        if name in skip_tags:
            return
        if name == "sref" and node.attrib.get("ref"):
            out.append((node.attrib["ref"], None))
            return
        if name == "mref" and node.attrib.get("ref"):
            out.append((node.attrib["ref"], node.attrib.get("endRef")))
            return
        if name == "a" and node.attrib.get("rel") == "symbol" and node.attrib.get("href"):
            out.append((node.attrib["href"], None))
            return
        if node.text:
            out.append(node.text)
        for child in list(node):
            visit(child)
            if child.tail:
                out.append(child.tail)
        if name in BLOCK_TAGS:
            out.append("\n")  # paragraphs, list items and cells end a sentence

    visit(element)
    # Definitions write a range as two anchors joined by a dash: merge them into one range.
    merged: Sequence = []
    for item in out:
        if (isinstance(item, tuple) and len(merged) >= 2 and isinstance(merged[-1], str)
                and merged[-1].strip() in RANGE_JOINERS and isinstance(merged[-2], tuple) and merged[-2][1] is None):
            start = merged[-2][0]
            del merged[-2:]
            merged.append((start, item[0]))
            continue
        merged.append(item)
    return merged


def _render(seq: Sequence) -> str:
    parts: list[str] = []
    for item in seq:
        if isinstance(item, str):
            parts.append(item)
        else:
            start, end = item
            parts.append(f" {display_symbol(start)}" + (f"-{display_symbol(end)}" if end else "") + " ")
    text = SPACE_RE.sub(" ", "".join(parts)).strip()
    return re.sub(r"\s+([.,;:)])", r"\1", text).replace("( ", "(")


def _refs(seq: Sequence) -> list[Ref]:
    return [item for item in seq if isinstance(item, tuple)]


def _ref_symbols(seq: Sequence) -> list[str]:
    return sorted({s for start, end in _refs(seq) for s in (start, end) if s})


def _entry_title(entry: ET.Element) -> str:
    parts = entry.findall("i:textBody/i:title/i:titlePart", NS)
    titles = [_render(_sequence(p, skip_tags={"entryReference"})) for p in parts]
    return "; ".join(t for t in titles if t)


def _entry_local_elements(entry: ET.Element, local_name: str) -> list[ET.Element]:
    body = entry.find("i:textBody", NS)
    if body is None:
        return []
    out: list[ET.Element] = []
    stack = list(reversed(list(body)))
    while stack:
        node = stack.pop()
        if node.tag == TAG + "ipcEntry":
            continue
        if node.tag == TAG + local_name:
            out.append(node)
        else:
            stack.extend(reversed(list(node)))
    return out


def _sentences(seq: Sequence) -> list[Sequence]:
    sentences: list[Sequence] = [[]]
    for item in seq:
        if isinstance(item, str):
            for i, piece in enumerate(SENTENCE_SPLIT_RE.split(item)):
                if i:
                    sentences.append([])
                if piece and piece.strip():
                    sentences[-1].append(piece)
        else:
            sentences[-1].append(item)
    return [s for s in sentences if s]


def _precedence(sentence: Sequence) -> dict[str, Any] | None:
    """Read "<preferred> take(s) precedence [over <dominated>]" from one sentence.

    Returns refs, plus flags for the parts the sentence leaves implicit; the caller fills those
    in from the owning scope. A leading "In <scope>," phrase names the scope, not a preferred place.
    """
    idx = next((i for i, item in enumerate(sentence) if isinstance(item, str) and PRECEDENCE_RE.search(item)), None)
    if idx is None:
        return None
    preferred: list[Ref] = []
    intro: list[Ref] = []
    first_text = next((item for item in sentence if isinstance(item, str)), "")
    in_intro = bool(INTRO_RE.match(first_text)) and isinstance(sentence[0], str)
    for item in sentence[:idx]:
        if isinstance(item, str):
            if in_intro and "," in item:
                in_intro = False
            continue
        (intro if in_intro else preferred).append(item)
    head = sentence[idx]
    assert isinstance(head, str)
    after = head[PRECEDENCE_RE.search(head).end():]  # type: ignore[union-attr]
    dominated: list[Ref] = []
    over = bool(OVER_RE.match(after))
    rest_of_scope = False
    if over:
        dominated = _refs(sentence[idx + 1:])
        # "over other groups [of subclass X]" means the rest of that scope, not only the cited symbols.
        tail = [after]
        for item in sentence[idx + 1:]:
            if isinstance(item, tuple):
                break
            tail.append(item)
        rest_of_scope = bool(OTHER_RE.search(" ".join(tail)))
    return {"preferred": preferred, "dominated": dominated, "intro": intro, "over": over,
            "rest_of_scope": rest_of_scope}


def _intro_refs(sentence: Sequence) -> list[Ref]:
    """Symbols in a leading "In <scope>," phrase."""
    if not sentence or not isinstance(sentence[0], str) or not INTRO_RE.match(sentence[0]):
        return []
    out: list[Ref] = []
    for item in sentence:
        if isinstance(item, str):
            if "," in item:
                break
        else:
            out.append(item)
    return out


class _Expander:
    """Resolve a (start, end) symbol range to the top-level scheme entries it covers."""

    def __init__(self, entries: list[OfficialEntry]) -> None:
        self.entries = sorted(entries, key=lambda e: e.source_order)
        self.order = {e.code: e.source_order for e in entries}
        self.position = {e.code: i for i, e in enumerate(self.entries)}

    def __call__(self, ref: Ref) -> list[str]:
        start, end = ref
        if not end or end == start:
            return [start]
        if start not in self.position or end not in self.position:
            return [start, end]
        a, b = sorted((self.position[start], self.position[end]))
        span = self.entries[a:b + 1]
        codes = {e.code for e in span}
        return [e.code for e in span if e.parent_code not in codes]

    def many(self, refs: list[Ref]) -> list[str]:
        return list(dict.fromkeys(code for ref in refs for code in self(ref)))


def _resolve_rules(item: OfficialText, expand: _Expander) -> None:
    """Fill rule_kind and rules from the wording; precedence and place rules carry explicit scopes."""
    seq = item.sequence
    scope = item.scope_symbols or [item.owner_symbol]
    text = item.text
    rules: list[dict[str, Any]] = []
    for sentence in _sentences(seq):
        parsed = _precedence(sentence)
        if parsed is None:
            continue
        preferred = expand.many(parsed["preferred"])
        if parsed["over"]:
            dominated = expand.many(parsed["dominated"]) or expand.many(parsed["intro"]) or scope
            if parsed["rest_of_scope"] and not parsed["dominated"]:
                dominated = expand.many(parsed["intro"]) or scope
        else:
            # "Y takes precedence" inside X's title or definition: Y is preferred over X itself.
            dominated = expand.many(parsed["intro"]) or ([item.container_symbol] if item.container_symbol else scope)
        if not preferred:
            preferred = scope
        dominated = [d for d in dominated if d not in preferred]
        rendered = _render(sentence)
        if preferred and dominated and not NEGATION_RE.search(rendered):
            rules.append({"kind": "precedence", "preferred": preferred, "dominated": dominated,
                          "rest_of_scope": parsed["rest_of_scope"],
                          "conditional": bool(CONDITION_RE.search(rendered)), "sentence": rendered})
    if not rules:
        for sentence in _sentences(seq):
            rendered = _render(sentence)
            kind = ("place_last" if LAST_PLACE_RE.search(rendered)
                    else "place_first" if FIRST_PLACE_RE.search(rendered) else None)
            if kind and not NEGATION_RE.search(rendered) and not ALSO_CLASSIFIED_RE.search(rendered):
                rules.append({"kind": kind, "scope": expand.many(_intro_refs(sentence)) or scope,
                              "sentence": rendered})
                break
    if rules:
        item.rule_kind = rules[0]["kind"]
    elif item.default_kind in ("limiting_reference", "informative_reference", "definition", "scope"):
        item.rule_kind = item.default_kind
    elif INDEXING_RE.search(text):
        item.rule_kind = "indexing"
    elif DEFINITION_RE.search(text):
        item.rule_kind = "definition"
    elif INFORMATIVE_RE.search(text):
        item.rule_kind = "informative_reference"
    elif item.default_kind == "title_reference" or LIMITING_RE.search(text):
        item.rule_kind = "limiting_reference"
    elif SCOPE_RE.search(text):
        item.rule_kind = "scope"
    else:
        item.rule_kind = "other"
    if item.rule_kind == "limiting_reference":
        targets = expand.many(_refs(seq))
        if targets:
            rules.append({"kind": "limiting_reference", "targets": targets})
    item.rules = rules


def _text_item(element: ET.Element, *, owner: str, source_kind: str, part_type: str, ordinal: int,
               note_type: str | None, container: str | None, scope_end: str | None, editions: str | None,
               default_kind: str | None) -> OfficialText:
    seq = _sequence(element)
    return OfficialText(
        owner_symbol=owner,
        source_kind=source_kind,
        part_type=part_type,
        ordinal=ordinal,
        note_type=note_type,
        text=_render(seq),
        content_xml=ET.tostring(element, encoding="unicode"),
        related_symbols=_ref_symbols(seq),
        container_symbol=container,
        scope_end_symbol=scope_end,
        editions=editions,
        sequence=seq,
        default_kind=default_kind,
    )


def parse_scheme(path: Path) -> tuple[str, list[OfficialEntry], list[OfficialText]]:
    root, edition = _read_package(path, "IPCScheme")
    entries: list[OfficialEntry] = []
    notes: list[OfficialText] = []
    text_ordinals: dict[tuple[str, str, str], int] = defaultdict(int)
    source_order = 0

    def next_ordinal(owner: str, source_kind: str, part_type: str) -> int:
        key = owner, source_kind, part_type
        ordinal = text_ordinals[key]
        text_ordinals[key] += 1
        return ordinal

    def visit(node: ET.Element, path_entries: list[OfficialEntry]) -> None:
        nonlocal source_order
        current_path = path_entries
        if node.tag == TAG + "ipcEntry":
            kind = node.attrib.get("kind", "")
            entry_type = node.attrib.get("entryType", "")
            symbol = node.attrib.get("symbol", "")
            if kind == "n":
                # A note entry sits inside the entry it belongs to; symbol..endSymbol is its scope.
                container = path_entries[-1].code if path_entries else None
                for note in node.findall("i:textBody/i:note", NS):
                    for paragraph in note.findall("i:noteParagraph", NS):
                        notes.append(_text_item(
                            paragraph, owner=symbol or container or "", source_kind="scheme_note", part_type="note",
                            ordinal=next_ordinal(symbol, "scheme_note", "note"),
                            note_type=note.attrib.get("type", "none"), container=container,
                            scope_end=node.attrib.get("endSymbol"), editions=node.attrib.get("edition"),
                            default_kind=None,
                        ))
            elif kind in CLASSIFICATION_KINDS and entry_type in ("K", "I") and symbol:
                ancestors = [p.code for p in path_entries]
                entry = OfficialEntry(
                    code=symbol,
                    kind=kind,
                    entry_type=entry_type,
                    editions=node.attrib.get("edition"),
                    title=_entry_title(node),
                    parent_code=path_entries[-1].code if path_entries else None,
                    ancestor_codes=ancestors,
                    depth=len(ancestors) + 1,
                    record_type=CLASSIFICATION_KINDS[kind],
                    source_order=source_order,
                )
                source_order += 1
                entries.append(entry)
                if path_entries:
                    path_entries[-1].child_codes.append(symbol)
                current_path = [*path_entries, entry]
                for part_type in ("entryReference", "references"):
                    for reference in _entry_local_elements(node, part_type):
                        notes.append(_text_item(
                            reference, owner=symbol, source_kind="scheme_reference", part_type=part_type,
                            ordinal=next_ordinal(symbol, "scheme_reference", part_type),
                            note_type=reference.attrib.get("type"), container=symbol, scope_end=None,
                            editions=entry.editions, default_kind="title_reference",
                        ))

        for child in list(node):
            visit(child, current_path)

    visit(root, [])

    by_code = {entry.code: entry for entry in entries}
    for entry in entries:
        chain = [*(by_code[c] for c in entry.ancestor_codes if c in by_code), entry]
        entry.section_code = next((e.code for e in chain if e.record_type == "section"), None)
        entry.class_code = next((e.code for e in chain if e.record_type == "class"), None)
        entry.subclass_code = next((e.code for e in chain if e.record_type == "subclass"), None)
        entry.maingroup_code = next((e.code for e in reversed(chain) if e.record_type == "maingroup"), None)

    expand = _Expander(entries)
    for item in notes:
        item.scope_symbols = expand((item.owner_symbol, item.scope_end_symbol))
        _resolve_rules(item, expand)
    return edition, entries, notes


def semantic_text(entry: OfficialEntry, by_code: dict[str, OfficialEntry]) -> str:
    """Embedded/searched text: the full WIPO path, including every dotted subgroup parent."""
    chain = [*(by_code[c] for c in entry.ancestor_codes if c in by_code), entry]
    blocks = [f"IPC {display_symbol(entry.code)}"]
    for e in chain:
        if e.record_type == "subgroup":
            dots = int(e.kind)
            label = f"Subgroup ({dots} dot{'s' if dots > 1 else ''})"
        else:
            label = LEVEL_LABELS[e.record_type]
        blocks.append(f"{label}:\n{e.title}")
    return "\n\n".join(blocks)


def _definition_parts(node: ET.Element) -> list[tuple[str, ET.Element]]:
    refs = {"REFERENCES"}
    out: list[tuple[str, ET.Element]] = []

    def visit(part: ET.Element) -> None:
        name = _local(part.tag)
        children = [c for c in list(part) if _local(c.tag).isupper()]
        if name in refs and children:
            for child in children:
                visit(child)
        else:
            out.append((name, part))

    for child in list(node):
        visit(child)
    return out


def parse_definitions(path: Path, entries: list[OfficialEntry] | None = None) -> tuple[str, list[OfficialText]]:
    root, edition = _read_package(path, "IPCDefinitions")
    texts: list[OfficialText] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for definition in root.iter(TAG + "IPC-DEFINITION"):
        owner = definition.attrib.get("IPC")
        if not owner:
            continue
        for part_type, part in _definition_parts(definition):
            key = owner, part_type
            ordinal = counters[key]
            counters[key] += 1
            texts.append(_text_item(
                part, owner=owner, source_kind="definition", part_type=part_type, ordinal=ordinal, note_type=None,
                container=owner, scope_end=None, editions=None, default_kind=DEFINITION_PART_KINDS.get(part_type),
            ))
    expand = _Expander(entries or [])
    for item in texts:
        item.scope_symbols = [item.owner_symbol]
        _resolve_rules(item, expand)
    return edition, texts


def parse_catchwords(path: Path) -> tuple[str, list[OfficialCatchword]]:
    root, edition = _read_package(path, "IPCCatchwordIndex")
    rows: list[OfficialCatchword] = []

    def visit(node: ET.Element, parent_phrase: str | None = None) -> None:
        phrase_node = node.find("i:CWIndication", NS)
        phrase = _render(_sequence(phrase_node)) if phrase_node is not None else ""
        if phrase:
            refs = node.find("i:CWReferences", NS)
            symbols = sorted({ref.attrib.get("ref", "") for ref in (list(refs) if refs is not None else [])
                              if ref.attrib.get("ref")})
            rows.append(OfficialCatchword(node.attrib.get("id", ""), phrase, symbols, parent_phrase))
        for child in node.findall("i:CWEntry", NS):
            visit(child, phrase or parent_phrase)

    for entry in root.findall("i:CWEntry", NS):
        visit(entry)
    return edition, rows
