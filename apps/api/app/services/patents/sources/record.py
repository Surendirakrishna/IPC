"""The source-independent patent record every reader yields (consumed by ingestion, TASK-0005)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

LabelScheme = Literal["compact", "structured"]
MIN_ABSTRACT_CHARS = 50


@dataclass(frozen=True)
class PatentRecord:
    """One labelled patent document as its source wrote it.

    ``raw_labels`` keep the source's form, main label first, without duplicates:
    - ``compact`` (HUPD): strings such as "G16H1060" (no main group / subgroup separator);
    - ``structured`` (USPTO): mappings with section / class / subclass / main_group / subgroup, which
      ``labels.resolve_labels(..., "structured", ...)`` accepts as is.
    Labels are resolved against the catalogue at ingestion, not here.
    """

    source: str  # "hupd" | "uspto_odp"
    document_id: str  # application number (HUPD) or patent number (USPTO grants)
    title: str | None
    abstract: str
    filing_date: date | None
    publication_date: date | None
    raw_labels: tuple[Any, ...]
    label_scheme: LabelScheme


@dataclass
class ReaderStats:
    """Counts a reader reports. ``skipped`` is broken down by reason in ``skipped_reasons``."""

    read: int = 0
    yielded: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        return sum(self.skipped_reasons.values())

    def skip(self, reason: str) -> None:
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {"read": self.read, "yielded": self.yielded, "skipped": self.skipped,
                "skipped_reasons": dict(self.skipped_reasons)}  # fmt: skip


def parse_date(value: Any) -> date | None:
    """'20171005', '2017-10-05' or an ISO datetime -> date; blank or unparseable -> None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def stripped(value: Any) -> str:
    """Leading/trailing whitespace removed, nothing else: inner whitespace is kept exactly as the source
    wrote it, as the offline corpus did (``prepare_corpus.py``), so the offline PaECTER vectors stay
    valid for the same documents (TASK-0006 compares texts before importing a vector)."""
    return "" if value is None else str(value).strip()
