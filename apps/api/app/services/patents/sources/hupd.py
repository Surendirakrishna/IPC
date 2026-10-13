"""Harvard USPTO Patent Dataset (HUPD) reader: one ``YYYY.tar.gz`` per filing year, one JSON per application.

The tarballs are streamed (``tarfile`` mode ``r|gz``), never extracted. Fields used: application_number,
title, abstract, filing_date, date_published, main_ipcr_label, ipcr_labels. IPC labels are compact
("G16H1060" = G16H 10/60), so records carry ``label_scheme="compact"``.

Matches the offline corpus build (data/manifests/benchmarks/knn_hupd/prepare_corpus.py): abstracts
shorter than 50 characters (after stripping) are skipped, the main label comes first, duplicates are
dropped, and title / abstract are only stripped. Unlike the offline build, labels are not filtered
here: every non-blank label is passed on, and resolution (with its statistics) happens at ingestion.

Nothing from a record's text is logged; only counts and member names.
"""

from __future__ import annotations

import json
import logging
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from app.services.patents.sources.record import (
    MIN_ABSTRACT_CHARS,
    PatentRecord,
    ReaderStats,
    parse_date,
    stripped,
)

log = logging.getLogger(__name__)

SOURCE = "hupd"


def hupd_paths(data_dir: str | Path, years: Iterable[int | str]) -> list[Path]:
    """``data_dir/YYYY.tar.gz`` for each year; a missing tarball is an error, not a silent gap."""
    paths = [Path(data_dir) / f"{int(y)}.tar.gz" for y in years]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"HUPD tarball(s) not found: {', '.join(missing)}")
    return paths


def hupd_labels(rec: dict[str, Any]) -> tuple[str, ...]:
    """Main IPC label first, then the others, upper-cased, blanks and duplicates dropped."""
    raw = [rec.get("main_ipcr_label")] + list(rec.get("ipcr_labels") or [])
    out: list[str] = []
    for label in raw:
        code = stripped(label).upper()
        if code and code not in out:
            out.append(code)
    return tuple(out)


def hupd_record(rec: dict[str, Any], *, fallback_id: str) -> PatentRecord:
    title = stripped(rec.get("title"))
    return PatentRecord(
        source=SOURCE,
        document_id=stripped(rec.get("application_number")) or fallback_id,
        title=title or None,
        abstract=stripped(rec.get("abstract")),
        filing_date=parse_date(rec.get("filing_date")),
        publication_date=parse_date(rec.get("date_published")),
        raw_labels=hupd_labels(rec),
        label_scheme="compact",
    )


class HupdReader:
    """Iterate ``PatentRecord``s from one or more HUPD year tarballs, in tarball and member order.

    ``stats`` is updated while iterating: ``read`` counts JSON members, ``skipped_reasons`` has
    ``short_abstract`` (under ``min_abstract_chars``) and ``unreadable`` (not valid JSON / not an object).
    """

    def __init__(self, paths: Iterable[str | Path], *, min_abstract_chars: int = MIN_ABSTRACT_CHARS) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("at least one HUPD tarball is required")
        self.min_abstract_chars = min_abstract_chars
        self.stats = ReaderStats()

    @classmethod
    def for_years(cls, data_dir: str | Path, years: Iterable[int | str], **kwargs: Any) -> HupdReader:
        return cls(hupd_paths(data_dir, years), **kwargs)

    def __iter__(self) -> Iterator[PatentRecord]:
        for path in self.paths:
            yield from self._read_tarball(path)
        log.info("HUPD read %s", self.stats.as_dict())

    def _read_tarball(self, path: Path) -> Iterator[PatentRecord]:
        with tarfile.open(path, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                self.stats.read += 1
                fh = tf.extractfile(member)
                try:
                    rec = json.load(fh) if fh is not None else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    rec = None
                if not isinstance(rec, dict):
                    self.stats.skip("unreadable")
                    log.warning("HUPD %s: skipping unreadable member %s", path.name, member.name)
                    continue
                record = hupd_record(rec, fallback_id=Path(member.name).stem)
                if len(record.abstract) < self.min_abstract_chars:
                    self.stats.skip("short_abstract")
                    continue
                self.stats.yielded += 1
                yield record
