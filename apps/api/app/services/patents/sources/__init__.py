"""Patent source readers. Every reader yields ``PatentRecord`` so ingestion does not depend on the source."""

from app.services.patents.sources.hupd import HupdReader, hupd_paths
from app.services.patents.sources.record import MIN_ABSTRACT_CHARS, LabelScheme, PatentRecord, ReaderStats
from app.services.patents.sources.uspto_odp import MissingApiKey, OdpCursor, UsptoOdpReader

__all__ = [
    "MIN_ABSTRACT_CHARS", "HupdReader", "LabelScheme", "MissingApiKey", "OdpCursor", "PatentRecord", "ReaderStats",
    "UsptoOdpReader", "hupd_paths",
]  # fmt: skip
