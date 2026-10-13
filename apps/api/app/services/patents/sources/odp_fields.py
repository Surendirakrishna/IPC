"""Every USPTO API assumption in one place: endpoint, auth header, query shape and field names.

NOT VERIFIED (2026-10-13): every call without a key returns 401, so none of this has been checked against
a live response. It follows the published PatentsView PatentSearch API documentation (the USPTO's
patent-grant search API that returns title, abstract and IPC-at-issue in one query; PatentsView is part
of the USPTO Open Data Portal family and uses the same kind of key). When a key exists, check each
constant below against one real response and adjust it here only; the reader has no field names of
its own. The open item is recorded in TASK-0004's Work log.

Why not the ODP "Patent File Wrapper" search (``api.uspto.gov/api/v1/patent/applications/search``):
its documented application metadata has no abstract, and the vote needs abstracts.
"""

from __future__ import annotations

from typing import Any

# --- endpoint and auth ---------------------------------------------------------------------------------------
BASE_URL = "https://search.patentsview.org"
SEARCH_PATH = "/api/v1/patent/"
API_KEY_HEADER = "X-Api-Key"
METHOD = "POST"  # the API also accepts GET with q/f/s/o as URL-encoded JSON; POST keeps the key-free URL short

# --- paging --------------------------------------------------------------------------------------------------
PAGE_SIZE_MAX = 1000  # documented maximum for "o": {"size": ...}
SORT_FIELD = "patent_id"  # cursor pagination: "o": {"after": <last sort value>} needs a unique sort key
TOTAL_FIELD = "total_hits"
RECORDS_FIELD = "patents"
ERROR_FIELD = "error"

# --- rate limit ------------------------------------------------------------------------------------------------
RATE_LIMIT_STATUS = 429  # documented limit: 45 requests per minute per key
RETRY_AFTER_HEADER = "Retry-After"

# --- record fields (response) ----------------------------------------------------------------------------------
F_ID = "patent_id"
F_TITLE = "patent_title"
F_ABSTRACT = "patent_abstract"
F_DATE = "patent_date"  # grant (publication) date, YYYY-MM-DD
F_TYPE = "patent_type"
F_APPLICATION = "application"  # nested list of applications
F_FILING_DATE = "filing_date"  # inside "application"
F_IPCR = "ipcr"  # nested list: IPC at issue
IPCR_SECTION = "ipc_section"
IPCR_CLASS = "ipc_class"
IPCR_SUBCLASS = "ipc_subclass"
IPCR_MAIN_GROUP = "ipc_main_group"
IPCR_SUBGROUP = "ipc_subgroup"
IPCR_SEQUENCE = "ipc_sequence"  # 0-based order as printed
IPCR_POSITION = "ipc_symbol_position"  # "F" = first (main) symbol, "L" = later
MAIN_POSITION = "F"

UTILITY = "utility"

# Fields requested ("f"). Nested fields use dotted names.
REQUEST_FIELDS: list[str] = [
    F_ID, F_TITLE, F_ABSTRACT, F_DATE, F_TYPE,
    f"{F_APPLICATION}.{F_FILING_DATE}",
    *(f"{F_IPCR}.{f}" for f in (IPCR_SEQUENCE, IPCR_POSITION, IPCR_SECTION, IPCR_CLASS, IPCR_SUBCLASS,
                                IPCR_MAIN_GROUP, IPCR_SUBGROUP)),
]  # fmt: skip


def year_query(year: int) -> dict[str, Any]:
    """Utility patents granted in ``year`` ("q")."""
    return {"_and": [{"_gte": {F_DATE: f"{year}-01-01"}}, {"_lte": {F_DATE: f"{year}-12-31"}}, {F_TYPE: UTILITY}]}


def request_body(year: int, size: int, after: str | None) -> dict[str, Any]:
    options: dict[str, Any] = {"size": size}
    if after is not None:
        options["after"] = after
    return {"q": year_query(year), "f": REQUEST_FIELDS, "s": [{SORT_FIELD: "asc"}], "o": options}


def structured_label(ipcr: dict[str, Any]) -> dict[str, Any]:
    """One IPC entry in the shape ``labels.resolve_labels(..., "structured")`` reads."""
    return {
        "section": ipcr.get(IPCR_SECTION),
        "class": ipcr.get(IPCR_CLASS),
        "subclass": ipcr.get(IPCR_SUBCLASS),
        "main_group": ipcr.get(IPCR_MAIN_GROUP),
        "subgroup": ipcr.get(IPCR_SUBGROUP),
    }


def label_order_key(ipcr: dict[str, Any]) -> tuple[int, int]:
    """Main (first-position) symbol first, then by printed sequence."""
    position = str(ipcr.get(IPCR_POSITION) or "").upper()
    try:
        seq = int(ipcr.get(IPCR_SEQUENCE))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        seq = 1 << 30
    return (0 if position == MAIN_POSITION else 1, seq)


def filing_date(record: dict[str, Any]) -> Any:
    apps = record.get(F_APPLICATION) or []
    if isinstance(apps, dict):
        apps = [apps]
    for app in apps:
        if isinstance(app, dict) and app.get(F_FILING_DATE):
            return app[F_FILING_DATE]
    return None
