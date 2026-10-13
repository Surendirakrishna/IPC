"""USPTO patent-grant reader (Open Data Portal key, ``USPTO_ODP_API_KEY``) for recent labelled grants.

Yields utility patents granted in the requested years, with title, abstract, filing and grant dates and
IPC at issue as ``structured`` labels (main symbol first). Endpoint, query and field names live in
``odp_fields`` and are NOT verified against a live response yet (no key on 2026-10-13).

Behaviour:
- No key: ``MissingApiKey`` naming ``USPTO_ODP_API_KEY`` before any request.
- Rate limits: HTTP 429 waits for ``Retry-After`` (seconds or an HTTP date), else exponential backoff.
  Server errors (5xx) and transport errors (timeouts, resets) back off exponentially. A rejected key
  (401/403) fails at once.
- Resumable: ``cursor`` (an ``OdpCursor``, serialisable with ``to_token``) always points just past the last
  record handed out (yielded or skipped), so a new reader started from it continues without gaps or
  repeats. ``pages()`` yields each page's records together with the cursor to persist after storing them.
- ``limit`` caps the records yielded, across years and across resumes (the cursor carries the count).

The key is sent only in the request header and is never logged or put in an error message; abstract text
is never logged.
"""

from __future__ import annotations

import base64
import email.utils
import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import SecretStr

from app.services.patents.sources import odp_fields as F
from app.services.patents.sources.record import (
    MIN_ABSTRACT_CHARS,
    PatentRecord,
    ReaderStats,
    parse_date,
    stripped,
)

log = logging.getLogger(__name__)

SOURCE = "uspto_odp"
ENV_NAME = "USPTO_ODP_API_KEY"


class UsptoReaderError(RuntimeError):
    pass


class MissingApiKey(UsptoReaderError):
    def __init__(self) -> None:
        super().__init__(
            f"no USPTO Open Data Portal API key: set {ENV_NAME} in the repository .env (or the environment) "
            "to load USPTO grants; HUPD needs no key"
        )


class UsptoAuthError(UsptoReaderError):
    pass


class UsptoApiError(UsptoReaderError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class OdpCursor:
    """Where a read stands: the year being read, the last sort key handed out in it, records yielded so far."""

    year: int
    after: str | None = None
    yielded: int = 0
    done: bool = False

    def to_token(self) -> str:
        return base64.urlsafe_b64encode(json.dumps(asdict(self), separators=(",", ":")).encode()).decode()

    @classmethod
    def from_token(cls, token: str) -> OdpCursor:
        try:
            data = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
            return cls(year=int(data["year"]), after=data.get("after"), yielded=int(data.get("yielded", 0)),
                       done=bool(data.get("done", False)))  # fmt: skip
        except (ValueError, KeyError, TypeError) as exc:
            raise UsptoReaderError("invalid USPTO cursor token") from exc


def _key(api_key: SecretStr | str | None) -> SecretStr:
    if isinstance(api_key, SecretStr):
        api_key = api_key.get_secret_value()
    if api_key is None or not str(api_key).strip():
        raise MissingApiKey()
    return SecretStr(str(api_key).strip())


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    """``Retry-After`` as seconds: a number, or an HTTP date relative to ``now``. None if absent or invalid."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def odp_record(raw: dict[str, Any]) -> PatentRecord:
    ipcr = raw.get(F.F_IPCR) or []
    if isinstance(ipcr, dict):
        ipcr = [ipcr]
    labels: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in sorted((e for e in ipcr if isinstance(e, dict)), key=F.label_order_key):
        label = F.structured_label(entry)
        key = tuple(label.values())
        if all(v in (None, "") for v in key) or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    title = stripped(raw.get(F.F_TITLE))
    return PatentRecord(
        source=SOURCE,
        document_id=stripped(raw.get(F.F_ID)),
        title=title or None,
        abstract=stripped(raw.get(F.F_ABSTRACT)),
        filing_date=parse_date(F.filing_date(raw)),
        publication_date=parse_date(raw.get(F.F_DATE)),
        raw_labels=tuple(labels),
        label_scheme="structured",
    )


class UsptoOdpReader:
    def __init__(
        self,
        years: Iterable[int],
        api_key: SecretStr | str | None,
        *,
        limit: int | None = None,
        page_size: int = F.PAGE_SIZE_MAX,
        cursor: OdpCursor | str | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 8,
        backoff_s: float = 2.0,
        max_backoff_s: float = 120.0,
        timeout_s: float = 60.0,
        min_abstract_chars: int = MIN_ABSTRACT_CHARS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = _key(api_key)  # first: no key, no further checks, no request
        self.years = sorted({int(y) for y in years})
        if not self.years:
            raise ValueError("at least one year is required")
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        if not 1 <= page_size <= F.PAGE_SIZE_MAX:
            raise ValueError(f"page_size must be in [1, {F.PAGE_SIZE_MAX}]")
        self.limit = limit
        self.page_size = page_size
        start = OdpCursor.from_token(cursor) if isinstance(cursor, str) else cursor
        if start is None:
            start = OdpCursor(year=self.years[0])
        if start.year not in self.years:
            raise UsptoReaderError(f"cursor year {start.year} is not among the requested years {self.years}")
        self.cursor = start
        self._client = client
        self._own_client = client is None
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self.timeout_s = timeout_s
        self.min_abstract_chars = min_abstract_chars
        self.sleep = sleep
        self.stats = ReaderStats()
        self.requests = 0

    def __repr__(self) -> str:  # never shows the key
        return f"UsptoOdpReader(years={self.years}, limit={self.limit}, cursor={self.cursor!r})"

    # --- HTTP --------------------------------------------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=F.BASE_URL, timeout=self.timeout_s)
        return self._client

    def close(self) -> None:
        if self._own_client and self._client is not None:
            self._client.close()
            self._client = None

    def _backoff(self, attempt: int) -> float:
        return min(self.max_backoff_s, self.backoff_s * (2 ** attempt))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            self.requests += 1
            try:
                resp = self._http().request(F.METHOD, F.SEARCH_PATH, json=body,
                                            headers={F.API_KEY_HEADER: self._api_key.get_secret_value()})  # fmt: skip
            except httpx.TransportError as exc:
                wait, reason = self._backoff(attempt), f"transport error ({type(exc).__name__})"
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        raise UsptoApiError("USPTO API returned a non-JSON body", status) from exc
                    if not isinstance(data, dict):
                        raise UsptoApiError("USPTO API returned an unexpected body", status)
                    if data.get(F.ERROR_FIELD) is True:
                        raise UsptoApiError("USPTO API reported an error in a 200 response", status)
                    return data
                if status in (401, 403):
                    raise UsptoAuthError(f"USPTO API rejected the request (HTTP {status}); check {ENV_NAME}")
                if status == F.RATE_LIMIT_STATUS:
                    hinted = retry_after_seconds(resp.headers.get(F.RETRY_AFTER_HEADER))
                    wait = hinted if hinted is not None else self._backoff(attempt)
                    reason = "rate limited (HTTP 429)"
                elif status >= 500:
                    wait, reason = self._backoff(attempt), f"server error (HTTP {status})"
                else:
                    raise UsptoApiError(f"USPTO API request failed (HTTP {status})", status)
            if attempt >= self.max_retries:
                raise UsptoApiError(f"USPTO API: giving up after {attempt + 1} attempts; last: {reason}")
            log.warning("USPTO API %s; retrying in %.1f s (attempt %d of %d)", reason, wait, attempt + 1,
                        self.max_retries)  # fmt: skip
            self.sleep(wait)
            attempt += 1

    # --- iteration -------------------------------------------------------------------------------------------

    def _remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.cursor.yielded)

    def pages(self) -> Iterator[tuple[list[PatentRecord], OdpCursor]]:
        """Each page's records and the cursor to persist once they are stored."""
        try:
            while not self.cursor.done and self._remaining() != 0:
                year = self.cursor.year
                data = self._post(F.request_body(year, self.page_size, self.cursor.after))
                rows = data.get(F.RECORDS_FIELD) or []
                if not isinstance(rows, list):
                    raise UsptoApiError(f"USPTO API: {F.RECORDS_FIELD!r} is not a list")
                out: list[PatentRecord] = []
                complete = True
                for raw in rows:
                    if self._remaining() == 0:
                        complete = False
                        break
                    record = odp_record(raw if isinstance(raw, dict) else {})
                    self.stats.read += 1
                    if not record.document_id:
                        raise UsptoApiError(f"USPTO API: a record has no {F.F_ID!r}")
                    yielded = self.cursor.yielded
                    if len(record.abstract) < self.min_abstract_chars:
                        self.stats.skip("short_abstract")
                    else:
                        self.stats.yielded += 1
                        yielded += 1
                        out.append(record)
                    self.cursor = replace(self.cursor, after=str(raw.get(F.SORT_FIELD, record.document_id)),
                                          yielded=yielded)  # fmt: skip
                if complete and len(rows) < self.page_size:  # a short (or empty) page ends the year
                    self.cursor = self._next_year()
                yield out, self.cursor
            log.info("USPTO read %s (%d requests)", self.stats.as_dict(), self.requests)
        finally:
            self.close()

    def _next_year(self) -> OdpCursor:
        i = self.years.index(self.cursor.year)
        if i + 1 < len(self.years):
            return OdpCursor(year=self.years[i + 1], after=None, yielded=self.cursor.yielded)
        return replace(self.cursor, done=True)

    def __iter__(self) -> Iterator[PatentRecord]:
        for records, _ in self.pages():
            yield from records
