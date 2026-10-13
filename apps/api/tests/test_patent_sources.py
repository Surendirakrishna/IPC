"""TASK-0004: patent source readers (HUPD tarballs, USPTO grants via an Open Data Portal key).

No network: the USPTO reader runs against recorded fixture responses through httpx.MockTransport.
"""

from __future__ import annotations

import json
import logging
import tarfile
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from app.services.patents.labels import resolve_labels
from app.services.patents.sources import (
    HupdReader,
    MissingApiKey,
    OdpCursor,
    PatentRecord,
    UsptoOdpReader,
    hupd_paths,
)
from app.services.patents.sources import odp_fields as F
from app.services.patents.sources.record import parse_date
from app.services.patents.sources.uspto_odp import (
    UsptoApiError,
    UsptoAuthError,
    UsptoReaderError,
    odp_record,
    retry_after_seconds,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "patents"
HUPD_SAMPLE = FIXTURES / "hupd_2017_sample.tar.gz"
RESPONSES = json.loads((FIXTURES / "uspto_odp_responses.json").read_text(encoding="utf-8"))


# --- PatentRecord ------------------------------------------------------------------------------------------


def test_patent_record_fields() -> None:
    names = [f for f in PatentRecord.__dataclass_fields__]
    assert names == ["source", "document_id", "title", "abstract", "filing_date", "publication_date", "raw_labels",
                     "label_scheme"]  # fmt: skip


@pytest.mark.parametrize(("value", "want"), [
    ("20171005", date(2017, 10, 5)), ("2021-02-09", date(2021, 2, 9)), ("2021-02-09T00:00:00Z", date(2021, 2, 9)),
    ("", None), (None, None), ("None", None), ("2017-13-45", None),
])  # fmt: skip
def test_parse_date(value, want) -> None:  # noqa: ANN001
    assert parse_date(value) == want


# --- HUPD ------------------------------------------------------------------------------------------------------


def test_hupd_fixture_has_three_records() -> None:
    with tarfile.open(HUPD_SAMPLE, mode="r:gz") as tf:
        assert sorted(m.name for m in tf.getmembers() if m.isfile()) == [
            "2017/15900001.json", "2017/15900002.json", "2017/15900003.json"]


def test_hupd_reader_streams_records_and_skips_short_abstracts() -> None:
    reader = HupdReader([HUPD_SAMPLE])
    records = list(reader)
    assert [r.document_id for r in records] == ["15900001", "15900002"]
    assert reader.stats.as_dict() == {"read": 3, "yielded": 2, "skipped": 1, "skipped_reasons": {"short_abstract": 1}}

    first, second = records
    assert first == PatentRecord(
        source="hupd", document_id="15900001", title="SYSTEM FOR DISTRIBUTING QUANTUM KEYS BETWEEN GRID NODES",
        abstract="A key management module distributes quantum keys between smart grid nodes over\n"
                 "authenticated classical channels and rotates them on detected interference.",
        filing_date=date(2017, 10, 5), publication_date=date(2018, 4, 12),
        raw_labels=("H04L908", "H04L2906"),  # main first; the duplicate "h04l908 " is dropped
        label_scheme="compact",
    )  # fmt: skip
    # inner whitespace is kept (only stripped), as the offline corpus did
    assert "\n" in first.abstract
    # no title -> None; blank main label -> the others in order; blank publication date -> None
    assert second.title is None and second.raw_labels == ("F16H2522", "B62D504") and second.publication_date is None


def test_hupd_reader_reads_several_tarballs_and_counts_across_them(tmp_path: Path) -> None:
    other = tmp_path / "2016.tar.gz"
    other.write_bytes(HUPD_SAMPLE.read_bytes())
    reader = HupdReader([HUPD_SAMPLE, other], min_abstract_chars=10)
    assert len(list(reader)) == 6 and reader.stats.read == 6 and reader.stats.skipped == 0


def test_hupd_reader_skips_unreadable_members(tmp_path: Path) -> None:
    import io

    path = tmp_path / "2018.tar.gz"
    with tarfile.open(path, mode="w:gz") as tf:
        for name, data in (("2018/a.json", b"{not json"), ("2018/b.json", b"[1, 2]"), ("2018/notes.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    reader = HupdReader([path])
    assert list(reader) == []
    assert reader.stats.as_dict() == {"read": 2, "yielded": 0, "skipped": 2, "skipped_reasons": {"unreadable": 2}}


def test_hupd_paths(tmp_path: Path) -> None:
    (tmp_path / "2017.tar.gz").write_bytes(b"")
    assert hupd_paths(tmp_path, [2017]) == [tmp_path / "2017.tar.gz"]
    with pytest.raises(FileNotFoundError, match="2016.tar.gz"):
        hupd_paths(tmp_path, ["2017", "2016"])
    with pytest.raises(ValueError):
        HupdReader([])


def test_hupd_labels_resolve_with_the_compact_scheme() -> None:
    catalogue = {"H04L", "H04L 9/00", "H04L 9/08", "H04L 29/00", "F16H", "F16H 25/00", "F16H 25/22", "B62D"}
    first = next(iter(HupdReader([HUPD_SAMPLE])))
    resolved = resolve_labels(first.raw_labels, first.label_scheme, catalogue)
    assert [tuple(lab) for lab in resolved.labels] == [("H04L", "H04L 9/00", "H04L 9/08"), ("H04L", "H04L 29/00", None)]


def test_hupd_reader_does_not_log_abstract_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    records = list(HupdReader([HUPD_SAMPLE]))
    assert records
    for r in records:
        assert r.abstract[:40] not in caplog.text
    assert "Too short to vote" not in caplog.text
    assert "read" in caplog.text  # counts are logged


# --- USPTO (recorded responses, no network) ----------------------------------------------------------------------


class Api:
    """MockTransport handler serving the recorded pages; ``script`` injects failures before a page."""

    def __init__(self, script: list | None = None) -> None:
        self.script = list(script or [])
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if isinstance(step, httpx.Response):
                return step
        body = json.loads(request.content)
        year = body["q"]["_and"][0]["_gte"][F.F_DATE][:4]
        after = body["o"].get("after") or "start"
        return httpx.Response(200, json=RESPONSES[f"{year}:{after}"])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self), base_url=F.BASE_URL)


KEY = "odp-test-" + uuid.uuid4().hex


def _reader(api: Api, **kwargs) -> tuple[UsptoOdpReader, list[float]]:  # noqa: ANN003
    sleeps: list[float] = []
    kwargs.setdefault("page_size", 2)
    reader = UsptoOdpReader(kwargs.pop("years", [2021, 2022]), kwargs.pop("api_key", KEY), client=api.client(),
                            sleep=sleeps.append, **kwargs)  # fmt: skip
    return reader, sleeps


@pytest.mark.parametrize("key", [None, "", "   ", SecretStr("")])
def test_uspto_reader_without_a_key_names_the_setting(key) -> None:  # noqa: ANN001
    api = Api()
    with pytest.raises(MissingApiKey, match="USPTO_ODP_API_KEY"):
        UsptoOdpReader([2021], key, client=api.client())
    assert api.requests == []


def test_uspto_reader_reads_every_year_and_page() -> None:
    api = Api()
    reader, sleeps = _reader(api)
    records = list(reader)
    assert [r.document_id for r in records] == ["10912345", "10912346", "10912400", "11212001"]
    assert reader.stats.as_dict() == {"read": 5, "yielded": 4, "skipped": 1, "skipped_reasons": {"short_abstract": 1}}
    assert reader.cursor.done and reader.cursor.yielded == 4 and sleeps == []
    assert len(api.requests) == 4  # 2021: full page + short page; 2022: full page + empty page

    first = records[0]
    assert first.source == "uspto_odp" and first.label_scheme == "structured"
    assert first.title == "Quantum key distribution for multi-node smart grids"
    assert first.filing_date == date(2018, 11, 20) and first.publication_date == date(2021, 2, 9)
    assert first.raw_labels[0] == {"section": "H", "class": "04", "subclass": "L", "main_group": "9", "subgroup": "08"}
    # main ("F") symbol first even when listed second; duplicates dropped
    assert [lab["subclass"] for lab in records[1].raw_labels] == ["H", "D"] and records[1].title == "Ball screw steering drive"
    assert records[2].title is None and records[2].raw_labels == () and records[2].filing_date is None


def test_uspto_request_shape_and_key_header() -> None:
    api = Api()
    reader, _ = _reader(api, years=[2021])
    list(reader)
    req = api.requests[0]
    assert req.method == "POST" and req.url.path == F.SEARCH_PATH and str(req.url).startswith(F.BASE_URL)
    assert req.headers[F.API_KEY_HEADER] == KEY
    assert KEY not in str(req.url) and KEY not in req.content.decode()
    body = json.loads(req.content)
    assert body["q"] == F.year_query(2021) and body["f"] == F.REQUEST_FIELDS
    assert body["s"] == [{"patent_id": "asc"}] and body["o"] == {"size": 2}
    assert json.loads(api.requests[1].content)["o"] == {"size": 2, "after": "10912346"}


def test_uspto_structured_labels_resolve() -> None:
    catalogue = {"H04L", "H04L 9/00", "H04L 9/08", "G06F", "G06F 21/00", "G06F 21/60"}
    record = odp_record(RESPONSES["2021:start"]["patents"][0])
    resolved = resolve_labels(record.raw_labels, record.label_scheme, catalogue)
    assert [tuple(lab) for lab in resolved.labels] == [("H04L", "H04L 9/00", "H04L 9/08"),
                                                       ("G06F", "G06F 21/00", "G06F 21/60")]  # fmt: skip


def test_uspto_rate_limit_honours_retry_after() -> None:
    api = Api(script=[httpx.Response(429, headers={"Retry-After": "7"}),
                      httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})])  # past date -> 0
    reader, sleeps = _reader(api, years=[2021])
    assert len(list(reader)) == 3
    assert sleeps == [7.0, 0.0]


def test_uspto_backs_off_on_server_and_transport_errors() -> None:
    api = Api(script=[httpx.Response(503), httpx.ConnectTimeout("timed out"), httpx.Response(500),
                      httpx.Response(429)])  # 429 without Retry-After: backoff too
    reader, sleeps = _reader(api, years=[2021], backoff_s=1.5, max_backoff_s=5.0)
    assert len(list(reader)) == 3
    assert sleeps == [1.5, 3.0, 5.0, 5.0]  # 1.5 * 2**n, capped


def test_uspto_gives_up_after_max_retries() -> None:
    api = Api(script=[httpx.Response(503)] * 3)
    reader, sleeps = _reader(api, years=[2021], max_retries=2)
    with pytest.raises(UsptoApiError, match="giving up after 3 attempts"):
        list(reader)
    assert len(sleeps) == 2


def test_uspto_rejected_key_fails_without_retry() -> None:
    api = Api(script=[httpx.Response(401, json=RESPONSES["unauthorized"])])
    reader, sleeps = _reader(api)
    with pytest.raises(UsptoAuthError, match="USPTO_ODP_API_KEY") as err:
        list(reader)
    assert sleeps == [] and len(api.requests) == 1 and KEY not in str(err.value)


def test_uspto_other_client_errors_and_bad_bodies() -> None:
    reader, _ = _reader(Api(script=[httpx.Response(400, text="bad query")]))
    with pytest.raises(UsptoApiError, match="HTTP 400"):
        list(reader)
    reader, _ = _reader(Api(script=[httpx.Response(200, json={"error": True, "patents": []})]))
    with pytest.raises(UsptoApiError, match="reported an error"):
        list(reader)
    reader, _ = _reader(Api(script=[httpx.Response(200, text="<html>")]))
    with pytest.raises(UsptoApiError, match="non-JSON"):
        list(reader)


def test_uspto_limit_and_resume_from_cursor() -> None:
    api = Api()
    reader, _ = _reader(api, limit=2)
    first = list(reader)
    assert [r.document_id for r in first] == ["10912345", "10912346"]
    token = reader.cursor.to_token()
    assert OdpCursor.from_token(token) == OdpCursor(year=2021, after="10912346", yielded=2)

    resumed, _ = _reader(Api(), cursor=token)  # no limit: read to the end
    rest = list(resumed)
    assert [r.document_id for r in rest] == ["10912400", "11212001"]
    assert resumed.cursor.done

    capped, _ = _reader(Api(), cursor=token, limit=3)  # the limit counts records yielded before the resume
    assert [r.document_id for r in capped] == ["10912400"]


def test_uspto_resume_mid_page_and_pages_api() -> None:
    reader, _ = _reader(Api(), limit=1)
    assert [r.document_id for r in reader] == ["10912345"]
    assert reader.cursor == OdpCursor(year=2021, after="10912345", yielded=1)
    # A resume asks for records after 10912345; the recorded API has no such page, so serve it from the fixture.
    page = {**RESPONSES["2021:start"], "patents": RESPONSES["2021:start"]["patents"][1:], "count": 1}
    api = Api(script=[httpx.Response(200, json=page)])
    resumed, _ = _reader(api, cursor=reader.cursor)
    pages = list(resumed.pages())
    assert json.loads(api.requests[0].content)["o"]["after"] == "10912345"
    # The page after 10912345 is short (1 < page_size 2), so it ends 2021 and the cursor moves to 2022.
    assert [[r.document_id for r in recs] for recs, _ in pages] == [["10912346"], ["11212001"], []]
    assert [(c.year, c.after, c.yielded, c.done) for _, c in pages] == [
        (2022, None, 2, False), (2022, "11212001", 3, False), (2022, "11212001", 3, True)]


def test_uspto_cursor_validation() -> None:
    with pytest.raises(UsptoReaderError, match="cursor year 2019"):
        UsptoOdpReader([2021], KEY, cursor=OdpCursor(year=2019))
    with pytest.raises(UsptoReaderError, match="invalid USPTO cursor"):
        UsptoOdpReader([2021], KEY, cursor="not-a-token")
    with pytest.raises(ValueError):
        UsptoOdpReader([], KEY)
    with pytest.raises(ValueError):
        UsptoOdpReader([2021], KEY, page_size=1001)


def test_retry_after_parsing() -> None:
    now = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
    assert retry_after_seconds("12", now=now) == 12.0
    assert retry_after_seconds("Tue, 13 Oct 2026 12:00:30 GMT", now=now) == 30.0
    assert retry_after_seconds(None) is None and retry_after_seconds("soon") is None


def test_uspto_reader_never_logs_the_key_or_abstracts(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)  # includes httpx's own request logging
    api = Api(script=[httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(503)])
    reader, _ = _reader(api)
    records = list(reader)
    assert "rate limited" in caplog.text and "server error" in caplog.text
    assert KEY not in caplog.text and KEY not in repr(reader)
    for r in records:
        assert r.abstract[:30] not in caplog.text
    assert "Too short to vote" not in caplog.text
