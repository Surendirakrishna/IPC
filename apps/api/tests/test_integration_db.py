"""Integration tests against a real PostgreSQL+pgvector database holding the imported snapshot.

Run:  IPC_DB_TARGET=local uv run pytest -m integration
(requires `docker compose -f infra/docker-compose.yml up -d`, `ipc --target local migrate`,
`ipc --target local import --activate`).
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    if os.environ.get("IPC_DB_TARGET") != "local":
        pytest.skip("set IPC_DB_TARGET=local to run DB integration tests")
    from fastapi.testclient import TestClient

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_health_is_cheap(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_display_code_lookup_preserves_source_codes(client):
    r = client.get("/v1/ipc/B60L 53/12")
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == "B60L0053120000" and body["display_code"] == "B60L 53/12"
    assert [p["level"] for p in body["parent_path"]] == ["section", "class", "subclass", "maingroup"]
    assert client.get("/v1/ipc/B60L0053120000").json()["display_code"] == "B60L 53/12"


def test_unknown_code_404(client):
    r = client.get("/v1/ipc/Z99Z 1/00")
    assert r.status_code == 404 and r.json()["error"] == "code_not_found"


def test_duplicate_symbol_contexts_preserved(client):
    body = client.get("/v1/ipc/A01B 3/00").json()
    assert body["duplicate_context"]["is_duplicate_symbol"] is True
    assert len(body["source_occurrences"]) == 2
    assert {o["original_title"] for o in body["source_occurrences"]} >= {"Ploughs"}


def test_keyword_search_is_fulltext_not_and(client):
    r = client.get("/v1/ipc/search", params={"q": "key distribution", "record_type": "subgroup"})
    body = r.json()
    assert r.status_code == 200 and body["total"] > 0
    assert any(i["display_code"] == "H04L 9/08" for i in body["items"][:20])
    assert "not BM25" in body["ranking"]


def test_code_prefix_search(client):
    body = client.get("/v1/ipc/search", params={"q": "H04L 9/0"}).json()
    assert body["items"] and all(i["display_code"].startswith("H04L 9/0") for i in body["items"])


def test_status_reports_source_quality(client):
    body = client.get("/v1/taxonomy/status").json()
    assert body["corpus"]["active_release"] == "20260101"
    assert body["corpus"]["records_total"] == 80145
    assert any("CPC" in w for w in body["source_warnings"])
    assert "not BM25" in body["keyword_mode"]


def test_classify_validation_and_unavailable(client):
    r = client.post("/v1/classify", json={"abstract": "   "})
    assert r.status_code == 422 and r.json()["error"] == "invalid_abstract"
    r = client.post("/v1/classify", json={"title": "x"})
    assert r.status_code == 422 and r.json()["error"] == "invalid_abstract"
    # Models are not loaded in this test process (IPC_SKIP_MODEL_LOAD=1): explicit 503, never a canned result.
    r = client.post("/v1/classify", json={"abstract": "A quantum key distribution method."})
    assert r.status_code == 503 and r.json()["error"] == "models_unavailable"


def test_reimport_is_idempotent():
    if os.environ.get("IPC_DB_TARGET") != "local":
        pytest.skip("local only")
    from app.config import get_settings
    from app.db.session import connect_direct
    from app.services.ingestion.importer import load_and_validate, run_import

    s = get_settings()
    src = load_and_validate(s)
    with connect_direct() as conn:
        manifest = run_import(conn, s, src)
    assert manifest["rows_changed"] == 0
    assert manifest["database_counts"] == {"records": 80145, "occurrences": 80734, "reference_edges": 15035}
