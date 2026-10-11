"""Database engines.

- direct engine: migrations, bulk ingestion, embedding/index jobs
- pooled engine: API traffic, bounded SQLAlchemy pool, statement timeout

Both require TLS for non-local hosts. URLs are never logged unredacted.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

import psycopg
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "db", "postgres"}


class DatabaseNotConfigured(RuntimeError):
    pass


def to_sqlalchemy_url(raw: str) -> str:
    """Use the psycopg 3 driver and require TLS for remote hosts."""
    parts = urllib.parse.urlsplit(raw)
    scheme = "postgresql+psycopg"
    query = dict(urllib.parse.parse_qsl(parts.query))
    if (parts.hostname or "") not in LOCAL_HOSTS:
        query.setdefault("sslmode", "require")
    return urllib.parse.urlunsplit(
        (scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), parts.fragment)
    )


def _build_engine(raw: str, settings: Settings, *, api: bool) -> Engine:
    """api: bounded request pool + statement timeout; otherwise a small pool for jobs."""
    engine = create_engine(
        to_sqlalchemy_url(raw),
        pool_size=settings.db_pool_size if api else 2,
        max_overflow=settings.db_pool_max_overflow if api else 1,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=settings.db_connect_timeout_s,
        connect_args={
            "connect_timeout": settings.db_connect_timeout_s,
            "application_name": "ipc-classifier-api" if api else "ipc-classifier-jobs",
            "prepare_threshold": 5,
        },
    )

    if api:
        timeout = int(settings.db_statement_timeout_ms)

        @event.listens_for(engine, "connect")
        def _set_timeouts(dbapi_conn, _record):  # pragma: no cover - driver hook
            with dbapi_conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {timeout}")
            dbapi_conn.commit()

    return engine


@lru_cache
def get_pooled_engine() -> Engine:
    s = get_settings()
    url = s.pooled_url()
    if url is None:
        raise DatabaseNotConfigured(f"database for target {s.ipc_db_target!r} is not configured (SAKTHI_DB_* in .env)")
    return _build_engine(url.get_secret_value(), s, api=True)


@lru_cache
def get_direct_engine() -> Engine:
    s = get_settings()
    url = s.direct_url()
    if url is None:
        raise DatabaseNotConfigured(f"database for target {s.ipc_db_target!r} is not configured (SAKTHI_DB_* in .env)")
    return _build_engine(url.get_secret_value(), s, api=False)


def database_configured() -> bool:
    return get_settings().pooled_url() is not None


def connect_direct() -> psycopg.Connection:
    """Raw psycopg connection for COPY/bulk work through the direct endpoint."""
    s = get_settings()
    url = s.direct_url()
    if url is None:
        raise DatabaseNotConfigured(f"database for target {s.ipc_db_target!r} is not configured (SAKTHI_DB_* in .env)")
    raw = url.get_secret_value()
    host = urllib.parse.urlsplit(raw).hostname or ""
    kwargs: dict = {"connect_timeout": s.db_connect_timeout_s, "application_name": "ipc-classifier-jobs"}
    if host not in LOCAL_HOSTS and "sslmode=" not in raw:
        kwargs["sslmode"] = "require"
    return psycopg.connect(raw, **kwargs)


@contextmanager
def api_session() -> Iterator[Session]:
    factory = sessionmaker(bind=get_pooled_engine(), expire_on_commit=False)
    with factory() as session:
        yield session


@contextmanager
def job_session() -> Iterator[Session]:
    factory = sessionmaker(bind=get_direct_engine(), expire_on_commit=False)
    with factory() as session:
        yield session
