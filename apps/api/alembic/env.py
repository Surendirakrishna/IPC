"""Alembic environment: always migrates through the direct (non-pooled) connection."""

from __future__ import annotations

from alembic import context
from app.db.session import get_direct_engine

config = context.config


def run_migrations_online() -> None:
    engine = get_direct_engine()
    with engine.connect() as connection:
        context.configure(connection=connection, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("Offline SQL generation is not supported; run against a database.")
run_migrations_online()
