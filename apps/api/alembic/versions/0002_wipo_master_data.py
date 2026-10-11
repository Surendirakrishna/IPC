"""Store authoritative WIPO hierarchy, rules, definitions and catchwords.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE taxonomy_releases ADD COLUMN official_source_hashes jsonb NOT NULL DEFAULT '{}'::jsonb")
    op.execute("ALTER TABLE ipc_records ADD COLUMN official_title text")
    op.execute("ALTER TABLE ipc_records ADD COLUMN official_keywords text NOT NULL DEFAULT ''")
    op.execute("DROP INDEX ipc_records_search")
    op.execute("ALTER TABLE ipc_records DROP COLUMN search_vector")
    op.execute(
        """
        ALTER TABLE ipc_records ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(official_title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(original_title, '')), 'B') ||
            setweight(to_tsvector('english', coalesce(context_heading, '')), 'B') ||
            setweight(to_tsvector('english', coalesce(official_keywords, '')), 'B') ||
            setweight(to_tsvector('english', semantic_text), 'C')
        ) STORED
        """
    )
    op.execute("CREATE INDEX ipc_records_search ON ipc_records USING gin (search_vector)")

    op.execute(
        """
        CREATE TABLE ipc_official_scheme_entries (
            taxonomy_version text NOT NULL REFERENCES taxonomy_releases(taxonomy_version),
            code text NOT NULL,
            kind text NOT NULL,
            entry_type text NOT NULL CHECK (entry_type IN ('K','I')),
            editions text,
            title text NOT NULL DEFAULT '',
            parent_code text,
            ancestor_codes text[] NOT NULL DEFAULT '{}',
            child_codes text[] NOT NULL DEFAULT '{}',
            depth smallint NOT NULL,
            record_type text NOT NULL CHECK (record_type IN ('section','class','subclass','maingroup','subgroup')),
            section_code text,
            class_code text,
            subclass_code text,
            maingroup_code text,
            source_order integer NOT NULL,
            PRIMARY KEY (taxonomy_version, code),
            FOREIGN KEY (taxonomy_version, code) REFERENCES ipc_records(taxonomy_version, code),
            FOREIGN KEY (taxonomy_version, parent_code)
                REFERENCES ipc_official_scheme_entries(taxonomy_version, code) DEFERRABLE INITIALLY DEFERRED
        )
        """
    )
    op.execute(
        "CREATE INDEX ipc_official_scheme_current ON ipc_official_scheme_entries (taxonomy_version, code) "
        "WHERE entry_type = 'K'"
    )

    op.execute(
        """
        CREATE TABLE ipc_official_texts (
            taxonomy_version text NOT NULL REFERENCES taxonomy_releases(taxonomy_version),
            owner_symbol text NOT NULL,
            source_kind text NOT NULL CHECK (source_kind IN ('scheme_note','scheme_reference','definition')),
            part_type text NOT NULL,
            ordinal integer NOT NULL,
            note_type text,
            content_text text NOT NULL,
            content_xml text NOT NULL,
            related_symbols text[] NOT NULL DEFAULT '{}',
            PRIMARY KEY (taxonomy_version, owner_symbol, source_kind, part_type, ordinal)
        )
        """
    )
    op.execute("CREATE INDEX ipc_official_texts_owner ON ipc_official_texts (taxonomy_version, owner_symbol)")

    op.execute(
        """
        CREATE TABLE ipc_official_catchwords (
            taxonomy_version text NOT NULL REFERENCES taxonomy_releases(taxonomy_version),
            catchword_id text NOT NULL,
            target_ordinal smallint NOT NULL,
            phrase text NOT NULL,
            parent_phrase text,
            target_symbol text,
            PRIMARY KEY (taxonomy_version, catchword_id, target_ordinal)
        )
        """
    )
    op.execute(
        "CREATE INDEX ipc_official_catchwords_target ON ipc_official_catchwords (taxonomy_version, target_symbol)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE ipc_official_catchwords")
    op.execute("DROP TABLE ipc_official_texts")
    op.execute("DROP TABLE ipc_official_scheme_entries")
    op.execute("DROP INDEX ipc_records_search")
    op.execute("ALTER TABLE ipc_records DROP COLUMN search_vector")
    op.execute(
        """
        ALTER TABLE ipc_records ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(original_title, '')), 'B') ||
            setweight(to_tsvector('english', coalesce(context_heading, '')), 'B') ||
            setweight(to_tsvector('english', semantic_text), 'C')
        ) STORED
        """
    )
    op.execute("CREATE INDEX ipc_records_search ON ipc_records USING gin (search_vector)")
    op.execute("ALTER TABLE ipc_records DROP COLUMN official_keywords")
    op.execute("ALTER TABLE ipc_records DROP COLUMN official_title")
    op.execute("ALTER TABLE taxonomy_releases DROP COLUMN official_source_hashes")
