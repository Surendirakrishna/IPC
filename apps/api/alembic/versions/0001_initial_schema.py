"""Initial schema: releases, records, source occurrences, references, embeddings, jobs, calibration.

Revision ID: 0001
Revises:
Create Date: 2026-10-08

The HNSW index is intentionally NOT created here; it is built by `ipc build-index`
after embeddings are loaded (see app/services/retrieval/index.py).
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.execute(
        """
        CREATE TABLE taxonomy_releases (
            taxonomy_version   text PRIMARY KEY,
            source_hashes      jsonb NOT NULL,
            source_files       jsonb NOT NULL,
            source_root        jsonb NOT NULL,
            validation_report  jsonb,
            import_state       text NOT NULL DEFAULT 'pending'
                CHECK (import_state IN ('pending','importing','imported','validated','failed')),
            is_active          boolean NOT NULL DEFAULT false,
            created_at         timestamptz NOT NULL DEFAULT now(),
            updated_at         timestamptz NOT NULL DEFAULT now(),
            activated_at       timestamptz,
            CONSTRAINT active_requires_validated CHECK (NOT is_active OR import_state = 'validated')
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX taxonomy_releases_one_active ON taxonomy_releases (is_active) WHERE is_active"
    )

    op.execute(
        """
        CREATE TABLE ipc_records (
            taxonomy_version     text NOT NULL REFERENCES taxonomy_releases(taxonomy_version),
            code                 text NOT NULL,
            display_code         text NOT NULL,
            record_type          text NOT NULL
                CHECK (record_type IN ('section','class','subclass','maingroup','subgroup')),
            title                text NOT NULL,
            title_raw            text,
            semantic_text        text NOT NULL,
            section_code         text NOT NULL,
            section_title        text NOT NULL,
            class_code           text,
            class_title          text,
            subclass_code        text,
            subclass_title       text,
            maingroup_code       text,
            maingroup_title      text,
            parent_code          text,
            ancestor_codes       text[] NOT NULL DEFAULT '{}',
            child_codes          text[] NOT NULL DEFAULT '{}',
            depth                smallint NOT NULL,
            aliases              jsonb NOT NULL DEFAULT '[]',
            heading_text         text[] NOT NULL DEFAULT '{}',
            context_heading      text,
            titles               text[],
            canonical_title      text,
            is_duplicate_symbol  boolean,
            original_title       text,
            original_title_source jsonb,
            search_vector        tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(original_title, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(context_heading, '')), 'B') ||
                setweight(to_tsvector('english', semantic_text), 'C')
            ) STORED,
            source_line          integer NOT NULL,
            source_payload       jsonb NOT NULL,
            semantic_text_sha256 text NOT NULL,
            PRIMARY KEY (taxonomy_version, code),
            FOREIGN KEY (taxonomy_version, parent_code)
                REFERENCES ipc_records (taxonomy_version, code) DEFERRABLE INITIALLY DEFERRED
        )
        """
    )
    op.execute("CREATE INDEX ipc_records_display_code ON ipc_records (taxonomy_version, display_code)")
    op.execute("CREATE INDEX ipc_records_parent ON ipc_records (taxonomy_version, parent_code)")
    op.execute("CREATE INDEX ipc_records_type ON ipc_records (taxonomy_version, record_type)")
    op.execute("CREATE INDEX ipc_records_subclass ON ipc_records (taxonomy_version, subclass_code)")
    op.execute("CREATE INDEX ipc_records_maingroup ON ipc_records (taxonomy_version, maingroup_code)")
    op.execute("CREATE INDEX ipc_records_search ON ipc_records USING gin (search_vector)")
    op.execute(
        "CREATE INDEX ipc_records_display_trgm ON ipc_records USING gin (display_code gin_trgm_ops)"
    )

    op.execute(
        """
        CREATE TABLE ipc_source_occurrences (
            taxonomy_version     text NOT NULL REFERENCES taxonomy_releases(taxonomy_version),
            occurrence_id        integer NOT NULL,
            code                 text NOT NULL,
            node_type            text,
            source_path          text[] NOT NULL,
            original_title       text NOT NULL,
            direct_child_symbols text[] NOT NULL DEFAULT '{}',
            parent_symbol        text,
            sibling_index        integer NOT NULL,
            context              jsonb NOT NULL DEFAULT '{}',
            PRIMARY KEY (taxonomy_version, occurrence_id)
        )
        """
    )
    op.execute("CREATE INDEX ipc_source_occurrences_code ON ipc_source_occurrences (taxonomy_version, code)")

    op.execute(
        """
        CREATE TABLE ipc_reference_edges (
            taxonomy_version  text NOT NULL REFERENCES taxonomy_releases(taxonomy_version),
            source_code       text NOT NULL,
            target_code       text NOT NULL,
            kind              text NOT NULL CHECK (kind IN ('cross_reference','precedence_reference')),
            original_wording  text,
            target_exists     boolean NOT NULL,
            PRIMARY KEY (taxonomy_version, source_code, target_code, kind),
            FOREIGN KEY (taxonomy_version, source_code) REFERENCES ipc_records (taxonomy_version, code)
        )
        """
    )
    op.execute("CREATE INDEX ipc_reference_edges_target ON ipc_reference_edges (taxonomy_version, target_code)")

    op.execute(
        """
        CREATE TABLE embedding_runs (
            run_id             text PRIMARY KEY,
            taxonomy_version   text NOT NULL REFERENCES taxonomy_releases(taxonomy_version),
            provider           text NOT NULL,
            model_id           text NOT NULL,
            model_revision     text NOT NULL,
            dimension          integer NOT NULL,
            normalized         boolean NOT NULL,
            storage_table      text NOT NULL,
            encoding_policy    jsonb NOT NULL,
            input_hash         text NOT NULL,
            expected_count     integer NOT NULL,
            completed_count    integer NOT NULL DEFAULT 0,
            state              text NOT NULL DEFAULT 'pending'
                CHECK (state IN ('pending','running','complete','validated','failed','superseded')),
            is_serving         boolean NOT NULL DEFAULT false,
            errors             jsonb NOT NULL DEFAULT '[]',
            created_at         timestamptz NOT NULL DEFAULT now(),
            updated_at         timestamptz NOT NULL DEFAULT now(),
            completed_at       timestamptz,
            CONSTRAINT serving_requires_validated CHECK (NOT is_serving OR state = 'validated')
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX embedding_runs_one_serving ON embedding_runs (taxonomy_version, storage_table)"
        " WHERE is_serving"
    )

    for table, dim in (("ipc_embeddings", 1024), ("ipc_embeddings_gemma", 768)):
        op.execute(
            f"""
            CREATE TABLE {table} (
                taxonomy_version      text NOT NULL,
                code                  text NOT NULL,
                record_type           text NOT NULL,
                subclass_code         text,
                embedding             vector({dim}) NOT NULL,
                embedding_run_id      text NOT NULL REFERENCES embedding_runs(run_id),
                semantic_text_sha256  text NOT NULL,
                created_at            timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (taxonomy_version, code),
                FOREIGN KEY (taxonomy_version, code) REFERENCES ipc_records (taxonomy_version, code)
            )
            """
        )
        op.execute(f"CREATE INDEX {table}_run ON {table} (embedding_run_id)")

    op.execute(
        """
        CREATE TABLE ingestion_jobs (
            job_id           text PRIMARY KEY,
            kind             text NOT NULL,
            taxonomy_version text,
            state            text NOT NULL DEFAULT 'pending'
                CHECK (state IN ('pending','running','succeeded','failed','cancelled')),
            progress         jsonb NOT NULL DEFAULT '{}',
            last_checkpoint  jsonb,
            errors           jsonb NOT NULL DEFAULT '[]',
            started_at       timestamptz NOT NULL DEFAULT now(),
            updated_at       timestamptz NOT NULL DEFAULT now(),
            finished_at      timestamptz
        )
        """
    )

    op.execute(
        """
        CREATE TABLE confidence_calibration_runs (
            calibrator_id      text PRIMARY KEY,
            method             text NOT NULL,
            parameters         jsonb NOT NULL,
            artifact_sha256    text NOT NULL,
            compatibility      jsonb NOT NULL,
            label_provenance   jsonb NOT NULL,
            splits             jsonb NOT NULL,
            metrics            jsonb NOT NULL,
            thresholds         jsonb NOT NULL DEFAULT '{}',
            status             text NOT NULL DEFAULT 'fitted'
                CHECK (status IN ('fitted','evaluated','rejected','active','retired')),
            is_active          boolean NOT NULL DEFAULT false,
            created_at         timestamptz NOT NULL DEFAULT now(),
            activated_at       timestamptz
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX confidence_calibration_one_active ON confidence_calibration_runs (is_active)"
        " WHERE is_active"
    )

    # Optional audit storage. Rows are written only when STORE_INPUT_DOCUMENTS/STORE_ABSTRACTS
    # is explicitly enabled; nothing is written by default.
    op.execute(
        """
        CREATE TABLE classification_audit (
            request_id       text PRIMARY KEY,
            created_at       timestamptz NOT NULL DEFAULT now(),
            taxonomy_version text NOT NULL,
            input_documents  jsonb NOT NULL,
            response         jsonb NOT NULL
        )
        """
    )


def downgrade() -> None:
    for table in (
        "classification_audit",
        "confidence_calibration_runs",
        "ingestion_jobs",
        "ipc_embeddings_gemma",
        "ipc_embeddings",
        "embedding_runs",
        "ipc_reference_edges",
        "ipc_source_occurrences",
        "ipc_records",
        "taxonomy_releases",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table}")
