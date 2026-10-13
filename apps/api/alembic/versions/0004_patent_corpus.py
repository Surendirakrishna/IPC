"""Labelled patent corpus for the PaECTER neighbour vote (ADR-0005).

A corpus is one load of labelled patents from one source, embedded by one encoder revision. Its
documents keep the raw source labels and the labels resolved against the catalogue (main label
first); its vectors are halfvec(1024). At most one validated corpus serves at a time.

New tables only: existing tables are not altered or referenced. The HNSW index on
patent_embeddings is built by service code (TASK-0007), not here.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

CORPUS_STATES = ("loading", "embedded", "validated", "failed")
LABEL_SCHEMES = ("compact", "structured")


def upgrade() -> None:
    states = ", ".join(f"'{s}'" for s in CORPUS_STATES)
    schemes = ", ".join(f"'{s}'" for s in LABEL_SCHEMES)
    op.execute(
        f"""
        CREATE TABLE patent_corpora (
            corpus_id                text PRIMARY KEY,
            source                   text NOT NULL,
            description              text,
            label_scheme             text NOT NULL CHECK (label_scheme IN ({schemes})),
            label_ipc_edition        text,
            labels_taxonomy_version  text,
            document_count           integer NOT NULL DEFAULT 0 CHECK (document_count >= 0),
            encoder_model_id         text NOT NULL,
            encoder_model_revision   text NOT NULL,
            input_hash               text,
            state                    text NOT NULL DEFAULT 'loading' CHECK (state IN ({states})),
            is_serving               boolean NOT NULL DEFAULT false,
            created_at               timestamptz NOT NULL DEFAULT now(),
            updated_at               timestamptz NOT NULL DEFAULT now(),
            served_at                timestamptz,
            CONSTRAINT patent_corpora_serving_requires_validated CHECK (NOT is_serving OR state = 'validated')
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX patent_corpora_one_serving ON patent_corpora (is_serving) WHERE is_serving")

    op.execute(
        """
        CREATE TABLE patent_documents (
            corpus_id         text NOT NULL REFERENCES patent_corpora (corpus_id) ON DELETE CASCADE,
            document_id       text NOT NULL,
            title             text,
            abstract          text NOT NULL,
            filing_date       date,
            publication_date  date,
            raw_labels        jsonb NOT NULL CHECK (jsonb_typeof(raw_labels) = 'array'),
            resolved_labels   jsonb NOT NULL CHECK (jsonb_typeof(resolved_labels) = 'array'),
            created_at        timestamptz NOT NULL DEFAULT now(),
            updated_at        timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (corpus_id, document_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE patent_embeddings (
            corpus_id    text NOT NULL,
            document_id  text NOT NULL,
            embedding    halfvec(1024) NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (corpus_id, document_id),
            FOREIGN KEY (corpus_id, document_id)
                REFERENCES patent_documents (corpus_id, document_id) ON DELETE CASCADE
        )
        """
    )
    # A halfvec(1024) value is 2,056 bytes, just over the ~2 KB TOAST threshold: keep vectors inline so
    # HNSW builds and exact scans do not fetch every row from TOAST.
    op.execute("ALTER TABLE patent_embeddings ALTER COLUMN embedding SET STORAGE PLAIN")


def downgrade() -> None:
    op.execute("DROP TABLE patent_embeddings")
    op.execute("DROP TABLE patent_documents")
    op.execute("DROP TABLE patent_corpora")
