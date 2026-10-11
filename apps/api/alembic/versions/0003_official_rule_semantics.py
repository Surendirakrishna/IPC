"""Keep each WIPO note's scope, edition and rule meaning.

A note entry carries the symbol range it applies to (symbol..endSymbol) and the edition that introduced
it. Its wording is stored with a rule reading (precedence, place rule, limiting reference, ...) so the
classifier can apply priority rules without treating every cited symbol as an assignment.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

RULE_KINDS = ("precedence", "place_last", "place_first", "indexing", "definition", "limiting_reference",
              "informative_reference", "scope", "other")


def upgrade() -> None:
    kinds = ", ".join(f"'{k}'" for k in RULE_KINDS)
    op.execute(
        f"""
        ALTER TABLE ipc_official_texts
            ADD COLUMN container_symbol text,
            ADD COLUMN scope_end_symbol text,
            ADD COLUMN scope_symbols text[] NOT NULL DEFAULT '{{}}',
            ADD COLUMN editions text,
            ADD COLUMN rule_kind text NOT NULL DEFAULT 'other' CHECK (rule_kind IN ({kinds})),
            ADD COLUMN rules jsonb NOT NULL DEFAULT '[]'::jsonb
        """
    )
    op.execute("CREATE INDEX ipc_official_texts_scope ON ipc_official_texts USING gin (scope_symbols)")


def downgrade() -> None:
    op.execute("DROP INDEX ipc_official_texts_scope")
    op.execute(
        """
        ALTER TABLE ipc_official_texts
            DROP COLUMN rules,
            DROP COLUMN rule_kind,
            DROP COLUMN editions,
            DROP COLUMN scope_symbols,
            DROP COLUMN scope_end_symbol,
            DROP COLUMN container_symbol
        """
    )
