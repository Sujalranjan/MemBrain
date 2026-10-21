"""BM25 keyword ranking: per-memory term postings and document lengths

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_terms",
        sa.Column(
            "memory_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("memories.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("lexeme", sa.Text(), primary_key=True),
        sa.Column("workspace", sa.String(100), nullable=False),
        sa.Column("tf", sa.Integer(), nullable=False),
    )
    op.create_index("ix_memory_terms_workspace_lexeme", "memory_terms", ["workspace", "lexeme"])
    op.add_column("memories", sa.Column("term_count", sa.Integer(), server_default="0", nullable=False))

    # Backfill existing memories (same rule as app.services.memories.INDEX_TERMS_SQL).
    op.execute(
        """
        INSERT INTO memory_terms (memory_id, lexeme, workspace, tf)
        SELECT m.id, t.lexeme, m.workspace,
               cardinality(t.positions) + (SELECT count(*) FROM unnest(t.weights) AS w WHERE w = 'A')
        FROM memories AS m, unnest(m.search_tsv) AS t
        """
    )
    op.execute(
        """
        UPDATE memories AS m SET term_count = s.total
        FROM (SELECT memory_id, sum(tf) AS total FROM memory_terms GROUP BY memory_id) AS s
        WHERE m.id = s.memory_id
        """
    )


def downgrade() -> None:
    op.drop_column("memories", "term_count")
    op.drop_table("memory_terms")
