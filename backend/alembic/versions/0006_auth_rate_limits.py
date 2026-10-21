"""workspaces, API keys and rate-limit buckets

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("slug", sa.String(100), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Register workspaces that already hold data.
    op.execute(
        "INSERT INTO workspaces (slug) SELECT DISTINCT workspace FROM memories "
        "UNION SELECT DISTINCT workspace FROM ingestion_jobs"
    )
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace", sa.String(100), sa.ForeignKey("workspaces.slug", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False, unique=True),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_api_keys_workspace", "api_keys", ["workspace"])
    # Token buckets are disposable counters: UNLOGGED skips the write-ahead log, so the write each request
    # makes is cheap. After a crash the table is emptied, which just refills every bucket.
    op.execute(
        """
        CREATE UNLOGGED TABLE rate_limits (
            bucket text PRIMARY KEY,
            tokens double precision NOT NULL,
            updated_at timestamptz NOT NULL
        )
        """
    )


def downgrade() -> None:
    op.drop_table("rate_limits")
    op.drop_table("api_keys")
    op.drop_table("workspaces")
