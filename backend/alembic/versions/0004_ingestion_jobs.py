"""ingestion jobs: queue for background document extraction

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workspace", sa.String(100), nullable=False),
        sa.Column("filename", sa.String(300), nullable=False),
        sa.Column("content_type", sa.String(200), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String(100)), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("chars", sa.Integer(), nullable=True),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("memories.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ingestion_jobs_workspace", "ingestion_jobs", ["workspace"])
    op.create_index(
        "ix_ingestion_jobs_pending",
        "ingestion_jobs",
        ["created_at"],
        postgresql_where=sa.text("status IN ('queued', 'processing')"),
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
