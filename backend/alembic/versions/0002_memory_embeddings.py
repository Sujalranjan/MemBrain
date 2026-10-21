"""document-level memory embeddings for the knowledge graph

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("embedding", Vector(384), nullable=True))
    # Backfill existing memories with the mean of their chunk vectors.
    op.execute(
        """
        UPDATE memories AS m
        SET embedding = c.centroid
        FROM (SELECT memory_id, avg(embedding) AS centroid FROM chunks GROUP BY memory_id) AS c
        WHERE m.id = c.memory_id
        """
    )
    op.create_index(
        "ix_memories_embedding_hnsw",
        "memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_memories_embedding_hnsw", table_name="memories")
    op.drop_column("memories", "embedding")
