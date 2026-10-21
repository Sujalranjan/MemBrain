"""forgetting-curve decay: stability and access count replace linear strength

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("stability", sa.Float(), server_default="10.0", nullable=False))
    op.add_column("memories", sa.Column("access_count", sa.Integer(), server_default="0", nullable=False))
    op.drop_column("memories", "strength")


def downgrade() -> None:
    op.add_column("memories", sa.Column("strength", sa.Float(), server_default="1.0", nullable=False))
    op.drop_column("memories", "access_count")
    op.drop_column("memories", "stability")
