"""Add eps_surprise_pct column to indicators table.

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-06 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("indicators", sa.Column("eps_surprise_pct", sa.Float()))


def downgrade() -> None:
    op.drop_column("indicators", "eps_surprise_pct")
