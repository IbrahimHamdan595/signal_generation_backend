"""Add fomc_days, cpi_days, nfp_days columns to indicators table.

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-06 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("indicators", sa.Column("fomc_days", sa.Float()))
    op.add_column("indicators", sa.Column("cpi_days",  sa.Float()))
    op.add_column("indicators", sa.Column("nfp_days",  sa.Float()))


def downgrade() -> None:
    op.drop_column("indicators", "nfp_days")
    op.drop_column("indicators", "cpi_days")
    op.drop_column("indicators", "fomc_days")
