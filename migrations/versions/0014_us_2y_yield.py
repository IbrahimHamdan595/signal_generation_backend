"""Add us_2y_yield column to fx_macro_features.

Captures the absolute level of the US 2-year Treasury yield — together
with the existing `yield_spread_10_2` (10Y minus 2Y), this gives the FX
model both the level and the slope of the US curve, which is the
strongest single-source FX driver per the macro literature.

Carry/short-rate proxy used by the FX classifier.

Revision ID: 0014
Revises: 0013
Create Date: 2026-05-15 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa


revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fx_macro_features",
        sa.Column("us_2y_yield", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("fx_macro_features", "us_2y_yield")
