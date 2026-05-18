"""Add fx_macro_features table for global FX-relevant macro inputs.

One row per calendar date, shared across all FX/metal tickers on that
date. Drives the three FX-only feature columns (`dxy_ret_5d`, `vix_close`,
`yield_spread_10_2`) the model uses when predicting on FX bars; equity
samples zero-pad these columns instead of joining.

Columns:
  date              — calendar date (UTC), primary key
  dxy_close         — Dollar Index daily close (yfinance: DX-Y.NYB)
  dxy_ret_5d        — 5-day return on DXY (computed when populated)
  vix_close         — VIX daily close (yfinance: ^VIX)
  yield_spread_10_2 — 10Y minus 2Y Treasury yield % (FRED: DGS10 - DGS2)
  updated_at        — last refresh timestamp

Revision ID: 0013
Revises: 0012
Create Date: 2026-05-14 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa


revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fx_macro_features",
        sa.Column("date",              sa.Date(),     primary_key=True),
        sa.Column("dxy_close",         sa.Float(),    nullable=True),
        sa.Column("dxy_ret_5d",        sa.Float(),    nullable=True),
        sa.Column("vix_close",         sa.Float(),    nullable=True),
        sa.Column("yield_spread_10_2", sa.Float(),    nullable=True),
        sa.Column("updated_at",        sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("fx_macro_features")
