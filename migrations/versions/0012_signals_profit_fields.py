"""Add profit-engineering fields to the signals table.

These columns persist the new outputs from predict_ticker that drive
Kelly-based sizing and trade gating in the execution layer:

  uncertainty       — MC-dropout std at the predicted class
  predicted_rr      — predicted take_profit / stop_loss ratio
  expected_value    — confidence·TP − (1−c)·SL (the headline EV number)
  kelly_full        — raw Kelly fraction (for reference / dashboards)
  kelly_fraction    — quarter-Kelly position size in [0, 1] (used by executor)
  reject_reasons    — JSONB list of filter gates that downgraded action to HOLD
  atr_stop_loss     — ATR-based SL alternative (1.5× ATR)
  atr_take_profit   — ATR-based TP alternative
  fomc_days, cpi_days, nfp_days, earnings_days — event proximity snapshot

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-13 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("uncertainty",     sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("predicted_rr",    sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("expected_value",  sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("kelly_full",      sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("kelly_fraction",  sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("reject_reasons",  postgresql.JSONB(), nullable=True))
    op.add_column("signals", sa.Column("atr_stop_loss",   sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("atr_take_profit", sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("fomc_days",       sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("cpi_days",        sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("nfp_days",        sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("earnings_days",   sa.Float(), nullable=True))


def downgrade() -> None:
    for col in (
        "earnings_days", "nfp_days", "cpi_days", "fomc_days",
        "atr_take_profit", "atr_stop_loss",
        "reject_reasons", "kelly_fraction", "kelly_full",
        "expected_value", "predicted_rr", "uncertainty",
    ):
        op.drop_column("signals", col)
