"""Indexes for the dashboard activity feed.

The activity feed UNIONs signals and trade_executions filtered by recency
and status. Two indexes ensure the 24h window query stays fast as the
tables grow:

  - trade_executions(status, created_at DESC)  — supports the
    'rejected_commission' / 'filled' filtered scans
  - signals(created_at DESC, source)           — supports the
    per-minute group-by-source aggregation

The new pseudo-status 'rejected_commission' is just a string written by
the Path A commission gate when a trade is blocked before submission. No
schema change required (status is sa.String(20), no CHECK constraint).

Revision ID: 0019
Revises: 0018
Create Date: 2026-06-02 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op


revision: str = "0019"
down_revision: Union[str, None] = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_te_status_created "
        "ON trade_executions(status, created_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_created_source "
        "ON signals(created_at DESC, source);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_te_status_created;")
    op.execute("DROP INDEX IF EXISTS idx_signals_created_source;")
