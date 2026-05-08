"""Create earnings_history table for per-ticker EPS surprise data.

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-06 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "earnings_history",
        sa.Column("ticker",           sa.String(10), nullable=False),
        sa.Column("report_date",      sa.Date(),     nullable=False),
        sa.Column("eps_actual",       sa.Float()),
        sa.Column("eps_estimate",     sa.Float()),
        sa.Column("eps_surprise_pct", sa.Float()),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("ticker", "report_date", name="pk_earnings_history"),
    )
    op.create_index("idx_earnings_history_ticker_date", "earnings_history", ["ticker", "report_date"])


def downgrade() -> None:
    op.drop_index("idx_earnings_history_ticker_date", table_name="earnings_history")
    op.drop_table("earnings_history")
