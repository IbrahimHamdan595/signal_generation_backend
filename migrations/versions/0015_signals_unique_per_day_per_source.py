"""Widen the per-day uniqueness on signals to include the source column.

Before this migration the unique key was (ticker, interval, date) — fine
when there was only one signal generator (ml_model). With the three-way
pipeline (ml_equities, ml_fx, rule_donchian) we need each source's
signal for a given ticker+day to coexist so the comparison tab can
aggregate by source. New key: (ticker, interval, date, source).

Also normalises any pre-existing source='ml_model' rows to 'ml_equities'
so the new naming is consistent end-to-end.

Revision ID: 0015
Revises: 0014
Create Date: 2026-05-18 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op


revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Rename legacy source value so new code paths see a consistent name
    op.execute("UPDATE signals SET source = 'ml_equities' WHERE source = 'ml_model'")

    # 2. Drop the old index that only covered (ticker, interval, date)
    op.execute("DROP INDEX IF EXISTS uq_signals_ticker_interval_day")

    # 3. Add the new four-column unique index. COALESCE so legacy rows with
    #    NULL source don't collide with new ones — they all bucket together
    #    under the empty string.
    op.execute("""
        CREATE UNIQUE INDEX uq_signals_ticker_interval_day_source
        ON signals (
            ticker,
            interval,
            DATE(created_at AT TIME ZONE 'UTC'),
            COALESCE(source, '')
        )
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_signals_ticker_interval_day_source")
    op.execute("""
        CREATE UNIQUE INDEX uq_signals_ticker_interval_day
        ON signals (ticker, interval, DATE(created_at AT TIME ZONE 'UTC'))
    """)
    # Don't reverse the source rename — it's idempotent and the new value
    # is the canonical going-forward name.
