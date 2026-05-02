"""Add unique constraint: one signal per ticker+interval per UTC day."""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    # Remove existing duplicates — keep the most recent row per group
    op.execute("""
        DELETE FROM signals
        WHERE id NOT IN (
            SELECT DISTINCT ON (ticker, interval, DATE(created_at AT TIME ZONE 'UTC'))
                id
            FROM signals
            ORDER BY ticker, interval, DATE(created_at AT TIME ZONE 'UTC'), created_at DESC
        )
    """)

    op.execute("""
        CREATE UNIQUE INDEX uq_signals_ticker_interval_day
        ON signals (ticker, interval, DATE(created_at AT TIME ZONE 'UTC'))
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS uq_signals_ticker_interval_day")
