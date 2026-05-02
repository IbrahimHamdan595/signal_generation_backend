"""Add unique constraints to sentiment tables for ON CONFLICT clauses.

NOTE: Supabase statement timeout blocks ALTER TABLE from completing via alembic.
Manual execution in Supabase SQL Editor required — see README below.
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # WARNING: Do NOT run via alembic on Supabase — statement timeout will kill it.
    # Execute these statements manually in Supabase SQL Editor instead:
    """
    ALTER TABLE sentiment_articles ADD CONSTRAINT uq_sentiment_articles_url UNIQUE (url);
    ALTER TABLE sentiment_snapshots ADD CONSTRAINT uq_sentiment_snapshots_ticker UNIQUE (ticker);
    ALTER TABLE daily_sentiment ADD CONSTRAINT uq_daily_sentiment_ticker_date UNIQUE (ticker, date);
    """
    pass


def downgrade() -> None:
    op.execute(
        "ALTER TABLE sentiment_articles DROP CONSTRAINT IF EXISTS uq_sentiment_articles_url"
    )
    op.execute(
        "ALTER TABLE sentiment_snapshots DROP CONSTRAINT IF EXISTS uq_sentiment_snapshots_ticker"
    )
    op.execute(
        "ALTER TABLE daily_sentiment DROP CONSTRAINT IF EXISTS uq_daily_sentiment_ticker_date"
    )
