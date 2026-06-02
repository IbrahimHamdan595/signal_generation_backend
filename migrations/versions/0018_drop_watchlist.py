"""Drop the users.watchlist column.

The watchlist was a UI-only feature with no impact on signal generation
or trading (the trading scheduler reads from ohlcv_data, not from any
user-scoped watchlist). The column added bookkeeping for no benefit, so
remove it along with its endpoints and frontend pages.

Revision ID: 0018
Revises: 0017
Create Date: 2026-06-02 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op


revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS watchlist;")


def downgrade() -> None:
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS watchlist text[] DEFAULT '{}';")
