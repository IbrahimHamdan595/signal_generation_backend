"""Add equity_snapshots table for daily equity guardrail tracking.

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-27 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "equity_snapshots",
        sa.Column("date",            sa.Date(),    primary_key=True),
        sa.Column("open_equity",     sa.Float(),   nullable=False),
        sa.Column("peak_equity",     sa.Float(),   nullable=False),
        sa.Column("current_equity",  sa.Float(),   nullable=False),
        sa.Column("recorded_at",     sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("equity_snapshots")
