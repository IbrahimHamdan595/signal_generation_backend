"""Create macro_events table for FOMC/CPI/NFP event dates.

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-06 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "macro_events",
        sa.Column("event_type", sa.String(20), nullable=False),
        sa.Column("event_date", sa.Date(),      nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("event_type", "event_date", name="pk_macro_events"),
    )
    op.create_index("idx_macro_events_type_date", "macro_events", ["event_type", "event_date"])


def downgrade() -> None:
    op.drop_index("idx_macro_events_type_date", table_name="macro_events")
    op.drop_table("macro_events")
