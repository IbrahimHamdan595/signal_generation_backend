"""Widen entry_time_label column from VARCHAR(20) to TEXT.

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-28 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "signals",
        "entry_time_label",
        existing_type=sa.String(20),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "signals",
        "entry_time_label",
        existing_type=sa.Text(),
        type_=sa.String(20),
        existing_nullable=True,
    )
