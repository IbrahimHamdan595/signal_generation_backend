"""Add v2 predicted fields to signals table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-26 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # entry_offset_pct: % offset from current close, predicted by regression head
    op.add_column("signals", sa.Column("entry_offset_pct", sa.Float(), nullable=True))
    # timing_bucket: 0=bar+1, 1=bar+2, 2=bar+3, 3=HOLD, predicted by timing head
    op.add_column("signals", sa.Column("timing_bucket", sa.SmallInteger(), nullable=True))
    # timing_bucket_label: human-readable label for timing_bucket
    op.add_column("signals", sa.Column("timing_bucket_label", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("signals", "timing_bucket_label")
    op.drop_column("signals", "timing_bucket")
    op.drop_column("signals", "entry_offset_pct")
