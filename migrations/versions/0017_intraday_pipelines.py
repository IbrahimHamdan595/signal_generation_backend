"""Add intraday (1h) counterparts to the three existing pipelines.

The three existing rows (`ml_equities`, `ml_fx`, `rule_donchian`) run on
daily bars and produce signals that hold for 3-10 days. This migration
adds three additional pipelines (`ml_equities_1h`, `ml_fx_1h`,
`rule_donchian_1h`) running on 1h bars so the forward-test can compare
daily vs intraday horizons side-by-side. The two ML 1h pipelines need
their own trained checkpoints — they start `enabled = FALSE` so the
scheduler doesn't try to run them before the user trains the models.
Donchian 1h doesn't need training but also starts disabled for
consistency with the other two new pipelines.

Revision ID: 0017
Revises: 0016
Create Date: 2026-05-20 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op


revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Three new pipeline_config rows. All start disabled — the ML two need
    # their checkpoints trained first; Donchian is enabled later from the UI.
    # ON CONFLICT skip lets the migration be re-runnable safely.
    op.execute("""
        INSERT INTO pipeline_config (source, enabled, config) VALUES
          ('ml_equities_1h',   FALSE, '{}'::jsonb),
          ('ml_fx_1h',         FALSE, '{}'::jsonb),
          ('rule_donchian_1h', FALSE, '{"window": 20, "k_sl": 1.5, "k_tp": 3.0}'::jsonb)
        ON CONFLICT (source) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM pipeline_config
        WHERE source IN ('ml_equities_1h', 'ml_fx_1h', 'rule_donchian_1h')
    """)
