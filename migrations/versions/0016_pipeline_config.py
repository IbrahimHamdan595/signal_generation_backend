"""Per-strategy operational state for the three-pipeline system.

One row per signal source. The pipeline page reads from this table to show
each strategy's status (enabled/last-run/last-error), and the scheduler
checks `enabled` before running each pipeline so users can disable a
misbehaving strategy without touching code.

Columns:
  source             — PRIMARY KEY, matches signals.source values
  enabled            — TRUE if the scheduler should run this strategy
  last_run_at        — Most recent completion timestamp (success OR failure)
  last_signals_count — How many signals the most recent run produced
  last_error         — Stack-trace head from the most recent failure, NULL on success
  config             — JSONB of strategy-specific tuning knobs (e.g. Donchian
                       window length); free-form so each strategy can read what it needs

Revision ID: 0016
Revises: 0015
Create Date: 2026-05-18 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pipeline_config",
        sa.Column("source",             sa.String(32), primary_key=True),
        sa.Column("enabled",            sa.Boolean(),  nullable=False, server_default=sa.true()),
        sa.Column("last_run_at",        sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_signals_count", sa.Integer(),  nullable=True),
        sa.Column("last_error",         sa.Text(),     nullable=True),
        sa.Column("config",             postgresql.JSONB(), nullable=True),
        sa.Column("updated_at",         sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    # Seed the three canonical sources so the API always has rows to return.
    op.execute("""
        INSERT INTO pipeline_config (source, enabled, config) VALUES
          ('ml_equities',   TRUE, '{}'::jsonb),
          ('ml_fx',         TRUE, '{}'::jsonb),
          ('rule_donchian', TRUE, '{"window": 20, "k_sl": 1.5, "k_tp": 3.0}'::jsonb)
    """)


def downgrade() -> None:
    op.drop_table("pipeline_config")
