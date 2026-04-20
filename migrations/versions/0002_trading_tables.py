"""Add trading_config and trade_executions tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-20 00:00:00.000000 UTC
"""

from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, TEXT

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── trading_config ────────────────────────────────────────────────────────
    # Per-user MT5 settings and risk parameters. One row per user.
    op.create_table(
        "trading_config",
        sa.Column("id",                  sa.Integer(),  primary_key=True),
        sa.Column("user_id",             sa.Integer(),
                  sa.ForeignKey("users.id", ondelete="CASCADE"),
                  nullable=False, unique=True),

        # Master switches
        sa.Column("enabled",             sa.Boolean(),  server_default="FALSE"),
        sa.Column("auto_trade",          sa.Boolean(),  server_default="FALSE"),

        # Risk parameters
        sa.Column("risk_per_trade_pct",  sa.Float(),    server_default="1.0"),
        sa.Column("max_open_positions",  sa.Integer(),  server_default="5"),
        sa.Column("max_daily_loss_pct",  sa.Float(),    server_default="3.0"),
        sa.Column("min_confidence",      sa.Float(),    server_default="0.70"),
        sa.Column("allowed_actions",     ARRAY(TEXT()), server_default=sa.text("'{BUY,SELL}'")),

        # MT5 connection (stored in DB so the scheduler can reconnect after restart)
        sa.Column("mt5_account",         sa.BigInteger()),
        sa.Column("mt5_server",          sa.String(100)),
        sa.Column("mt5_path",            sa.String(500)),   # path to terminal64.exe
        sa.Column("symbol_suffix",       sa.String(10),  server_default=sa.text("''")),

        sa.Column("created_at",          sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at",          sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── trade_executions ──────────────────────────────────────────────────────
    # Immutable audit log — one row per order sent to MT5.
    op.create_table(
        "trade_executions",
        sa.Column("id",               sa.Integer(),  primary_key=True),
        sa.Column("signal_id",        sa.Integer(),
                  sa.ForeignKey("signals.id", ondelete="SET NULL")),
        sa.Column("user_id",          sa.Integer(),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("symbol",           sa.String(20),  nullable=False),
        sa.Column("mt5_ticket",       sa.BigInteger()),   # MT5 position/order ticket
        sa.Column("order_type",       sa.String(10),  nullable=False),   # BUY | SELL
        sa.Column("volume",           sa.Float()),
        sa.Column("requested_price",  sa.Float()),
        sa.Column("fill_price",       sa.Float()),
        sa.Column("stop_loss",        sa.Float()),
        sa.Column("take_profit",      sa.Float()),
        sa.Column("status",           sa.String(20),  server_default="pending"),
        # filled | closed | cancelled | error
        sa.Column("mt5_retcode",      sa.Integer()),
        sa.Column("mt5_comment",      sa.Text()),
        sa.Column("pnl",              sa.Float()),
        sa.Column("closed_at",        sa.DateTime(timezone=True)),
        sa.Column("created_at",       sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_executions_user",        "trade_executions", ["user_id"])
    op.create_index("idx_executions_signal",      "trade_executions", ["signal_id"])
    op.create_index("idx_executions_ticket",      "trade_executions", ["mt5_ticket"])
    op.create_index("idx_executions_status_date", "trade_executions",
                    ["user_id", "status", sa.text("created_at DESC")])


def downgrade() -> None:
    op.drop_table("trade_executions")
    op.drop_table("trading_config")
