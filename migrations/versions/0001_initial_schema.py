"""Initial schema — all tables and indexes.

Revision ID: 0001
Revises:
Create Date: 2026-04-18 00:00:00.000000 UTC

This migration represents the full schema that was previously created
inline inside database.py on every server start.  On a brand-new database
run `alembic upgrade head`.  On an existing database that already has these
tables, run `alembic stamp 0001` to mark this migration as applied without
re-running the DDL.
"""

from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",            sa.Integer(),     primary_key=True),
        sa.Column("email",         sa.String(255),   nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255),   nullable=False),
        sa.Column("full_name",     sa.String(255)),
        sa.Column("is_active",     sa.Boolean(),     server_default="TRUE"),
        sa.Column("is_admin",      sa.Boolean(),     server_default="FALSE"),
        sa.Column("watchlist",     sa.ARRAY(sa.Text()), server_default="{}"),
        sa.Column("created_at",    sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at",    sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_unique_constraint("idx_users_email", "users", ["email"])

    # ── ohlcv_data ───────────────────────────────────────────────────────────
    op.create_table(
        "ohlcv_data",
        sa.Column("id",          sa.Integer(),  primary_key=True),
        sa.Column("ticker",      sa.String(10), nullable=False),
        sa.Column("interval",    sa.String(5),  nullable=False),
        sa.Column("timestamp",   sa.DateTime(timezone=True), nullable=False),
        sa.Column("open",        sa.Float(),    nullable=False),
        sa.Column("high",        sa.Float(),    nullable=False),
        sa.Column("low",         sa.Float(),    nullable=False),
        sa.Column("close",       sa.Float(),    nullable=False),
        sa.Column("volume",      sa.Float(),    nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at",  sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("ticker", "interval", "timestamp", name="uq_ohlcv_ticker_interval_ts"),
    )
    op.create_index("idx_ohlcv_ticker_interval", "ohlcv_data", ["ticker", "interval"])

    # ── indicators ───────────────────────────────────────────────────────────
    _float = sa.Float
    op.create_table(
        "indicators",
        sa.Column("id",                    sa.Integer(), primary_key=True),
        sa.Column("ticker",                sa.String(10), nullable=False),
        sa.Column("interval",              sa.String(5),  nullable=False),
        sa.Column("timestamp",             sa.DateTime(timezone=True), nullable=False),
        sa.Column("sma_20",                _float()),
        sa.Column("sma_50",                _float()),
        sa.Column("ema_12",                _float()),
        sa.Column("ema_26",                _float()),
        sa.Column("rsi_14",                _float()),
        sa.Column("macd_line",             _float()),
        sa.Column("macd_signal",           _float()),
        sa.Column("macd_histogram",        _float()),
        sa.Column("atr_14",                _float()),
        sa.Column("bb_upper",              _float()),
        sa.Column("bb_middle",             _float()),
        sa.Column("bb_lower",              _float()),
        sa.Column("bb_bandwidth",          _float()),
        sa.Column("obv",                   _float()),
        sa.Column("mfi_14",                _float()),
        sa.Column("volume_roc",            _float()),
        sa.Column("stoch_k",               _float()),
        sa.Column("stoch_d",               _float()),
        sa.Column("day_of_week",           _float()),
        sa.Column("day_of_month",          _float()),
        sa.Column("month",                 _float()),
        sa.Column("is_trading_day",        _float()),
        sa.Column("adx",                   _float()),
        sa.Column("plus_di",               _float()),
        sa.Column("minus_di",              _float()),
        sa.Column("pivot",                 _float()),
        sa.Column("resistance_1",          _float()),
        sa.Column("support_1",             _float()),
        sa.Column("resistance_2",          _float()),
        sa.Column("support_2",             _float()),
        sa.Column("price_sma20_dist",      _float()),
        sa.Column("price_sma50_dist",      _float()),
        sa.Column("high_vol_regime",       _float()),
        sa.Column("above_sma50",           _float()),
        sa.Column("above_sma200",          _float()),
        sa.Column("normalized_volatility", _float()),
        sa.Column("bb_position",           _float()),
        sa.Column("roc_5",                 _float()),
        sa.Column("roc_10",                _float()),
        sa.Column("higher_high",           _float()),
        sa.Column("lower_low",             _float()),
        sa.Column("price_change_pct",      _float()),
        sa.Column("volume_above_avg",      _float()),
        sa.Column("vix_level",             _float()),
        sa.Column("vix_change",            _float()),
        sa.Column("earnings_days",         _float()),
        sa.Column("social_sentiment",      _float()),
        sa.Column("options_put_call_ratio", _float()),
        sa.Column("computed_at",  sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at",   sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("ticker", "interval", "timestamp", name="uq_indicators_ticker_interval_ts"),
    )
    op.create_index("idx_indicators_ticker_interval", "indicators", ["ticker", "interval"])

    # ── signals ───────────────────────────────────────────────────────────────
    op.create_table(
        "signals",
        sa.Column("id",               sa.Integer(),  primary_key=True),
        sa.Column("ticker",           sa.String(10), nullable=False),
        sa.Column("interval",         sa.String(5),  nullable=False),
        sa.Column("action",           sa.String(10), nullable=False),
        sa.Column("confidence",       sa.Float(),    nullable=False),
        sa.Column("entry_price",      sa.Float()),
        sa.Column("stop_loss",        sa.Float()),
        sa.Column("take_profit",      sa.Float()),
        sa.Column("net_profit",       sa.Float()),
        sa.Column("bars_to_entry",    sa.Float()),
        sa.Column("entry_time",       sa.DateTime(timezone=True)),
        sa.Column("entry_time_label", sa.String(20)),
        sa.Column("prob_buy",         sa.Float()),
        sa.Column("prob_sell",        sa.Float()),
        sa.Column("prob_hold",        sa.Float()),
        sa.Column("source",           sa.String(50), server_default="ml_model"),
        sa.Column("user_id",          sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("created_at",       sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_signals_ticker_created",  "signals", ["ticker",     sa.text("created_at DESC")])
    op.create_index("idx_signals_action_created",  "signals", ["action",     sa.text("created_at DESC")])
    op.create_index("idx_signals_confidence",      "signals", [sa.text("confidence DESC")])
    op.create_index("idx_signals_user",            "signals", ["user_id"])

    # ── sentiment_articles ────────────────────────────────────────────────────
    op.create_table(
        "sentiment_articles",
        sa.Column("id",              sa.Integer(),   primary_key=True),
        sa.Column("ticker",          sa.String(10),  nullable=False),
        sa.Column("title",           sa.Text()),
        sa.Column("description",     sa.Text()),
        sa.Column("url",             sa.String(500), unique=True),
        sa.Column("source",          sa.String(255)),
        sa.Column("published_at",    sa.DateTime(timezone=True)),
        sa.Column("sentiment_label", sa.String(20)),
        sa.Column("positive_score",  sa.Float()),
        sa.Column("negative_score",  sa.Float()),
        sa.Column("neutral_score",   sa.Float()),
        sa.Column("compound_score",  sa.Float()),
        sa.Column("ingested_at",     sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at",      sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_sentiment_articles_ticker", "sentiment_articles", ["ticker"])

    # ── sentiment_snapshots ───────────────────────────────────────────────────
    op.create_table(
        "sentiment_snapshots",
        sa.Column("id",                 sa.Integer(),  primary_key=True),
        sa.Column("ticker",             sa.String(10), nullable=False, unique=True),
        sa.Column("avg_positive",       sa.Float()),
        sa.Column("avg_negative",       sa.Float()),
        sa.Column("avg_neutral",        sa.Float()),
        sa.Column("avg_compound",       sa.Float()),
        sa.Column("dominant_sentiment", sa.String(20)),
        sa.Column("article_count",      sa.Integer()),
        sa.Column("computed_at",        sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_sentiment_snapshots_ticker", "sentiment_snapshots", ["ticker"])

    # ── signal_outcomes ───────────────────────────────────────────────────────
    op.create_table(
        "signal_outcomes",
        sa.Column("id",            sa.Integer(), primary_key=True),
        sa.Column("signal_id",     sa.Integer(), sa.ForeignKey("signals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker",        sa.String(10), nullable=False),
        sa.Column("action",        sa.String(10), nullable=False),
        sa.Column("entry_price",   sa.Float()),
        sa.Column("stop_loss",     sa.Float()),
        sa.Column("take_profit",   sa.Float()),
        sa.Column("outcome",       sa.String(10)),
        sa.Column("actual_return", sa.Float()),
        sa.Column("bars_held",     sa.Integer()),
        sa.Column("exit_price",    sa.Float()),
        sa.Column("exit_time",     sa.DateTime(timezone=True)),
        sa.Column("checked_at",    sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("signal_id", name="uq_signal_outcomes_signal_id"),
    )
    op.create_index("idx_signal_outcomes_ticker", "signal_outcomes", ["ticker"])
    op.create_index("idx_signal_outcomes_signal", "signal_outcomes", ["signal_id"])

    # ── alerts ────────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id",         sa.Integer(),  primary_key=True),
        sa.Column("ticker",     sa.String(10), nullable=False),
        sa.Column("action",     sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(),    nullable=False),
        sa.Column("signal_id",  sa.Integer(), sa.ForeignKey("signals.id", ondelete="SET NULL")),
        sa.Column("message",    sa.Text()),
        sa.Column("is_read",    sa.Boolean(),  server_default="FALSE"),
        sa.Column("user_id",    sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_alerts_created", "alerts", [sa.text("created_at DESC")])
    op.create_index("idx_alerts_read",    "alerts", ["is_read"])
    op.create_index("idx_alerts_user",    "alerts", ["user_id"])

    # ── daily_sentiment ───────────────────────────────────────────────────────
    op.create_table(
        "daily_sentiment",
        sa.Column("id",            sa.Integer(),  primary_key=True),
        sa.Column("ticker",        sa.String(10), nullable=False),
        sa.Column("date",          sa.Date(),     nullable=False),
        sa.Column("avg_positive",  sa.Float(),    server_default="0"),
        sa.Column("avg_negative",  sa.Float(),    server_default="0"),
        sa.Column("avg_neutral",   sa.Float(),    server_default="1"),
        sa.Column("avg_compound",  sa.Float(),    server_default="0"),
        sa.Column("article_count", sa.Integer(),  server_default="0"),
        sa.Column("computed_at",   sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("ticker", "date", name="uq_daily_sentiment_ticker_date"),
    )
    op.create_index("idx_daily_sentiment_ticker_date", "daily_sentiment", ["ticker", "date"])

    # ── jobs ──────────────────────────────────────────────────────────────────
    op.create_table(
        "jobs",
        sa.Column("id",          sa.Integer(),   primary_key=True),
        sa.Column("job_type",    sa.String(50),  nullable=False),
        sa.Column("status",      sa.String(20),  server_default="pending"),
        sa.Column("progress",    JSONB(),         server_default="{}"),
        sa.Column("error",       sa.Text()),
        sa.Column("started_at",  sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("created_at",  sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_jobs_type_status", "jobs", ["job_type", "status"])

    # ── positions ─────────────────────────────────────────────────────────────
    op.create_table(
        "positions",
        sa.Column("id",           sa.Integer(), primary_key=True),
        sa.Column("user_id",      sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker",       sa.String(10), nullable=False),
        sa.Column("quantity",     sa.Float(),   nullable=False, server_default="0"),
        sa.Column("avg_cost",     sa.Float(),   nullable=False, server_default="0"),
        sa.Column("opened_at",    sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("closed_at",    sa.DateTime(timezone=True)),
        sa.Column("realized_pnl", sa.Float(),   server_default="0"),
        sa.Column("is_open",      sa.Boolean(), server_default="TRUE"),
    )
    op.create_index("idx_positions_user", "positions", ["user_id", "is_open"])

    # ── position_trades ───────────────────────────────────────────────────────
    op.create_table(
        "position_trades",
        sa.Column("id",          sa.Integer(), primary_key=True),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("positions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("signal_id",   sa.Integer(), sa.ForeignKey("signals.id",   ondelete="SET NULL")),
        sa.Column("action",      sa.String(10), nullable=False),
        sa.Column("quantity",    sa.Float(),   nullable=False),
        sa.Column("price",       sa.Float(),   nullable=False),
        sa.Column("pnl",         sa.Float()),
        sa.Column("executed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── price_alert_rules ─────────────────────────────────────────────────────
    op.create_table(
        "price_alert_rules",
        sa.Column("id",           sa.Integer(),  primary_key=True),
        sa.Column("user_id",      sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker",       sa.String(10), nullable=False),
        sa.Column("condition",    sa.String(10), nullable=False),
        sa.Column("target_price", sa.Float(),    nullable=False),
        sa.Column("is_active",    sa.Boolean(),  server_default="TRUE"),
        sa.Column("triggered_at", sa.DateTime(timezone=True)),
        sa.Column("created_at",   sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_price_alerts_user", "price_alert_rules", ["user_id", "is_active"])


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("price_alert_rules")
    op.drop_table("position_trades")
    op.drop_table("positions")
    op.drop_table("jobs")
    op.drop_table("daily_sentiment")
    op.drop_table("alerts")
    op.drop_table("signal_outcomes")
    op.drop_table("sentiment_snapshots")
    op.drop_table("sentiment_articles")
    op.drop_table("signals")
    op.drop_table("indicators")
    op.drop_table("ohlcv_data")
    op.drop_table("users")
