import logging
import asyncpg
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

pool: Optional[asyncpg.Pool] = None
client = None


async def connect_db():
    global pool
    pool = await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        statement_cache_size=0,
    )
    await create_tables()
    logger.info(f"✅ Connected to PostgreSQL: {settings.DATABASE_URL}")


async def close_db():
    global pool
    if pool:
        await pool.close()
        logger.info("🔌 PostgreSQL connection closed")


async def get_db():
    return pool


async def create_indexes():
    logger.info("✅ PostgreSQL indexes created")


async def create_tables():
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                is_admin BOOLEAN DEFAULT FALSE,
                watchlist TEXT[] DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                interval VARCHAR(5) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                ingested_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(ticker, interval, timestamp)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                interval VARCHAR(5) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                sma_20 DOUBLE PRECISION,
                sma_50 DOUBLE PRECISION,
                ema_12 DOUBLE PRECISION,
                ema_26 DOUBLE PRECISION,
                rsi_14 DOUBLE PRECISION,
                macd_line DOUBLE PRECISION,
                macd_signal DOUBLE PRECISION,
                macd_histogram DOUBLE PRECISION,
                atr_14 DOUBLE PRECISION,
                bb_upper DOUBLE PRECISION,
                bb_middle DOUBLE PRECISION,
                bb_lower DOUBLE PRECISION,
                bb_bandwidth DOUBLE PRECISION,
                obv DOUBLE PRECISION,
                mfi_14 DOUBLE PRECISION,
                volume_roc DOUBLE PRECISION,
                stoch_k DOUBLE PRECISION,
                stoch_d DOUBLE PRECISION,
                day_of_week DOUBLE PRECISION,
                day_of_month DOUBLE PRECISION,
                month DOUBLE PRECISION,
                is_trading_day DOUBLE PRECISION,
                adx DOUBLE PRECISION,
                plus_di DOUBLE PRECISION,
                minus_di DOUBLE PRECISION,
                pivot DOUBLE PRECISION,
                resistance_1 DOUBLE PRECISION,
                support_1 DOUBLE PRECISION,
                resistance_2 DOUBLE PRECISION,
                support_2 DOUBLE PRECISION,
                price_sma20_dist DOUBLE PRECISION,
                price_sma50_dist DOUBLE PRECISION,
                high_vol_regime DOUBLE PRECISION,
                above_sma50 DOUBLE PRECISION,
                above_sma200 DOUBLE PRECISION,
                normalized_volatility DOUBLE PRECISION,
                bb_position DOUBLE PRECISION,
                roc_5 DOUBLE PRECISION,
                roc_10 DOUBLE PRECISION,
                higher_high DOUBLE PRECISION,
                lower_low DOUBLE PRECISION,
                price_change_pct DOUBLE PRECISION,
                volume_above_avg DOUBLE PRECISION,
                vix_level DOUBLE PRECISION,
                vix_change DOUBLE PRECISION,
                earnings_days DOUBLE PRECISION,
                social_sentiment DOUBLE PRECISION,
                options_put_call_ratio DOUBLE PRECISION,
                computed_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(ticker, interval, timestamp)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                interval VARCHAR(5) NOT NULL,
                action VARCHAR(10) NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                entry_price DOUBLE PRECISION,
                stop_loss DOUBLE PRECISION,
                take_profit DOUBLE PRECISION,
                net_profit DOUBLE PRECISION,
                bars_to_entry DOUBLE PRECISION,
                entry_time TIMESTAMPTZ,
                entry_time_label VARCHAR(20),
                prob_buy DOUBLE PRECISION,
                prob_sell DOUBLE PRECISION,
                prob_hold DOUBLE PRECISION,
                source VARCHAR(50) DEFAULT 'ml_model',
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_articles (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                title TEXT,
                content TEXT,
                url VARCHAR(500) UNIQUE,
                published_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_snapshots (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                avg_positive DOUBLE PRECISION,
                avg_negative DOUBLE PRECISION,
                avg_neutral DOUBLE PRECISION,
                avg_compound DOUBLE PRECISION,
                computed_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id SERIAL PRIMARY KEY,
                signal_id INTEGER NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
                ticker VARCHAR(10) NOT NULL,
                action VARCHAR(10) NOT NULL,
                entry_price DOUBLE PRECISION,
                stop_loss DOUBLE PRECISION,
                take_profit DOUBLE PRECISION,
                outcome VARCHAR(10),
                actual_return DOUBLE PRECISION,
                bars_held INTEGER,
                exit_price DOUBLE PRECISION,
                exit_time TIMESTAMPTZ,
                checked_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(signal_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                action VARCHAR(10) NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                signal_id INTEGER REFERENCES signals(id) ON DELETE SET NULL,
                message TEXT,
                is_read BOOLEAN DEFAULT FALSE,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                job_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                progress JSONB DEFAULT '{}',
                error TEXT,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                ticker VARCHAR(10) NOT NULL,
                quantity DOUBLE PRECISION NOT NULL DEFAULT 0,
                avg_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
                opened_at TIMESTAMPTZ DEFAULT NOW(),
                closed_at TIMESTAMPTZ,
                realized_pnl DOUBLE PRECISION DEFAULT 0,
                is_open BOOLEAN DEFAULT TRUE
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS position_trades (
                id SERIAL PRIMARY KEY,
                position_id INTEGER NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
                signal_id INTEGER REFERENCES signals(id) ON DELETE SET NULL,
                action VARCHAR(10) NOT NULL,
                quantity DOUBLE PRECISION NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                pnl DOUBLE PRECISION,
                executed_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_alert_rules (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                ticker VARCHAR(10) NOT NULL,
                condition VARCHAR(10) NOT NULL,
                target_price DOUBLE PRECISION NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                triggered_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Migrate existing tables to add new columns if they don't exist
        for stmt in [
            "ALTER TABLE signals ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE SET NULL",
            "ALTER TABLE alerts ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE",
        ]:
            try:
                await conn.execute(stmt)
            except Exception:
                pass

        try:
            await conn.execute("""
                SELECT create_hypertable('ohlcv_data', 'timestamp',
                    if_not_exists => TRUE,
                    migrate_data => TRUE)
            """)
            logger.info("✅ ohlcv_data hypertable created")
        except Exception as e:
            logger.warning(f"⚠️ ohlcv_data hypertable: {e}")

        try:
            await conn.execute("""
                SELECT create_hypertable('indicators', 'timestamp',
                    if_not_exists => TRUE,
                    migrate_data => TRUE)
            """)
            logger.info("✅ indicators hypertable created")
        except Exception as e:
            logger.warning(f"⚠️ indicators hypertable: {e}")

        try:
            await conn.execute("""
                SELECT create_hypertable('signals', 'created_at',
                    if_not_exists => TRUE,
                    migrate_data => TRUE)
            """)
            logger.info("✅ signals hypertable created")
        except Exception as e:
            logger.warning(f"⚠️ signals hypertable: {e}")

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_interval ON ohlcv_data(ticker, interval)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_indicators_ticker_interval ON indicators(ticker, interval)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_ticker_created ON signals(ticker, created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_action_created ON signals(action, created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_articles_ticker ON sentiment_articles(ticker)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_snapshots_ticker ON sentiment_snapshots(ticker)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_outcomes_ticker ON signal_outcomes(ticker)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signal_outcomes_signal ON signal_outcomes(signal_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(is_read)"
        )
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_user ON signals(user_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_type_status ON jobs(job_type, status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positions_user ON positions(user_id, is_open)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_alerts_user ON price_alert_rules(user_id, is_active)"
        )

        logger.info("✅ PostgreSQL tables and indexes created")
