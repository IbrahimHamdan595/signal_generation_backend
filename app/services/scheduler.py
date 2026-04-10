import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
import logging

from app.db.database import get_db
from app.services.ohlcv_service import OHLCVService
from app.services.sentiment_service import SentimentService
from app.services.signal_service import SignalService
from app.services.outcome_service import OutcomeService
from app.services.ticker_list_service import ticker_list_service
from app.ml.models.registry import is_model_trained

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()

MAX_RETRIES = 3
RETRY_DELAYS = [60, 300, 900]

_job_failure_counts: dict[str, int] = {}


def _alert_on_failure(job_id: str, exc: Exception):
    logger.error(
        f"🚨 Scheduler job '{job_id}' failed after {MAX_RETRIES} retries: {exc}"
    )


def _listener(event):
    job_id = event.job_id

    if event.exception:
        _job_failure_counts[job_id] = _job_failure_counts.get(job_id, 0) + 1
        count = _job_failure_counts[job_id]

        if count >= MAX_RETRIES:
            _alert_on_failure(job_id, event.exception)
            _job_failure_counts[job_id] = 0
        else:
            delay = RETRY_DELAYS[min(count - 1, len(RETRY_DELAYS) - 1)]
            logger.warning(
                f"⚠️  Job '{job_id}' failed (attempt {count}/{MAX_RETRIES}), "
                f"retrying in {delay}s: {event.exception}"
            )
    else:
        if job_id in _job_failure_counts:
            _job_failure_counts[job_id] = 0


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
    "NFLX",
]


async def _get_tracked_tickers() -> list:
    try:
        pool = await get_db()
        if pool:
            svc = OHLCVService(pool)
            tickers = await svc.get_available_tickers()
            if tickers:
                return tickers
    except Exception:
        pass
    return await ticker_list_service.get_tickers()


async def regenerate_signals_sp500():
    if not is_model_trained():
        logger.info("⏭  S&P 500 Signal regen skipped — model not trained yet.")
        return

    logger.info("🔄 Scheduler: S&P 500 signal generation...")
    try:
        pool = await get_db()
        tickers = await ticker_list_service.get_tickers()
        svc = SignalService(pool)

        logger.info(f"📊 Generating signals for {len(tickers)} S&P 500 tickers...")

        skipped = []
        for i, ticker in enumerate(tickers):
            try:
                result = await svc.generate_and_store(ticker, "1d")
                if result is None:
                    skipped.append(ticker)
                if (i + 1) % 50 == 0:
                    logger.info(
                        f"📊 Progress: {i + 1}/{len(tickers)} tickers processed"
                    )
            except Exception as e:
                logger.warning(f"⚠️  Failed to generate signal for {ticker}: {e}")
                skipped.append(ticker)

        logger.info(
            f"✅ S&P 500 signals: {len(tickers) - len(skipped)} generated, {len(skipped)} skipped"
        )
    except Exception as e:
        logger.error(f"❌ S&P 500 signal generation error: {e}")
        raise


async def refresh_daily_data():
    logger.info("🔄 Scheduler: daily data refresh...")
    try:
        pool = await get_db()
        tickers = await _get_tracked_tickers()
        svc = OHLCVService(pool)
        success, failed, count = await svc.ingest_tickers(tickers, "1d", "3mo")
        logger.info(f"✅ Daily refresh: {count} records | failed: {failed}")
    except Exception as e:
        logger.error(f"❌ Daily refresh error: {e}")
        raise


async def refresh_hourly_data():
    logger.info("🔄 Scheduler: hourly data refresh...")
    try:
        pool = await get_db()
        tickers = await _get_tracked_tickers()
        svc = OHLCVService(pool)
        success, failed, count = await svc.ingest_tickers(tickers, "1h", "7d")
        logger.info(f"✅ Hourly refresh: {count} records | failed: {failed}")
    except Exception as e:
        logger.error(f"❌ Hourly refresh error: {e}")
        raise


async def refresh_sentiment():
    logger.info("🔄 Scheduler: sentiment refresh...")
    try:
        pool = await get_db()
        tickers = await _get_tracked_tickers()
        svc = SentimentService(pool)
        success, failed, total = await svc.run_pipeline(tickers[:20], limit=5)
        logger.info(f"✅ Sentiment refresh: {total} articles | failed: {failed}")
    except Exception as e:
        logger.error(f"❌ Sentiment refresh error: {e}")
        raise


async def enrich_social_sentiment():
    """
    Weekly job: pull Alpha Vantage historical sentiment and write per-bar
    social_sentiment scores into the indicators table.
    Processes up to 12 tickers per run to stay within the 25 calls/day limit
    (2 AV calls per ticker × 12 tickers = 24 calls).
    """
    logger.info("🔄 Scheduler: social sentiment enrichment (Alpha Vantage)...")
    try:
        pool = await get_db()
        tickers = await _get_tracked_tickers()
        svc = SentimentService(pool)
        # 12 tickers × 2 years = 24 AV calls — stays within free-tier limit
        success, failed = await svc.enrich_social_sentiment(tickers[:12], period_years=2)
        logger.info(f"✅ Social sentiment enriched: {len(success)} ok, {len(failed)} failed")
    except Exception as e:
        logger.error(f"❌ Social sentiment enrichment error: {e}")
        raise


async def regenerate_signals():
    if not is_model_trained():
        logger.info("⏭  Signal regen skipped — model not trained yet.")
        return

    logger.info("🔄 Scheduler: regenerating signals...")
    try:
        pool = await get_db()
        tickers = await _get_tracked_tickers()
        svc = SignalService(pool)

        async def generate_for_interval(interval: str):
            results = await svc.generate_batch(tickers, interval)
            logger.info(f"✅ Signals [{interval}]: {len(results)} generated")
            return results

        await asyncio.gather(generate_for_interval("1d"), generate_for_interval("1h"))
    except Exception as e:
        logger.error(f"❌ Signal regen error: {e}")
        raise


def start_scheduler():
    scheduler.add_listener(_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    scheduler.add_job(
        refresh_daily_data,
        CronTrigger(hour=17, minute=5),
        id="daily_data",
        replace_existing=True,
    )

    scheduler.add_job(
        refresh_hourly_data,
        CronTrigger(day_of_week="mon-fri", hour="13-21", minute=2),
        id="hourly_data",
        replace_existing=True,
    )

    scheduler.add_job(
        refresh_sentiment,
        CronTrigger(hour="0,6,12,18", minute=30),
        id="sentiment_refresh",
        replace_existing=True,
    )

    scheduler.add_job(
        regenerate_signals,
        CronTrigger(hour="*", minute=15),
        id="signal_regen",
        replace_existing=True,
    )

    scheduler.add_job(
        regenerate_signals_sp500,
        CronTrigger(hour=9, minute=30),
        id="sp500_signal_gen",
        replace_existing=True,
    )

    # Weekly on Sunday at 02:00 — AV free tier: 25 calls/day, 12 tickers × 2 = 24
    scheduler.add_job(
        enrich_social_sentiment,
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="social_sentiment_enrichment",
        replace_existing=True,
    )

    scheduler.add_job(
        check_signal_outcomes,
        CronTrigger(hour="*", minute=45),
        id="outcome_check",
        replace_existing=True,
    )

    # Post-market: ingest closing bars at 22:00 UTC (5 PM EST) on weekdays
    scheduler.add_job(
        refresh_post_market,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=0),
        id="post_market_ingest",
        replace_existing=True,
    )

    # Post-market signal generation at 22:30 UTC after data is ingested
    scheduler.add_job(
        regenerate_signals_post_close,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=30),
        id="post_market_signals",
        replace_existing=True,
    )

    # Check user-defined price alert rules every 5 minutes
    scheduler.add_job(
        check_price_alert_rules,
        CronTrigger(minute="*/5"),
        id="price_alert_rules",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("✅ Scheduler started (daily data | hourly data | sentiment | signals | post-market)")


async def refresh_post_market():
    """Ingest latest daily bars right after market close (22:00 UTC = 5 PM EST)."""
    logger.info("🔄 Scheduler: post-market data refresh...")
    try:
        pool = await get_db()
        tickers = await _get_tracked_tickers()
        svc = OHLCVService(pool)
        success, failed, count = await svc.ingest_tickers(tickers, "1d", "5d")
        logger.info(f"✅ Post-market refresh: {count} records | failed: {failed}")
    except Exception as e:
        logger.error(f"❌ Post-market refresh error: {e}")
        raise


async def regenerate_signals_post_close():
    """Generate fresh signals for all tracked tickers after market close."""
    if not is_model_trained():
        logger.info("⏭  Post-close signal regen skipped — model not trained.")
        return
    logger.info("🔄 Scheduler: post-close signal generation...")
    try:
        pool = await get_db()
        tickers = await _get_tracked_tickers()
        svc = SignalService(pool)
        results = await svc.generate_batch(tickers, "1d")
        logger.info(f"✅ Post-close signals: {len(results)} generated")
    except Exception as e:
        logger.error(f"❌ Post-close signal regen error: {e}")
        raise


async def check_price_alert_rules():
    """Check user-defined price threshold alert rules against current DB prices."""
    try:
        pool = await get_db()
        if pool is None:
            return
        async with pool.acquire() as conn:
            rules = await conn.fetch("""
                SELECT par.*, o.close AS current_price
                FROM price_alert_rules par
                LEFT JOIN LATERAL (
                    SELECT close FROM ohlcv_data
                    WHERE ticker = par.ticker AND interval = '1d'
                    ORDER BY timestamp DESC LIMIT 1
                ) o ON TRUE
                WHERE par.is_active = TRUE AND o.close IS NOT NULL
            """)
            for rule in rules:
                price = rule["current_price"]
                triggered = (
                    (rule["condition"] == "above" and price >= rule["target_price"]) or
                    (rule["condition"] == "below" and price <= rule["target_price"])
                )
                if triggered:
                    msg = (
                        f"{rule['ticker']} hit your {rule['condition']} "
                        f"${rule['target_price']:.2f} target (now ${price:.2f})"
                    )
                    await conn.execute("""
                        INSERT INTO alerts (ticker, action, confidence, message, user_id)
                        VALUES ($1, 'HOLD', 1.0, $2, $3)
                    """, rule["ticker"], msg, rule["user_id"])
                    await conn.execute("""
                        UPDATE price_alert_rules
                        SET is_active = FALSE, triggered_at = NOW()
                        WHERE id = $1
                    """, rule["id"])
                    logger.info(f"🔔 Price rule triggered: {msg}")
    except Exception as e:
        logger.error(f"❌ Price alert rule check error: {e}")


async def check_signal_outcomes():
    logger.info("🔄 Scheduler: checking signal outcomes...")
    try:
        pool = await get_db()
        svc = OutcomeService(pool)
        result = await svc.check_pending_outcomes()
        logger.info(f"✅ Outcomes: {result['resolved']} resolved, {result['skipped']} pending")
    except Exception as e:
        logger.error(f"❌ Outcome check error: {e}")
        raise


def stop_scheduler():
    scheduler.shutdown()
    logger.info("🔌 Scheduler stopped")
