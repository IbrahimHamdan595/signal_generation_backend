"""
Bulk S&P 500 ingest script
==========================

Ingests OHLCV + indicators for all (or a subset of) S&P 500 tickers,
then optionally enriches social_sentiment from Alpha Vantage.

Usage
-----
    # Full S&P 500, 5-year history (recommended before first training run)
    python scripts/ingest_sp500.py --period 5y

    # Test with first 20 tickers only
    python scripts/ingest_sp500.py --period 5y --limit 20

    # Also enrich social sentiment after ingestion (needs ALPHAVANTAGE_KEY in .env)
    python scripts/ingest_sp500.py --period 5y --enrich-sentiment

    # Resume: skip tickers already in the DB
    python scripts/ingest_sp500.py --period 5y --skip-existing

Options
-------
    --period          yfinance period: 1y 2y 5y (default: 5y)
    --interval        bar interval: 1d 1h (default: 1d)
    --limit           only process first N tickers (default: all ~503)
    --batch           tickers per HTTP batch (default: 10)
    --delay           seconds between batches (default: 3)
    --skip-existing   skip tickers already present in the DB
    --enrich-sentiment  run Alpha Vantage social_sentiment enrichment after ingest
    --enrich-years    years of AV history per ticker (default: 2, costs 2 AV calls)
    --enrich-batch    tickers per AV enrichment batch (default: 12, stays ≤25 calls/day)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Make sure we can import app modules ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.services.ohlcv_service import OHLCVService
from app.services.sentiment_service import SentimentService
from app.services.news_service import get_sp500_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_sp500")


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def progress_bar(done: int, total: int, width: int = 40) -> str:
    filled = int(width * done / max(total, 1))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * done / max(total, 1)
    return f"[{bar}] {done}/{total} ({pct:.1f}%)"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace):
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set in .env")
        sys.exit(1)

    # asyncpg needs postgresql:// not postgresql+asyncpg://
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    logger.info("🔌 Connecting to database…")
    pool = await asyncpg.create_pool(db_url, min_size=2, max_size=5)

    sp500_map = get_sp500_map()
    all_tickers = list(sp500_map.keys())

    if args.limit:
        all_tickers = all_tickers[: args.limit]

    # ── Optionally skip already-ingested tickers ──────────────────────────────
    if args.skip_existing:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT DISTINCT ticker FROM ohlcv_data")
            existing = {r["ticker"] for r in rows}
        before = len(all_tickers)
        all_tickers = [t for t in all_tickers if t not in existing]
        logger.info(
            f"⏩ Skipping {before - len(all_tickers)} already-ingested tickers"
        )

    total   = len(all_tickers)
    success : list[str] = []
    failed  : list[str] = []

    logger.info(
        f"📊 Ingesting {total} tickers | period={args.period} "
        f"interval={args.interval} batch={args.batch}"
    )
    start = time.time()

    svc = OHLCVService(pool)

    for batch_num, batch in enumerate(chunk(all_tickers, args.batch), 1):
        batch_ok, batch_fail, count = await svc.ingest_tickers(
            batch, args.interval, args.period
        )
        success.extend(batch_ok)
        failed.extend(batch_fail)

        done = len(success) + len(failed)
        elapsed = time.time() - start
        rate    = done / max(elapsed, 1)
        eta_s   = (total - done) / max(rate, 0.001)
        eta_str = f"{int(eta_s // 60)}m {int(eta_s % 60)}s"

        logger.info(
            f"{progress_bar(done, total)}  "
            f"batch {batch_num}  records +{count}  "
            f"ETA {eta_str}"
        )

        if batch_num * args.batch < total:
            logger.info(f"  ⏳ Waiting {args.delay}s before next batch…")
            await asyncio.sleep(args.delay)

    elapsed_total = time.time() - start
    logger.info(
        f"\n✅ Ingest complete in {elapsed_total:.0f}s\n"
        f"   Success : {len(success)}\n"
        f"   Failed  : {len(failed)}\n"
        f"   Failed tickers: {failed or 'none'}"
    )

    # ── Save a run report ─────────────────────────────────────────────────────
    report = {
        "run_at":   datetime.now(timezone.utc).isoformat(),
        "period":   args.period,
        "interval": args.interval,
        "total":    total,
        "success":  len(success),
        "failed":   failed,
        "elapsed_s": round(elapsed_total, 1),
    }
    report_path = ROOT / "checkpoints" / "ingest_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"📄 Report saved → {report_path}")

    # ── Optional: Alpha Vantage social_sentiment enrichment ───────────────────
    if args.enrich_sentiment:
        av_key = os.environ.get("ALPHAVANTAGE_KEY", "")
        if not av_key:
            logger.warning(
                "⚠️  --enrich-sentiment requested but ALPHAVANTAGE_KEY not set in .env — skipping"
            )
        else:
            logger.info(
                f"\n🧠 Starting social_sentiment enrichment…\n"
                f"   AV free tier: 25 calls/day  |  "
                f"{args.enrich_batch} tickers × {args.enrich_years} years "
                f"= {args.enrich_batch * args.enrich_years} calls/run\n"
                f"   Processing {len(success)} ingested tickers "
                f"in batches of {args.enrich_batch}"
            )

            sent_svc = SentimentService(pool)
            enrich_ok: list[str] = []
            enrich_fail: list[str] = []

            for batch_num, batch in enumerate(
                chunk(success, args.enrich_batch), 1
            ):
                logger.info(
                    f"  AV batch {batch_num}: {batch}"
                )
                ok, fail = await sent_svc.enrich_social_sentiment(
                    batch, period_years=args.enrich_years
                )
                enrich_ok.extend(ok)
                enrich_fail.extend(fail)

                remaining_batches = (len(success) // args.enrich_batch) - batch_num
                if remaining_batches > 0:
                    # Stay well within 25 calls/day — sleep between batches
                    wait = max(args.delay, 5)
                    logger.info(f"  ⏳ Waiting {wait}s…")
                    await asyncio.sleep(wait)

            logger.info(
                f"\n✅ Enrichment complete\n"
                f"   Enriched : {len(enrich_ok)}\n"
                f"   Failed   : {len(enrich_fail)}"
            )

    await pool.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk S&P 500 OHLCV ingest with optional AV sentiment enrichment"
    )
    parser.add_argument("--period",    default="5y",
                        choices=["1y", "2y", "5y", "3mo", "6mo"],
                        help="yfinance history period (default: 5y)")
    parser.add_argument("--interval",  default="1d",
                        choices=["1d", "1h"],
                        help="Bar interval (default: 1d)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Only process first N tickers (default: all)")
    parser.add_argument("--batch",     type=int, default=10,
                        help="Tickers per batch (default: 10)")
    parser.add_argument("--delay",     type=float, default=3.0,
                        help="Seconds between batches (default: 3)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tickers already present in the DB")
    parser.add_argument("--enrich-sentiment", action="store_true",
                        help="Run Alpha Vantage sentiment enrichment after ingest")
    parser.add_argument("--enrich-years", type=int, default=2,
                        help="Years of AV history per ticker (default: 2)")
    parser.add_argument("--enrich-batch", type=int, default=12,
                        help="Tickers per AV enrichment batch (default: 12, ≤25 calls/day)")

    args = parser.parse_args()
    asyncio.run(main(args))
