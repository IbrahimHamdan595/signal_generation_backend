"""Backfill 4h FX bars + indicators for the 17 yfinance-covered pairs.

yfinance limits sub-daily intervals to the last 730 days, so this gives us
~2 years of 4h bars per pair (~4,300 bars). The MTF (multi-timeframe)
training step joins these against the existing daily bars to give the model
shorter-horizon context alongside the daily features.

MT5-only pairs (USDCNH, XAUEUR, XAUAUD) are skipped here; they can be added
later via the MT5 dispatcher in `_ingest_single` if a live session is up.

Idempotent — re-runs UPSERT into ohlcv_data + indicators.

Usage:
    cd backend
    .venv/Scripts/python scripts/backfill_fx_4h.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> int:
    from app.db.database import connect_db, get_db, close_db
    from app.core.asset_class import FX_TICKERS, yfinance_symbol_for
    from app.services.ohlcv_service import OHLCVService

    await connect_db()
    pool = await get_db()
    svc = OHLCVService(pool)

    # Only pairs with yfinance coverage — MT5-only pairs need a live session
    # that may not be up while this script runs.
    tickers = sorted(t for t in FX_TICKERS if yfinance_symbol_for(t) is not None)
    print(f"[backfill_4h] {len(tickers)} pairs (yfinance-covered)")
    print(f"[backfill_4h] interval=4h, period=730d (yfinance limit)")

    success: list[str] = []
    failed:  list[tuple[str, str]] = []
    total_rows = 0

    for t in tickers:
        print(f"\n[{t}] 4h ingest...")
        try:
            rows = await svc._ingest_single(t, interval="4h", period="730d")
            total_rows += rows
            print(f"[{t}] ingested {rows} 4h bars")
            success.append(t)
        except Exception as e:
            print(f"[{t}] FAILED: {e}")
            failed.append((t, str(e)))

    print()
    print("=" * 60)
    print(f"[done] success={len(success)}  failed={len(failed)}  total_rows={total_rows}")
    if failed:
        print("\nFailed pairs:")
        for t, err in failed:
            print(f"  {t}: {err}")

    # Spot-check
    async with pool.acquire() as conn:
        for t in success[:3]:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM ohlcv_data WHERE ticker = $1 AND interval = '4h'",
                t,
            )
            ind_count = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM indicators WHERE ticker = $1 AND interval = '4h'",
                t,
            )
            print(f"  {t:8s}  ohlcv 4h rows = {row['n']}  indicators 4h rows = {ind_count['n']}")

    await close_db()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
