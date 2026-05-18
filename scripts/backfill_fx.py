"""One-shot 20-year FX backfill.

Ingests all 20 FX/metal pairs from `fx_majors.json` into `ohlcv_data` +
`indicators` using the multi-source dispatcher already wired into
`OHLCVService` (Phase 2). Pairs with a yfinance mapping pull from
yfinance; the three MT5-only pairs (USDCNH, XAUEUR, XAUAUD) attempt MT5
copy_rates and skip cleanly if the broker session isn't connected.

Idempotent: every UPSERT against ohlcv_data / indicators is keyed on
(ticker, interval, timestamp), so re-running just fills gaps without
duplicating rows.

Usage:
    cd backend
    .venv/Scripts/python scripts/backfill_fx.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

# Ensure the script can import from the backend package when run directly
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> int:
    from app.db.database import connect_db, get_db, close_db
    from app.core.asset_class import FX_TICKERS, asset_class_for, yfinance_symbol_for
    from app.services.ohlcv_service import OHLCVService

    await connect_db()
    pool = await get_db()
    svc = OHLCVService(pool)

    tickers = sorted(FX_TICKERS)
    print(f"[backfill] starting FX backfill for {len(tickers)} pairs")
    print(f"[backfill] period=max -> yfinance gives ~20-22y daily for majors")

    success: list[str] = []
    failed:  list[tuple[str, str]] = []
    total_rows = 0

    for t in tickers:
        cls = asset_class_for(t)
        src = yfinance_symbol_for(t)
        src_label = f"yfinance({src})" if src else "MT5"
        print(f"\n[{t}] class={cls}  source={src_label}")

        try:
            # _ingest_single dispatches to yfinance or MT5 internally based on
            # the registry entry. period="max" pulls every available daily bar
            # from yfinance (~20-22y for majors); MT5-only pairs accept the
            # ~20-year cap mapped in ohlcv_service.
            rows = await svc._ingest_single(t, interval="1d", period="max")
            total_rows += rows
            print(f"[{t}] ingested {rows} bars")
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

    # Spot-check row counts
    async with pool.acquire() as conn:
        for t in success[:5]:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n, MIN(timestamp) AS first, MAX(timestamp) AS last "
                "FROM ohlcv_data WHERE ticker = $1 AND interval = '1d'",
                t,
            )
            print(f"  {t:8s}  n={row['n']}  range={row['first']} -> {row['last']}")

    await close_db()
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
