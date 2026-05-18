"""Macro feature service — global daily inputs for the FX model.

Refreshes the `fx_macro_features` table with three macro-regime indicators:

    dxy_close          — Dollar Index daily close (yfinance: DX-Y.NYB)
    dxy_ret_5d         — 5-day return on DXY (computed locally)
    vix_close          — VIX daily close (yfinance: ^VIX)
    yield_spread_10_2  — 10Y minus 2Y US Treasury yield % (FRED: DGS10 - DGS2)
    us_2y_yield        — US 2-year Treasury yield % (FRED: DGS2) — carry proxy

One row per calendar date, shared by all FX/metal tickers — these are
regime variables, not per-pair signals. Equity training samples zero-pad
these columns (handled in the dataset builder, not here).

The service is async-friendly and fetches all three sources concurrently.
A 20-year backfill takes ~10 seconds.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import asyncpg
import httpx
import pandas as pd
import yfinance as yf

from app.core.config import settings


logger = logging.getLogger(__name__)


class MacroFeatureService:
    """Refreshes the `fx_macro_features` table from yfinance + FRED."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    # ── Public entry point ────────────────────────────────────────────────────

    async def refresh_fx_macros(self, years_back: float = 1.0) -> int:
        """Pull DXY/VIX/yield-spread for the last `years_back` years and UPSERT.

        Use years_back=20 for first-time historical backfill; years_back=1
        is plenty for the daily incremental refresh job.

        Returns the number of dates upserted (one row per calendar date).
        """
        start = date.today() - timedelta(days=int(years_back * 366))
        end   = date.today()

        # Fan out all four fetches concurrently — independent sources.
        dxy_task   = asyncio.create_task(self._fetch_yf_series("DX-Y.NYB", start, end))
        vix_task   = asyncio.create_task(self._fetch_yf_series("^VIX",      start, end))
        yld_task   = asyncio.create_task(self._fetch_yield_spread(start, end))
        dgs2_task  = asyncio.create_task(self._fetch_fred_series("DGS2", start, end))

        dxy_s, vix_s, yld_s, dgs2_s = await asyncio.gather(
            dxy_task, vix_task, yld_task, dgs2_task
        )

        # Build a unioned date index so we UPSERT one row per date with all
        # available columns; missing values stay NULL (idempotent re-runs
        # fill them in when their source recovers).
        all_dates: set[date] = set()
        all_dates.update(dxy_s.index.date  if not dxy_s.empty  else [])
        all_dates.update(vix_s.index.date  if not vix_s.empty  else [])
        all_dates.update(yld_s.index.date  if not yld_s.empty  else [])
        all_dates.update(dgs2_s.index.date if not dgs2_s.empty else [])

        if not all_dates:
            logger.warning("⚠️  Macro refresh: all sources returned empty")
            return 0

        # Compute DXY 5-day return on its own series before unioning so the
        # rolling math sees a contiguous index.
        dxy_ret_5d = dxy_s.pct_change(5) if not dxy_s.empty else pd.Series(dtype=float)

        rows = []
        now = datetime.now(timezone.utc)
        for d in sorted(all_dates):
            ts = pd.Timestamp(d)
            rows.append((
                d,
                float(dxy_s.get(ts))      if ts in dxy_s.index      else None,
                float(dxy_ret_5d.get(ts)) if ts in dxy_ret_5d.index else None,
                float(vix_s.get(ts))      if ts in vix_s.index      else None,
                float(yld_s.get(ts))      if ts in yld_s.index      else None,
                float(dgs2_s.get(ts))     if ts in dgs2_s.index     else None,
                now,
            ))

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO fx_macro_features
                    (date, dxy_close, dxy_ret_5d, vix_close, yield_spread_10_2, us_2y_yield, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (date) DO UPDATE SET
                    dxy_close         = COALESCE(EXCLUDED.dxy_close,         fx_macro_features.dxy_close),
                    dxy_ret_5d        = COALESCE(EXCLUDED.dxy_ret_5d,        fx_macro_features.dxy_ret_5d),
                    vix_close         = COALESCE(EXCLUDED.vix_close,         fx_macro_features.vix_close),
                    yield_spread_10_2 = COALESCE(EXCLUDED.yield_spread_10_2, fx_macro_features.yield_spread_10_2),
                    us_2y_yield       = COALESCE(EXCLUDED.us_2y_yield,       fx_macro_features.us_2y_yield),
                    updated_at        = EXCLUDED.updated_at
                """,
                rows,
            )

        logger.info(
            f"✅ fx_macro_features refreshed: {len(rows)} dates "
            f"(DXY {len(dxy_s)} pts, VIX {len(vix_s)} pts, "
            f"spread {len(yld_s)} pts, DGS2 {len(dgs2_s)} pts)"
        )
        return len(rows)

    # ── Source-specific fetchers ──────────────────────────────────────────────

    async def _fetch_yf_series(self, symbol: str, start: date, end: date) -> pd.Series:
        """Fetch a yfinance daily close series, tz-naive, indexed by Timestamp."""
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    symbol,
                    start=start,
                    end=end + timedelta(days=1),  # yfinance end is exclusive
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    actions=False,
                    threads=False,
                ),
            )
        except Exception as e:
            logger.warning(f"⚠️  yfinance fetch failed for {symbol}: {e}")
            return pd.Series(dtype=float)

        if df is None or df.empty:
            return pd.Series(dtype=float)

        # Flatten MultiIndex if present (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            close_cols = [c for c in df.columns if (c[0] if isinstance(c, tuple) else c) == "Close"]
            s = df[close_cols[0]] if close_cols else df.iloc[:, 0]
        else:
            s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

        # Strip timezone, normalize to date-aligned timestamps
        s = pd.Series(s.values, index=pd.to_datetime(s.index).tz_localize(None).normalize())
        return s.dropna()

    async def _fetch_fred_series(self, series_id: str, start: date, end: date) -> pd.Series:
        """Fetch a single FRED daily series. Returns empty Series when API
        key is unset or the request fails."""
        api_key = settings.FRED_API_KEY
        if not api_key:
            logger.info(f"ℹ️  FRED_API_KEY unset — {series_id} will be NULL")
            return pd.Series(dtype=float)

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id":         series_id,
            "api_key":           api_key,
            "file_type":         "json",
            "observation_start": start.isoformat(),
            "observation_end":   end.isoformat(),
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"⚠️  FRED {series_id} fetch failed: {e}")
            return pd.Series(dtype=float)

        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        df = pd.DataFrame(obs)
        df["date"]  = pd.to_datetime(df["date"])
        # FRED uses "." for missing observations
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"]).set_index("date")["value"]

    async def _fetch_yield_spread(self, start: date, end: date) -> pd.Series:
        """Fetch DGS10 - DGS2 (10Y minus 2Y) from FRED."""
        dgs10, dgs2 = await asyncio.gather(
            self._fetch_fred_series("DGS10", start, end),
            self._fetch_fred_series("DGS2",  start, end),
        )
        if dgs10.empty or dgs2.empty:
            return pd.Series(dtype=float)

        # Align on shared dates and compute spread
        joined = pd.concat([dgs10, dgs2], axis=1, keys=["dgs10", "dgs2"]).dropna()
        spread = joined["dgs10"] - joined["dgs2"]
        return spread

    # ── Read helper for the dataset builder ────────────────────────────────────

    async def load_for_dates(self, start: date, end: date) -> dict[date, dict]:
        """Return {date: {dxy_close, dxy_ret_5d, vix_close, yield_spread_10_2, us_2y_yield}}
        for fast O(1) lookup in the dataset builder's per-bar loop."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT date, dxy_close, dxy_ret_5d, vix_close, yield_spread_10_2, us_2y_yield
                FROM fx_macro_features
                WHERE date BETWEEN $1 AND $2
                """,
                start, end,
            )
        return {
            r["date"]: {
                "dxy_close":         r["dxy_close"],
                "dxy_ret_5d":        r["dxy_ret_5d"],
                "vix_close":         r["vix_close"],
                "yield_spread_10_2": r["yield_spread_10_2"],
                "us_2y_yield":       r["us_2y_yield"],
            }
            for r in rows
        }
