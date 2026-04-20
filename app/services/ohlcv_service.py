import asyncio
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import asyncpg
import httpx
from datetime import datetime, date, timezone
from typing import List, Tuple, Optional
import logging

# ── Module-level caches (shared across all OHLCVService instances) ────────────
# VIX: keyed by period string → (Series, fetched_date)
_VIX_CACHE: dict[str, tuple[pd.Series, date]] = {}
# CBOE equity put/call: single cache entry → (Series, fetched_date)
_PUTCALL_CACHE: Optional[tuple[pd.Series, date]] = None

logger = logging.getLogger(__name__)

VALID_INTERVALS = {"1d", "1h", "5m", "15m", "30m"}
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"}


class OHLCVService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ingest_tickers(
        self,
        tickers: List[str],
        interval: str = "1d",
        period: str = "1y",
        concurrency: int = 5,
    ) -> Tuple[List[str], List[str], int]:
        """
        Ingest multiple tickers concurrently using a semaphore to stay within
        Yahoo Finance's rate limit (~2 req/s).

        concurrency=5 with a 0.5s stagger between slot acquisitions keeps the
        effective request rate at ~2/s while finishing 5x faster than serial.
        For 500 tickers: ~8 min instead of ~40 min.
        """
        success: List[str] = []
        failed:  List[str] = []
        total = 0
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()   # protects the shared result lists

        async def _fetch_one(ticker: str) -> None:
            nonlocal total
            async with sem:
                # Stagger starts so N concurrent slots don't all fire at t=0
                await asyncio.sleep(0.5)
                try:
                    count = await self._ingest_single(ticker.upper(), interval, period)
                    async with lock:
                        success.append(ticker.upper())
                        total += count
                    logger.info(f"✅ {ticker}: {count} records ingested")
                except Exception as e:
                    async with lock:
                        failed.append(ticker.upper())
                    logger.error(f"❌ {ticker}: {e}")

        await asyncio.gather(*[_fetch_one(t) for t in tickers])
        return success, failed, total

    async def _ingest_single(self, ticker: str, interval: str, period: str) -> int:
        loop = asyncio.get_event_loop()
        max_retries = 3
        df = None

        for attempt in range(max_retries):
            try:
                # yf.download() is more stable than Ticker.history() in yfinance 0.2.x+
                # threads=False prevents nested thread spawning inside the executor
                _t = ticker  # capture for lambda closure
                _i = interval
                _p = period
                df = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        _t,
                        period=_p,
                        interval=_i,
                        progress=False,
                        auto_adjust=True,
                        actions=False,
                        threads=False,
                    ),
                )

                if df is not None and not df.empty:
                    logger.info(f"✅ Successfully fetched {ticker} (attempt {attempt + 1})")
                    break
                else:
                    # Empty result is usually Yahoo Finance rate-limiting or a transient
                    # parse error inside yfinance — always retry with a longer wait
                    wait_time = 10 * (attempt + 1)
                    logger.warning(
                        f"⚠️ Empty response for {ticker}, attempt {attempt + 1}/{max_retries} "
                        f"— retrying in {wait_time}s"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)

            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"⏳ Retrying {ticker} in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for {ticker} after {max_retries} attempts"
            )

        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        col_map = {
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df.rename(columns=col_map, inplace=True)

        available = [
            c
            for c in ["timestamp", "open", "high", "low", "close", "volume"]
            if c in df.columns
        ]
        df = df[available].dropna(subset=["close"])

        if df.empty:
            raise ValueError(f"DataFrame empty after cleaning for {ticker}")

        # Strip timezone so all downstream comparisons with tz-naive VIX/earnings
        # series don't raise TypeError (yf.download returns tz-aware index)
        if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # ── Fetch real external data (all blocking — run in thread pool) ────────
        vix_series, earnings_dates, putcall_series = await asyncio.gather(
            loop.run_in_executor(None, lambda: self._fetch_vix_series(period)),
            loop.run_in_executor(None, lambda: self._fetch_earnings_dates(ticker)),
            loop.run_in_executor(None, lambda: self._fetch_putcall_series()),
        )

        indicators_df = await loop.run_in_executor(
            None,
            lambda: self._compute_indicators(df, vix_series, earnings_dates, putcall_series),
        )

        def safe(val):
            try:
                v = float(val)
                return None if np.isnan(v) else v
            except Exception:
                return None

        def norm_ts(ts):
            ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            return ts

        ingested_at = datetime.now(timezone.utc)

        # Build bulk records — one round-trip per table instead of one per row
        ohlcv_records = [
            (
                ticker, interval, norm_ts(row["timestamp"]),
                float(row.get("open", 0)), float(row.get("high", 0)),
                float(row.get("low", 0)), float(row.get("close", 0)),
                float(row.get("volume", 0)), ingested_at,
            )
            for _, row in df.iterrows()
        ]

        indicator_records = [
            (
                ticker, interval, norm_ts(row["timestamp"]),
                safe(row.get("sma_20")), safe(row.get("sma_50")),
                safe(row.get("ema_12")), safe(row.get("ema_26")),
                safe(row.get("rsi_14")), safe(row.get("macd_line")),
                safe(row.get("macd_signal")), safe(row.get("macd_histogram")),
                safe(row.get("atr_14")), safe(row.get("bb_upper")),
                safe(row.get("bb_middle")), safe(row.get("bb_lower")),
                safe(row.get("bb_bandwidth")), safe(row.get("obv")),
                safe(row.get("mfi_14")), safe(row.get("volume_roc")),
                safe(row.get("stoch_k")), safe(row.get("stoch_d")),
                row.get("day_of_week"), row.get("day_of_month"),
                row.get("month"), row.get("is_trading_day"),
                safe(row.get("adx")), safe(row.get("plus_di")),
                safe(row.get("minus_di")), safe(row.get("pivot")),
                safe(row.get("resistance_1")), safe(row.get("support_1")),
                safe(row.get("resistance_2")), safe(row.get("support_2")),
                safe(row.get("price_sma20_dist")), safe(row.get("price_sma50_dist")),
                row.get("high_vol_regime"), row.get("above_sma50"),
                row.get("above_sma200"), safe(row.get("normalized_volatility")),
                safe(row.get("bb_position")), safe(row.get("roc_5")),
                safe(row.get("roc_10")), row.get("higher_high"),
                row.get("lower_low"), safe(row.get("price_change_pct")),
                row.get("volume_above_avg"), safe(row.get("vix_level")),
                safe(row.get("vix_change")), row.get("earnings_days"),
                safe(row.get("social_sentiment")),
                safe(row.get("options_put_call_ratio")), ingested_at,
            )
            for _, row in indicators_df.iterrows()
        ]

        async with self.pool.acquire() as conn:
            # OHLCV — single executemany call (one server round-trip)
            await conn.executemany(
                """
                INSERT INTO ohlcv_data (ticker, interval, timestamp, open, high, low, close, volume, ingested_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (ticker, interval, timestamp) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                    close = EXCLUDED.close, volume = EXCLUDED.volume, ingested_at = EXCLUDED.ingested_at
                """,
                ohlcv_records,
            )

            # Indicators — single executemany call
            await conn.executemany(
                """
                INSERT INTO indicators (
                    ticker, interval, timestamp, sma_20, sma_50, ema_12, ema_26, rsi_14,
                    macd_line, macd_signal, macd_histogram, atr_14, bb_upper, bb_middle, bb_lower,
                    bb_bandwidth, obv, mfi_14, volume_roc, stoch_k, stoch_d, day_of_week,
                    day_of_month, month, is_trading_day, adx, plus_di, minus_di, pivot,
                    resistance_1, support_1, resistance_2, support_2, price_sma20_dist,
                    price_sma50_dist, high_vol_regime, above_sma50, above_sma200,
                    normalized_volatility, bb_position, roc_5, roc_10, higher_high,
                    lower_low, price_change_pct, volume_above_avg, vix_level, vix_change,
                    earnings_days, social_sentiment, options_put_call_ratio, computed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25,
                    $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37,
                    $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49,
                    $50, $51, $52
                )
                ON CONFLICT (ticker, interval, timestamp) DO UPDATE SET
                    sma_20 = EXCLUDED.sma_20, sma_50 = EXCLUDED.sma_50, ema_12 = EXCLUDED.ema_12,
                    ema_26 = EXCLUDED.ema_26, rsi_14 = EXCLUDED.rsi_14, macd_line = EXCLUDED.macd_line,
                    macd_signal = EXCLUDED.macd_signal, macd_histogram = EXCLUDED.macd_histogram,
                    atr_14 = EXCLUDED.atr_14, bb_upper = EXCLUDED.bb_upper, bb_middle = EXCLUDED.bb_middle,
                    bb_lower = EXCLUDED.bb_lower, bb_bandwidth = EXCLUDED.bb_bandwidth,
                    obv = EXCLUDED.obv, mfi_14 = EXCLUDED.mfi_14, volume_roc = EXCLUDED.volume_roc,
                    stoch_k = EXCLUDED.stoch_k, stoch_d = EXCLUDED.stoch_d,
                    day_of_week = EXCLUDED.day_of_week, day_of_month = EXCLUDED.day_of_month,
                    month = EXCLUDED.month, is_trading_day = EXCLUDED.is_trading_day,
                    adx = EXCLUDED.adx, plus_di = EXCLUDED.plus_di, minus_di = EXCLUDED.minus_di,
                    pivot = EXCLUDED.pivot, resistance_1 = EXCLUDED.resistance_1,
                    support_1 = EXCLUDED.support_1, resistance_2 = EXCLUDED.resistance_2,
                    support_2 = EXCLUDED.support_2, price_sma20_dist = EXCLUDED.price_sma20_dist,
                    price_sma50_dist = EXCLUDED.price_sma50_dist, high_vol_regime = EXCLUDED.high_vol_regime,
                    above_sma50 = EXCLUDED.above_sma50, above_sma200 = EXCLUDED.above_sma200,
                    normalized_volatility = EXCLUDED.normalized_volatility,
                    bb_position = EXCLUDED.bb_position, roc_5 = EXCLUDED.roc_5,
                    roc_10 = EXCLUDED.roc_10, higher_high = EXCLUDED.higher_high,
                    lower_low = EXCLUDED.lower_low, price_change_pct = EXCLUDED.price_change_pct,
                    volume_above_avg = EXCLUDED.volume_above_avg, vix_level = EXCLUDED.vix_level,
                    vix_change = EXCLUDED.vix_change, earnings_days = EXCLUDED.earnings_days,
                    social_sentiment = EXCLUDED.social_sentiment,
                    options_put_call_ratio = EXCLUDED.options_put_call_ratio,
                    computed_at = EXCLUDED.computed_at
                """,
                indicator_records,
            )

        return len(df)

    # ── External data fetchers ────────────────────────────────────────────────

    def _fetch_vix_series(self, period: str) -> pd.Series:
        """Download ^VIX history. Returns a date-normalised Series (date → close)."""
        global _VIX_CACHE
        today = date.today()
        cached = _VIX_CACHE.get(period)
        if cached and cached[1] == today:
            return cached[0]
        try:
            raw = yf.download("^VIX", period=period, interval="1d",
                              progress=False, auto_adjust=True)
            if raw.empty:
                return pd.Series(dtype=float)
            close = raw["Close"]
            # yfinance sometimes returns a MultiIndex column
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            idx = pd.to_datetime(close.index)
            if idx.tz is not None:
                idx = idx.tz_convert(None)
            close.index = idx.normalize()
            _VIX_CACHE[period] = (close, today)
            logger.info(f"✅ VIX: loaded {len(close)} bars")
            return close
        except Exception as e:
            logger.warning(f"⚠️  VIX fetch failed: {e}")
            return pd.Series(dtype=float)

    def _fetch_earnings_dates(self, ticker: str) -> pd.DatetimeIndex:
        """Return normalised tz-naive DatetimeIndex of all known earnings dates for ticker."""
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is None or ed.empty:
                return pd.DatetimeIndex([])
            idx = pd.to_datetime(ed.index)
            # yfinance returns tz-aware index (America/New_York) — strip for tz-naive comparison
            if idx.tz is not None:
                idx = idx.tz_convert(None)
            idx = idx.normalize().sort_values()
            logger.info(f"✅ Earnings for {ticker}: {len(idx)} dates")
            return idx
        except Exception as e:
            logger.warning(f"⚠️  Earnings fetch failed for {ticker}: {e}")
            return pd.DatetimeIndex([])

    def _fetch_putcall_series(self) -> pd.Series:
        """
        Download CBOE equity put/call ratio history (public CSV).
        Returns a date-indexed Series (date → ratio).
        Cached for the day; a failed fetch is also cached to avoid
        re-attempting (and re-logging) on every ticker.
        """
        global _PUTCALL_CACHE
        today = date.today()
        if _PUTCALL_CACHE and _PUTCALL_CACHE[1] == today:
            return _PUTCALL_CACHE[0]  # may be an empty Series if previously failed

        url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/EQPC_History.csv"
        empty = pd.Series(dtype=float)
        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), skiprows=1, names=["date", "pc"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            series = df.set_index("date")["pc"].astype(float)
            series.index = series.index.normalize()
            _PUTCALL_CACHE = (series, today)
            logger.info(f"✅ CBOE put/call: loaded {len(series)} days")
            return series
        except Exception as e:
            # Cache the failure so we don't retry (and re-log) on every ticker
            _PUTCALL_CACHE = (empty, today)
            logger.warning(f"⚠️  CBOE put/call unavailable (will use stub 1.0): {e}")
            return empty

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(
        self,
        df: pd.DataFrame,
        vix_series: pd.Series = None,
        earnings_dates: pd.DatetimeIndex = None,
        putcall_series: pd.Series = None,
    ) -> pd.DataFrame:
        out = df[["timestamp"]].copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        out["sma_20"] = ta.trend.sma_indicator(close, window=20)
        out["sma_50"] = ta.trend.sma_indicator(close, window=50)
        out["ema_12"] = ta.trend.ema_indicator(close, window=12)
        out["ema_26"] = ta.trend.ema_indicator(close, window=26)
        out["rsi_14"] = ta.momentum.rsi(close, window=14)

        macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        out["macd_line"] = macd_obj.macd()
        out["macd_signal"] = macd_obj.macd_signal()
        out["macd_histogram"] = macd_obj.macd_diff()

        out["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

        bb_obj = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        out["bb_upper"] = bb_obj.bollinger_hband()
        out["bb_middle"] = bb_obj.bollinger_mavg()
        out["bb_lower"] = bb_obj.bollinger_lband()
        out["bb_bandwidth"] = bb_obj.bollinger_wband()

        out["obv"] = ta.volume.on_balance_volume(close, volume)
        out["mfi_14"] = ta.volume.money_flow_index(high, low, close, volume, window=14)

        sma_volume = ta.trend.sma_indicator(volume, window=20)
        out["volume_roc"] = ((volume - sma_volume) / sma_volume * 100).fillna(0)

        out["stoch_k"] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        out["stoch_d"] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)

        out["day_of_week"] = df["timestamp"].dt.dayofweek
        out["day_of_month"] = df["timestamp"].dt.day
        out["month"] = df["timestamp"].dt.month
        out["is_trading_day"] = (~df["timestamp"].dt.dayofweek.isin([5, 6])).astype(int)

        _adx = ta.trend.ADXIndicator(high, low, close, window=14)
        out["adx"] = _adx.adx()
        out["plus_di"] = _adx.adx_pos()
        out["minus_di"] = _adx.adx_neg()

        pivot = (high + low + close) / 3
        out["pivot"] = pivot
        out["resistance_1"] = pivot + (high - low)
        out["support_1"] = pivot - (high - low)
        out["resistance_2"] = pivot + 2 * (high - low)
        out["support_2"] = pivot - 2 * (high - low)

        atr = out["atr_14"]
        out["price_sma20_dist"] = ((close - out["sma_20"]) / atr).fillna(0)
        out["price_sma50_dist"] = ((close - out["sma_50"]) / atr).fillna(0)

        atr_ma = ta.trend.sma_indicator(atr, window=20)
        out["high_vol_regime"] = (atr > atr_ma).astype(int)
        out["above_sma50"] = (close > out["sma_50"]).astype(int)
        out["above_sma200"] = (
            close > ta.trend.sma_indicator(close, window=200)
        ).astype(int)

        out["normalized_volatility"] = (
            atr / ta.trend.sma_indicator(atr, window=63)
        ) * 100

        bb_range = out["bb_upper"] - out["bb_lower"]
        bb_position = (close - out["bb_lower"]) / bb_range
        out["bb_position"] = bb_position.clip(0, 1) * 100

        out["roc_5"] = ta.momentum.roc(close, window=5)
        out["roc_10"] = ta.momentum.roc(close, window=10)

        out["higher_high"] = (high > high.shift(1)).astype(int)
        out["lower_low"] = (low < low.shift(1)).astype(int)

        out["price_change_pct"] = close.pct_change() * 100
        out["volume_above_avg"] = (
            volume > ta.trend.sma_indicator(volume, window=20)
        ).astype(int)

        # ── VIX ───────────────────────────────────────────────────────────────
        bar_dates = pd.to_datetime(df["timestamp"]).dt.normalize()
        if vix_series is not None and not vix_series.empty:
            mapped = bar_dates.map(vix_series)
            # Forward-fill gaps (weekends/holidays) up to 5 days
            mapped = mapped.ffill(limit=5).fillna(20.0)
            out["vix_level"] = mapped.values
            out["vix_change"] = (
                pd.Series(out["vix_level"]).pct_change().fillna(0.0) * 100
            ).values
        else:
            out["vix_level"]  = 20.0
            out["vix_change"] = 0.0

        # ── Days to next earnings ─────────────────────────────────────────────
        if earnings_dates is not None and len(earnings_dates) > 0:
            def _days_to_next(dt: pd.Timestamp) -> int:
                future = earnings_dates[earnings_dates >= dt]
                if len(future) == 0:
                    return 90   # no known upcoming date → neutral value
                return int((future[0] - dt).days)
            out["earnings_days"] = bar_dates.apply(_days_to_next).values
        else:
            out["earnings_days"] = 30

        # social_sentiment is populated later by Alpha Vantage pipeline;
        # keep a neutral default here so the column exists in every row.
        out["social_sentiment"] = 0.0

        # ── CBOE equity put/call ratio ────────────────────────────────────────
        if putcall_series is not None and not putcall_series.empty:
            mapped_pc = bar_dates.map(putcall_series)
            out["options_put_call_ratio"] = (
                mapped_pc.ffill(limit=5).fillna(1.0).values
            )
        else:
            out["options_put_call_ratio"] = 1.0

        return out

    async def get_ohlcv(
        self, ticker: str, interval: str = "1d", limit: int = 100
    ) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT open, high, low, close, volume, timestamp
                FROM ohlcv_data
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC
                LIMIT $3
                """,
                ticker.upper(),
                interval,
                limit,
            )
            return [dict(r) for r in reversed(rows)]

    async def get_indicators(
        self, ticker: str, interval: str = "1d", limit: int = 100
    ) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM indicators
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC
                LIMIT $3
                """,
                ticker.upper(),
                interval,
                limit,
            )
            return [dict(r) for r in reversed(rows)]

    async def get_latest_indicator(self, ticker: str, interval: str = "1d") -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM indicators
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                ticker.upper(),
                interval,
            )
            return dict(row) if row else None

    async def get_available_tickers(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT DISTINCT ticker FROM ohlcv_data")
            return [r["ticker"] for r in rows]
