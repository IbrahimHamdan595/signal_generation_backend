import yfinance as yf
import pandas as pd
import ta
import numpy as np
import asyncpg
from datetime import datetime, timezone
from typing import List, Tuple
import logging
import time

logger = logging.getLogger(__name__)

VALID_INTERVALS = {"1d", "1h", "5m", "15m", "30m"}
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"}

YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://finance.yahoo.com/",
}


class OHLCVService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ingest_tickers(
        self, tickers: List[str], interval: str = "1d", period: str = "1y"
    ) -> Tuple[List[str], List[str], int]:
        success, failed, total = [], [], 0
        for ticker in tickers:
            try:
                count = await self._ingest_single(ticker.upper(), interval, period)
                success.append(ticker.upper())
                total += count
                logger.info(f"✅ {ticker}: {count} records ingested")
            except Exception as e:
                failed.append(ticker.upper())
                logger.error(f"❌ {ticker}: {e}")
            time.sleep(2)
        return success, failed, total

    async def _ingest_single(self, ticker: str, interval: str, period: str) -> int:
        max_retries = 3
        df = None

        for attempt in range(max_retries):
            try:
                yf_ticker = yf.Ticker(ticker)
                df = yf_ticker.history(interval=interval, period=period)

                if df is not None and not df.empty:
                    logger.info(
                        f"✅ Successfully fetched {ticker} (attempt {attempt + 1})"
                    )
                    break
                else:
                    logger.warning(
                        f"⚠️ Empty response for {ticker}, attempt {attempt + 1}/{max_retries}"
                    )

            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                wait_time = 2**attempt
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Retrying {ticker} in {wait_time}s...")
                    time.sleep(wait_time)

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

        indicators_df = self._compute_indicators(df)

        async with self.pool.acquire() as conn:
            for _, row in df.iterrows():
                ts = row["timestamp"]
                ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)

                await conn.execute(
                    """
                    INSERT INTO ohlcv_data (ticker, interval, timestamp, open, high, low, close, volume, ingested_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (ticker, interval, timestamp) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume, ingested_at = EXCLUDED.ingested_at
                    """,
                    ticker,
                    interval,
                    ts,
                    float(row.get("open", 0)),
                    float(row.get("high", 0)),
                    float(row.get("low", 0)),
                    float(row.get("close", 0)),
                    float(row.get("volume", 0)),
                    datetime.now(timezone.utc),
                )

            for _, row in indicators_df.iterrows():
                ts = row["timestamp"]
                ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)

                def safe(val):
                    try:
                        v = float(val)
                        return None if np.isnan(v) else v
                    except Exception:
                        return None

                await conn.execute(
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
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                        $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29,
                        $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43,
                        $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56)
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
                    ticker,
                    interval,
                    ts,
                    safe(row.get("sma_20")),
                    safe(row.get("sma_50")),
                    safe(row.get("ema_12")),
                    safe(row.get("ema_26")),
                    safe(row.get("rsi_14")),
                    safe(row.get("macd_line")),
                    safe(row.get("macd_signal")),
                    safe(row.get("macd_histogram")),
                    safe(row.get("atr_14")),
                    safe(row.get("bb_upper")),
                    safe(row.get("bb_middle")),
                    safe(row.get("bb_lower")),
                    safe(row.get("bb_bandwidth")),
                    safe(row.get("obv")),
                    safe(row.get("mfi_14")),
                    safe(row.get("volume_roc")),
                    safe(row.get("stoch_k")),
                    safe(row.get("stoch_d")),
                    row.get("day_of_week"),
                    row.get("day_of_month"),
                    row.get("month"),
                    row.get("is_trading_day"),
                    safe(row.get("adx")),
                    safe(row.get("plus_di")),
                    safe(row.get("minus_di")),
                    safe(row.get("pivot")),
                    safe(row.get("resistance_1")),
                    safe(row.get("support_1")),
                    safe(row.get("resistance_2")),
                    safe(row.get("support_2")),
                    safe(row.get("price_sma20_dist")),
                    safe(row.get("price_sma50_dist")),
                    row.get("high_vol_regime"),
                    row.get("above_sma50"),
                    row.get("above_sma200"),
                    safe(row.get("normalized_volatility")),
                    safe(row.get("bb_position")),
                    safe(row.get("roc_5")),
                    safe(row.get("roc_10")),
                    row.get("higher_high"),
                    row.get("lower_low"),
                    safe(row.get("price_change_pct")),
                    row.get("volume_above_avg"),
                    safe(row.get("vix_level")),
                    safe(row.get("vix_change")),
                    row.get("earnings_days"),
                    safe(row.get("social_sentiment")),
                    safe(row.get("options_put_call_ratio")),
                    datetime.now(timezone.utc),
                )

        return len(df)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

        stoch = ta.momentum.stoch(high, low, close, window=14, smooth_k=3, smooth_d=3)
        out["stoch_k"] = stoch
        out["stoch_d"] = ta.momentum.stoch_signal(
            high, low, close, window=14, smooth_k=3, smooth_d=3
        )

        out["day_of_week"] = df["timestamp"].dt.dayofweek
        out["day_of_month"] = df["timestamp"].dt.day
        out["month"] = df["timestamp"].dt.month
        out["is_trading_day"] = (~df["timestamp"].dt.dayofweek.isin([5, 6])).astype(int)

        adx_indicator = ta.trend.adx(high, low, close, window=14)
        out["adx"] = adx_indicator
        plus_di = ta.trend.plus_di(high, low, close, window=14)
        minus_di = ta.trend.minus_di(high, low, close, window=14)
        out["plus_di"] = plus_di
        out["minus_di"] = minus_di

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

        out["vix_level"] = 20.0
        out["vix_change"] = 0.0
        out["earnings_days"] = 30
        out["social_sentiment"] = 0.0
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
