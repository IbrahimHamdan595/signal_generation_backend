import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import asyncpg
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

INDICATOR_COLS = [
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "atr_14",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_bandwidth",
    "obv",
    "mfi_14",
    "volume_roc",
    "stoch_k",
    "stoch_d",
    "day_of_week",
    "day_of_month",
    "month",
    "is_trading_day",
    "adx",
    "plus_di",
    "minus_di",
    "pivot",
    "resistance_1",
    "support_1",
    "resistance_2",
    "support_2",
    "price_sma20_dist",
    "price_sma50_dist",
    "high_vol_regime",
    "above_sma50",
    "above_sma200",
    "normalized_volatility",
    "bb_position",
    "roc_5",
    "roc_10",
    "higher_high",
    "lower_low",
    "price_change_pct",
    "volume_above_avg",
    "vix_level",
    "vix_change",
    "earnings_days",
    "social_sentiment",
    "options_put_call_ratio",
]

SENTIMENT_COLS = ["avg_positive", "avg_negative", "avg_neutral", "avg_compound"]

FEATURE_COLS = OHLCV_COLS + INDICATOR_COLS

SEQUENCE_LEN = 60

REGRESSION_TARGETS = [
    "entry_price",
    "stop_loss",
    "take_profit",
    "net_profit",
    "bars_to_entry",
]


def compute_entry_time(current_ts: datetime, interval: str) -> datetime:
    if interval == "1d":
        next_day = current_ts + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day.replace(hour=14, minute=30, second=0, microsecond=0)
    elif interval == "1h":
        next_hour = current_ts.replace(minute=0, second=0, microsecond=0) + timedelta(
            hours=1
        )
        return next_hour
    else:
        return current_ts + timedelta(hours=1)


def bars_until_entry(interval: str) -> float:
    return 1.0


def compute_optimal_entry_bar(
    current_idx: int, closes: np.ndarray, label: int, lookahead_window: int = None
) -> float:
    if lookahead_window is None:
        lookahead_window = settings.LOOKAHEAD_WINDOW

    if label == 0:
        return 0.0

    end_idx = min(current_idx + lookahead_window + 1, len(closes))
    future_prices = closes[current_idx:end_idx]

    if len(future_prices) < 2:
        return 1.0

    current_price = closes[current_idx]

    if label == 1:
        optimal_idx = np.argmin(future_prices)
    elif label == 2:
        optimal_idx = np.argmax(future_prices)
    else:
        return 0.0

    optimal_price = future_prices[optimal_idx]

    if optimal_price == current_price:
        return 0.0

    bars = float(optimal_idx)
    bars = max(0.0, min(bars, float(lookahead_window)))
    return bars


class DatasetBuilder:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def build(
        self,
        tickers: List[str],
        interval: str = "1d",
        sequence_len: int = SEQUENCE_LEN,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        all_price, all_sent, all_cls, all_reg = [], [], [], []

        for ticker in tickers:
            try:
                p, s, c, r = await self._build_ticker(ticker, interval, sequence_len)
                if p is not None and len(p) > 0:
                    all_price.append(p)
                    all_sent.append(s)
                    all_cls.append(c)
                    all_reg.append(r)
                    logger.info(f"✅ {ticker}: {len(p)} sequences")
            except Exception as e:
                logger.error(f"❌ {ticker}: {e}")

        if not all_price:
            raise ValueError("No data built — run ingest + sentiment fetch first.")

        X_price = np.concatenate(all_price, axis=0).astype(np.float32)
        X_sentiment = np.concatenate(all_sent, axis=0).astype(np.float32)
        y_class = np.concatenate(all_cls, axis=0).astype(np.int64)
        y_regression = np.concatenate(all_reg, axis=0).astype(np.float32)

        X_price, scaler_params = self._normalise(X_price)

        logger.info(
            f"✅ Dataset: {X_price.shape[0]} sequences | "
            f"Buy={int((y_class == 1).sum())} "
            f"Sell={int((y_class == 2).sum())} "
            f"Hold={int((y_class == 0).sum())}"
        )
        return X_price, X_sentiment, y_class, y_regression, scaler_params

    async def _build_ticker(self, ticker: str, interval: str, sequence_len: int):
        async with self.pool.acquire() as conn:
            ohlcv_rows = await conn.fetch(
                """
                SELECT open, high, low, close, volume, timestamp
                FROM ohlcv_data
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp ASC
                """,
                ticker,
                interval,
            )

        if len(ohlcv_rows) < sequence_len + 2:
            logger.warning(f"⚠️  {ticker}: not enough rows ({len(ohlcv_rows)})")
            return None, None, None, None

        async with self.pool.acquire() as conn:
            ind_rows = await conn.fetch(
                """
                SELECT timestamp, """
                + ", ".join(INDICATOR_COLS)
                + """
                FROM indicators
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp ASC
                """,
                ticker,
                interval,
            )

            sent_row = await conn.fetchrow(
                """
                SELECT avg_positive, avg_negative, avg_neutral, avg_compound
                FROM sentiment_snapshots
                WHERE ticker = $1
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                ticker,
            )

        df_ohlcv = pd.DataFrame([dict(r) for r in ohlcv_rows])[
            ["timestamp"] + OHLCV_COLS
        ]
        df_ind = (
            pd.DataFrame([dict(r) for r in ind_rows])[["timestamp"] + INDICATOR_COLS]
            if ind_rows
            else pd.DataFrame()
        )

        if not df_ind.empty:
            df = pd.merge(df_ohlcv, df_ind, on="timestamp", how="left")
        else:
            df = df_ohlcv.copy()
            for col in INDICATOR_COLS:
                df[col] = np.nan

        df = df.sort_values("timestamp").reset_index(drop=True)
        df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

        if sent_row:
            sent_vec = np.array(
                [
                    sent_row.get("avg_positive", 0.0),
                    sent_row.get("avg_negative", 0.0),
                    sent_row.get("avg_neutral", 1.0),
                    sent_row.get("avg_compound", 0.0),
                ],
                dtype=np.float32,
            )
        else:
            sent_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        prices = df[FEATURE_COLS].values
        closes = df["close"].values
        timestamps = df["timestamp"].values
        atrs = df["atr_14"].values if "atr_14" in df.columns else np.zeros(len(df))

        X_price, X_sent, y_cls, y_reg = [], [], [], []

        for i in range(sequence_len, len(df) - 1):
            window = prices[i - sequence_len : i]
            future_close = closes[i + 1]
            current_close = closes[i]
            _ = pd.Timestamp(timestamps[i]).to_pydatetime()
            atr = atrs[i] if atrs[i] > 0 else current_close * 0.02

            ret = (future_close - current_close) / current_close
            if ret > 0.005:
                label = 1
            elif ret < -0.005:
                label = 2
            else:
                label = 0

            entry = current_close
            stop_loss = entry - 1.5 * atr if label == 1 else entry + 1.5 * atr
            take_profit = entry + 3.0 * atr if label == 1 else entry - 3.0 * atr
            net_profit = abs(take_profit - entry) - abs(stop_loss - entry)

            bars_to_entry = compute_optimal_entry_bar(i, closes, label)

            X_price.append(window)
            X_sent.append(sent_vec)
            y_cls.append(label)
            y_reg.append([entry, stop_loss, take_profit, net_profit, bars_to_entry])

        return (
            np.array(X_price, dtype=np.float32),
            np.array(X_sent, dtype=np.float32),
            np.array(y_cls, dtype=np.int64),
            np.array(y_reg, dtype=np.float32),
        )

    def _normalise(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        mean = X_flat.mean(axis=0)
        std = X_flat.std(axis=0) + 1e-8
        return (X_flat - mean).reshape(N, T, F) / std, {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
