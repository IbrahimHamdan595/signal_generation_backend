import numpy as np
import pandas as pd
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
        """
        Build the full dataset with per-ticker Z-score normalisation.
        Each ticker gets its own mean/std computed on its own sequences only,
        preventing price-level contamination between different stocks.
        scaler_params is a dict  ticker → {mean: [...], std: [...]}.
        """
        all_price, all_sent, all_cls, all_reg = [], [], [], []
        scaler_params: dict = {}

        for ticker in tickers:
            try:
                p, s, c, r = await self._build_ticker(ticker, interval, sequence_len)
                if p is not None and len(p) > 0:
                    # ── Per-ticker normalisation ──────────────────────────────
                    p_norm, t_scaler = self._normalise(p)
                    scaler_params[ticker] = t_scaler
                    all_price.append(p_norm)
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

        if len(ohlcv_rows) < sequence_len + settings.LOOKAHEAD_WINDOW + 1:
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

            # Per-day sentiment from daily_sentiment table (Finnhub + FinBERT)
            daily_sent_rows = await conn.fetch(
                """
                SELECT date, avg_positive, avg_negative, avg_neutral, avg_compound
                FROM daily_sentiment
                WHERE ticker = $1
                ORDER BY date ASC
                """,
                ticker,
            )

            # Global snapshot as fallback
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

        # ── Build per-bar sentiment lookup ────────────────────────────────────
        # Priority:
        #  1. daily_sentiment table (Finnhub + FinBERT, per trading day) — best
        #  2. social_sentiment column (AV compound scalar per bar) — partial
        #  3. Global snapshot — coarse fallback
        #  4. Neutral zeros — last resort

        # Build date → sentiment dict from daily_sentiment table
        daily_sent_map: dict = {}
        for r in daily_sent_rows:
            date_key = str(r["date"])  # "YYYY-MM-DD"
            daily_sent_map[date_key] = {
                "avg_positive": float(r["avg_positive"] or 0.0),
                "avg_negative": float(r["avg_negative"] or 0.0),
                "avg_neutral":  float(r["avg_neutral"]  or 1.0),
                "avg_compound": float(r["avg_compound"] or 0.0),
            }

        has_daily = len(daily_sent_map) > 0

        # social_sentiment column as scalar compound fallback
        social_col = "social_sentiment"
        has_social = (
            not has_daily
            and social_col in df.columns
            and df[social_col].notna().any()
            and (df[social_col] != 0.0).any()
        )
        if has_social:
            social_series = df[social_col].ffill().bfill().fillna(0.0)
        else:
            social_series = pd.Series(0.0, index=df.index)

        # Global snapshot fallback
        if sent_row:
            global_pos = float(sent_row.get("avg_positive", 0.0))
            global_neg = float(sent_row.get("avg_negative", 0.0))
            global_neu = float(sent_row.get("avg_neutral",  1.0))
            global_cmp = float(sent_row.get("avg_compound", 0.0))
        else:
            global_pos, global_neg, global_neu, global_cmp = 0.0, 0.0, 1.0, 0.0

        # Pre-build a list of date strings aligned to df rows for fast lookup
        df_dates = [
            pd.Timestamp(ts).strftime("%Y-%m-%d") if ts is not None else ""
            for ts in df["timestamp"].values
        ]

        # For daily_sentiment, forward-fill missing days (weekend/holiday gaps)
        # by expanding the map to cover all dates in df
        if has_daily:
            sorted_daily_keys = sorted(daily_sent_map.keys())
            last_known = {"avg_positive": 0.0, "avg_negative": 0.0,
                          "avg_neutral": 1.0, "avg_compound": 0.0}
            daily_sent_filled: dict = {}
            daily_ptr = 0
            for date_str in df_dates:
                # Advance pointer to pick up any daily entries on or before this date
                while daily_ptr < len(sorted_daily_keys) and sorted_daily_keys[daily_ptr] <= date_str:
                    last_known = daily_sent_map[sorted_daily_keys[daily_ptr]]
                    daily_ptr += 1
                daily_sent_filled[date_str] = last_known
        else:
            daily_sent_filled = {}

        prices     = df[FEATURE_COLS].values
        closes     = df["close"].values
        timestamps = df["timestamp"].values
        atrs       = df["atr_14"].values if "atr_14" in df.columns else np.zeros(len(df))

        X_price, X_sent, y_cls, y_reg = [], [], [], []

        lookahead   = settings.LOOKAHEAD_WINDOW
        buy_thresh  = settings.BUY_THRESHOLD
        sell_thresh = settings.SELL_THRESHOLD

        # Leave enough future bars for the lookahead window
        for i in range(sequence_len, len(df) - lookahead):
            window        = prices[i - sequence_len : i]
            current_close = closes[i]
            atr           = atrs[i] if atrs[i] > 0 else current_close * 0.02

            # Per-bar sentiment vector — priority: daily_sentiment > social_col > global
            if has_daily:
                day_sent = daily_sent_filled.get(df_dates[i], {
                    "avg_positive": global_pos, "avg_negative": global_neg,
                    "avg_neutral": global_neu, "avg_compound": global_cmp,
                })
                bar_pos     = day_sent["avg_positive"]
                bar_neg     = day_sent["avg_negative"]
                bar_neu     = day_sent["avg_neutral"]
                bar_compound = day_sent["avg_compound"]
            elif has_social:
                bar_compound = float(social_series.iloc[i])
                bar_pos = max(0.0,  bar_compound)
                bar_neg = max(0.0, -bar_compound)
                bar_neu = round(1.0 - abs(bar_compound), 6)
            else:
                bar_pos, bar_neg, bar_neu, bar_compound = global_pos, global_neg, global_neu, global_cmp

            sent_vec = np.array(
                [bar_pos, bar_neg, bar_neu, bar_compound], dtype=np.float32
            )

            # ── Fix 1: Triple-barrier labeling ───────────────────────────────
            # Walk forward bar-by-bar; assign label at first barrier touched.
            # BUY  barrier: close rises   ≥ buy_thresh  (default +2%)
            # SELL barrier: close falls   ≥ sell_thresh (default -1%)
            # HOLD: neither barrier touched within lookahead window.
            # Asymmetric thresholds encode a 2:1 reward-to-risk expectation.
            future_closes = closes[i + 1 : i + 1 + lookahead]
            label = 0  # HOLD default
            for fc in future_closes:
                ret = (fc - current_close) / current_close
                if ret >= buy_thresh:
                    label = 1   # BUY — upper barrier hit first
                    break
                if ret <= -sell_thresh:
                    label = 2   # SELL — lower barrier hit first
                    break

            # ── Regression targets as % deviations from entry ────────────────
            # Storing absolute prices (e.g. NVDA ~$900) causes regression
            # gradients that are ~1000× larger than classification gradients,
            # collapsing the model to predict only one class.
            # Solution: all price targets are stored as decimal % of entry.
            #   entry_price  = 0.0  (always — "no deviation from current close")
            #   stop_loss    = negative % (e.g. -0.015 = 1.5% below entry)
            #   take_profit  = positive % (e.g. +0.030 = 3.0% above entry)
            #   net_profit   = TP% + SL% combined
            # At inference time, multiply by current price to recover dollars.
            entry = current_close
            if entry <= 0:
                continue

            if label == 1:  # BUY
                tp_abs      = float(np.max(future_closes)) if len(future_closes) > 0 else entry * (1 + buy_thresh)
                sl_abs      = float(np.min(future_closes)) if len(future_closes) > 0 else entry * (1 - sell_thresh)
                tp_pct      = (tp_abs - entry) / entry          # e.g. +0.03
                sl_pct      = (sl_abs - entry) / entry          # e.g. -0.015
                net_pct     = tp_pct + sl_pct                   # net reward-risk %
            elif label == 2:  # SELL
                tp_abs      = float(np.min(future_closes)) if len(future_closes) > 0 else entry * (1 - sell_thresh)
                sl_abs      = float(np.max(future_closes)) if len(future_closes) > 0 else entry * (1 + buy_thresh)
                tp_pct      = (entry - tp_abs) / entry          # positive for SELL
                sl_pct      = (entry - sl_abs) / entry          # negative for SELL
                net_pct     = tp_pct + sl_pct
            else:  # HOLD
                tp_pct      = atr / entry if entry > 0 else 0.01
                sl_pct      = -atr / entry if entry > 0 else -0.01
                net_pct     = 0.0

            bars_to_entry = compute_optimal_entry_bar(i, closes, label)

            X_price.append(window)
            X_sent.append(sent_vec)
            y_cls.append(label)
            # [0]=entry(always 0.0), [1]=sl_pct, [2]=tp_pct, [3]=net_pct, [4]=bars
            y_reg.append([0.0, sl_pct, tp_pct, net_pct, bars_to_entry])

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
