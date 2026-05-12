import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import asyncpg
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Columns fetched from the indicators table. Some of these are kept for
# transformation (cyclical encoding) and not exposed directly to the model.
DB_INDICATOR_COLS = [
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
    "fomc_days",
    "cpi_days",
    "nfp_days",
    "eps_surprise_pct",
]

# Features dropped from the model input (still queried from DB but not used):
#   • is_trading_day               — constant for all rows (all bars are trading days)
#   • above_sma50/200, higher_high, — binary flags redundant with continuous
#     lower_low, volume_above_avg,    distances (price_sma50_dist, roc_5/10, etc.)
#     high_vol_regime
#   • vix_level, vix_change        — market-wide; same value for all tickers
#                                    on the same bar, weakens per-ticker signal
#   • day_of_week, day_of_month,   — replaced with sin/cos cyclical encoding so
#     month                          the model can learn periodic structure
_DROPPED_FEATURES = {
    "is_trading_day",
    "above_sma50", "above_sma200",
    "higher_high", "lower_low",
    "volume_above_avg", "high_vol_regime",
    "vix_level", "vix_change",
    "day_of_week", "day_of_month", "month",
}

# Cyclical replacements computed in _build_ticker from the raw integer columns
CYCLICAL_FEATURES = ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos"]

# Final ordered feature list the model actually sees
INDICATOR_COLS = [c for c in DB_INDICATOR_COLS if c not in _DROPPED_FEATURES] + CYCLICAL_FEATURES

SENTIMENT_COLS = ["avg_positive", "avg_negative", "avg_neutral", "avg_compound"]

FEATURE_COLS = OHLCV_COLS + INDICATOR_COLS

SEQUENCE_LEN = 60

REGRESSION_TARGETS = [
    "entry_price",    # [0] % offset from current close → convert to $ at inference
    "stop_loss",      # [1] signed % from entry
    "take_profit",    # [2] signed % from entry
    "net_profit",     # [3] net reward-risk %
    "bars_to_entry",  # [4] predicted bars until entry fills
]

# Separate discrete timing head: cleaner CE loss than regressing a float
#   0=bar+1  1=bar+2  2=bar+3  3=no-fill/HOLD
TIMING_BUCKETS = 4
ENTRY_WINDOW   = 3  # short forward window for entry labels (learnable from momentum)



def compute_entry_labels(
    current_close: float,
    label: int,
    early_window: np.ndarray,  # closes[i+1 .. i+1+ENTRY_WINDOW]
) -> tuple:
    """
    Compute (entry_offset_pct, bars_to_entry_float, timing_class) using only
    a tight ENTRY_WINDOW (3-bar) forward look — narrow enough that current-bar
    momentum / volatility carries genuine predictive signal.

    entry_offset_pct:
        BUY  → % offset to the lowest close in the window (negative = buy dip)
        SELL → % offset to the highest close in the window (positive = sell spike)
        HOLD → 0.0

    bars_to_entry_float:
        Index (1-based) of the optimal bar within the window. HOLD → 0.0.

    timing_class (int 0-3):
        0 = bar+1, 1 = bar+2, 2 = bar+3, 3 = HOLD/no-fill
        Mirrors bars_to_entry as a discrete class for the timing head.
    """
    if label == 0 or len(early_window) == 0:
        return 0.0, 0.0, 3  # HOLD

    if label == 1:          # BUY — target the dip
        idx = int(np.argmin(early_window))
    else:                   # SELL — target the spike
        idx = int(np.argmax(early_window))

    best_price       = float(early_window[idx])
    entry_offset_pct = (best_price - current_close) / current_close
    # Clip: 2% max — prevents rare gap-opens from dominating MSE
    entry_offset_pct = float(np.clip(entry_offset_pct, -0.02, 0.02))
    bars_to_entry    = float(idx + 1)          # 1-based bar offset
    timing_class     = min(idx, TIMING_BUCKETS - 2)  # cap at class 2

    return entry_offset_pct, bars_to_entry, timing_class


class DatasetBuilder:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def build(
        self,
        tickers: List[str],
        interval: str = "1d",
        sequence_len: int = SEQUENCE_LEN,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Build the full dataset with per-ticker Z-score normalisation.
        Returns (X_price, X_sentiment, y_class, y_regression, y_timing, scaler_params).
        scaler_params is a dict  ticker → {mean: [...], std: [...]}.
        """
        all_price, all_sent, all_cls, all_reg, all_timing = [], [], [], [], []
        scaler_params: dict = {}

        # NOTE: this builder returns RAW (un-normalised) price arrays. Callers
        # are responsible for normalisation so they can fit the scaler on a
        # train slice only (no val/test leakage). Inference path uses the
        # saved per-ticker scaler from the trained checkpoint.
        for ticker in tickers:
            try:
                p, s, c, r, t = await self._build_ticker(ticker, interval, sequence_len)
                if p is not None and len(p) > 0:
                    all_price.append(p)
                    all_sent.append(s)
                    all_cls.append(c)
                    all_reg.append(r)
                    all_timing.append(t)
                    logger.info(f"✅ {ticker}: {len(p)} sequences")
            except Exception as e:
                logger.error(f"❌ {ticker}: {e}")

        if not all_price:
            raise ValueError("No data built — run ingest + sentiment fetch first.")

        X_price      = np.concatenate(all_price,  axis=0).astype(np.float32)
        X_sentiment  = np.concatenate(all_sent,   axis=0).astype(np.float32)
        y_class      = np.concatenate(all_cls,    axis=0).astype(np.int64)
        y_regression = np.concatenate(all_reg,    axis=0).astype(np.float32)
        y_timing_out = np.concatenate(all_timing, axis=0).astype(np.int64)

        logger.info(
            f"✅ Dataset: {X_price.shape[0]} sequences | "
            f"Buy={int((y_class == 1).sum())} "
            f"Sell={int((y_class == 2).sum())} "
            f"Hold={int((y_class == 0).sum())}"
        )
        return X_price, X_sentiment, y_class, y_regression, y_timing_out, scaler_params

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
            return None, None, None, None, None

        async with self.pool.acquire() as conn:
            ind_rows = await conn.fetch(
                """
                SELECT timestamp, """
                + ", ".join(DB_INDICATOR_COLS)
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
            pd.DataFrame([dict(r) for r in ind_rows])[["timestamp"] + DB_INDICATOR_COLS]
            if ind_rows
            else pd.DataFrame()
        )

        if not df_ind.empty:
            df = pd.merge(df_ohlcv, df_ind, on="timestamp", how="left")
        else:
            df = df_ohlcv.copy()
            for col in DB_INDICATOR_COLS:
                df[col] = np.nan

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Cyclical encoding: integer time features → (sin, cos) pairs so the
        # model learns periodic structure (Mon→Sun wrap, Dec→Jan wrap, etc.)
        # Falls back to 0 when source columns are missing.
        dow = df["day_of_week"].fillna(0).astype(float)   if "day_of_week" in df.columns else 0.0
        dom = df["day_of_month"].fillna(1).astype(float)  if "day_of_month" in df.columns else 1.0
        mon = df["month"].fillna(1).astype(float)         if "month" in df.columns else 1.0
        df["dow_sin"]   = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"]   = np.cos(2 * np.pi * dow / 7.0)
        df["dom_sin"]   = np.sin(2 * np.pi * dom / 31.0)
        df["dom_cos"]   = np.cos(2 * np.pi * dom / 31.0)
        df["month_sin"] = np.sin(2 * np.pi * mon / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * mon / 12.0)

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

        X_price, X_sent, y_cls, y_reg, y_timing = [], [], [], [], []

        lookahead   = settings.LOOKAHEAD_WINDOW
        buy_thresh  = settings.BUY_THRESHOLD
        sell_thresh = settings.SELL_THRESHOLD

        # Leave enough future bars for the lookahead window
        for i in range(sequence_len, len(df) - lookahead):
            window        = prices[i - sequence_len : i]
            current_close = closes[i]

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

            # ── Regression targets — all as % deviations from current close ──
            # All price-level targets stored as decimal % so gradients stay in
            # the same range as classification loss. Multiply by current_close
            # at inference to recover dollar values.
            entry = current_close
            if entry <= 0:
                continue

            if label == 1:  # BUY
                tp_abs  = float(np.max(future_closes)) if len(future_closes) > 0 else entry * (1 + buy_thresh)
                sl_abs  = float(np.min(future_closes)) if len(future_closes) > 0 else entry * (1 - sell_thresh)
                tp_pct  = (tp_abs - entry) / entry
                sl_pct  = (sl_abs - entry) / entry
                net_pct = tp_pct + sl_pct
            elif label == 2:  # SELL
                tp_abs  = float(np.min(future_closes)) if len(future_closes) > 0 else entry * (1 - sell_thresh)
                sl_abs  = float(np.max(future_closes)) if len(future_closes) > 0 else entry * (1 + buy_thresh)
                tp_pct  = (entry - tp_abs) / entry
                sl_pct  = (entry - sl_abs) / entry
                net_pct = tp_pct + sl_pct
            else:  # HOLD — no trade
                tp_pct  = 0.0
                sl_pct  = 0.0
                net_pct = 0.0

            # ── Entry price + timing — use tight 3-bar window ────────────────
            # Short window (ENTRY_WINDOW=3 bars) so current-bar momentum /
            # volatility carries genuine predictive signal. Full-lookahead
            # argmin was unpredictable hindsight; next 3 bars is learnable.
            early_window = closes[i + 1 : i + 1 + ENTRY_WINDOW]
            entry_offset_pct, bars_to_entry, timing_class = compute_entry_labels(
                current_close, label, early_window
            )

            X_price.append(window)
            X_sent.append(sent_vec)
            y_cls.append(label)
            # [0]=entry_offset_pct  [1]=sl_pct  [2]=tp_pct  [3]=net_pct  [4]=bars
            y_reg.append([entry_offset_pct, sl_pct, tp_pct, net_pct, bars_to_entry])
            y_timing.append(timing_class)

        return (
            np.array(X_price,  dtype=np.float32),
            np.array(X_sent,   dtype=np.float32),
            np.array(y_cls,    dtype=np.int64),
            np.array(y_reg,    dtype=np.float32),
            np.array(y_timing, dtype=np.int64),
        )

    def _normalise(
        self,
        X: np.ndarray,
        fit_slice: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Z-score normalise X using statistics computed from a slice (typically
        the chronological train portion only) — prevents val/test data from
        leaking into the scaler, which previously made the gen-gap invisible.

        fit_slice = (start, end) of the rows used to FIT mean/std; transform
        is then applied to the entire array. If None, fits on the full array
        (legacy behaviour).
        """
        N, T, F = X.shape
        if fit_slice is None:
            fit_data = X.reshape(-1, F)
        else:
            s, e = fit_slice
            fit_data = X[s:e].reshape(-1, F) if e > s else X.reshape(-1, F)

        mean = fit_data.mean(axis=0)
        # Use a meaningful std floor (1e-2) so features that are near-constant
        # in the fit slice don't blow up to astronomical values when a non-zero
        # appears later. With 1e-8 floor, a 5% eps_surprise → 5_000_000 → NaN.
        std = np.maximum(fit_data.std(axis=0), 1e-2)
        X_norm = (X.reshape(-1, F) - mean).reshape(N, T, F) / std
        # Final safety cap — prevents any remaining outlier from cascading into
        # transformer overflow on small fold subsets.
        X_norm = np.clip(X_norm, -10.0, 10.0)
        return X_norm, {"mean": mean.tolist(), "std": std.tolist()}
