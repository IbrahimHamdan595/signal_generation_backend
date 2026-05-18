import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import asyncpg
import logging

from app.core.config import settings
from app.core.asset_class import is_fx

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

# FX-only feature additions appended to the per-bar feature row when the
# ticker is an FX/metal pair, left out for equities (which keep n_features=51).
#
# Two groups:
#   GLOBAL  — same value for all FX pairs on the same date; sourced from
#             fx_macro_features (refreshed daily by the scheduler).
#   PER_PAIR — computed per-bar per-pair inside _build_ticker from OHLCV.
FX_MACRO_COLS         = ["dxy_ret_5d", "vix_close", "yield_spread_10_2", "us_2y_yield"]
FX_PER_PAIR_VOL_COLS  = ["realized_vol_20d", "vol_of_vol_5d"]

# Multi-timeframe: most-recent-4h indicator snapshot joined to each daily bar.
# Sourced from `indicators` table where interval='4h' (backfilled by
# scripts/backfill_fx_4h.py for the last ~2 years per pair).
#
# Experimentally, naively appending these 4 features to a daily-context model
# REGRESSED test Sharpe (0.29 → 0.19) — likely because the simple "append" loses
# the sequence structure that MTF needs. The columns are still computed by the
# dataset builder (the join runs harmlessly) but are NOT in FEATURE_COLS_FX, so
# the model doesn't see them. To re-enable, append `+ FX_4H_COLS` below.
FX_4H_COLS = ["rsi_14_4h", "macd_hist_4h", "atr_14_4h", "roc_5_4h"]

# Base feature column list used for equity samples (n_features = 51).
FEATURE_COLS = OHLCV_COLS + INDICATOR_COLS

# FX feature column list adds 4 macro globals + 2 per-pair vol features at the
# end (n_features = 57). Equities stays at 51.
FEATURE_COLS_FX = FEATURE_COLS + FX_MACRO_COLS + FX_PER_PAIR_VOL_COLS


def feature_cols_for(ticker: str) -> list[str]:
    """Return the feature column list appropriate for `ticker`'s asset class."""
    return FEATURE_COLS_FX if is_fx(ticker) else FEATURE_COLS

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

    async def _build_ticker(
        self,
        ticker: str,
        interval: str,
        sequence_len: int,
        buy_thresh: Optional[float] = None,
        sell_thresh: Optional[float] = None,
        lookahead: Optional[int] = None,
        barrier_atr_mult: Optional[float] = None,
    ):
        """Build a single ticker's training arrays.

        `buy_thresh` / `sell_thresh` / `lookahead` override `settings` for
        this build only — used by per-asset-class training (FX needs smaller
        barriers + longer lookahead than the equity defaults).

        `barrier_atr_mult` activates volatility-normalised labelling: when
        set, the triple-barrier thresholds become `barrier_atr_mult × ATR(t) / close(t)`
        per bar instead of the fixed `buy_thresh` / `sell_thresh`. This makes
        label density consistent across pairs with very different volatility
        profiles (EURUSD ~0.5% ATR vs XAUUSD ~$30/oz ≈ 1.5%) — the textbook
        FX-ML fix. `buy_thresh`/`sell_thresh` then serve only as a fallback
        floor for bars where ATR is missing/zero.
        """
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

        # ── FX-only global macro features ───────────────────────────────────
        # Left-join fx_macro_features on bar date so the model sees DXY 5d
        # return, VIX level, and yield-curve spread alongside the per-pair
        # indicators. Equity samples skip this join — their checkpoint is
        # trained on 51 features, not 54.
        ticker_is_fx = is_fx(ticker)
        if ticker_is_fx:
            async with self.pool.acquire() as conn:
                macro_rows = await conn.fetch(
                    """
                    SELECT date, dxy_ret_5d, vix_close, yield_spread_10_2, us_2y_yield
                    FROM fx_macro_features
                    ORDER BY date ASC
                    """
                )
            if macro_rows:
                df_macro = pd.DataFrame([dict(r) for r in macro_rows])
                # Normalise both join keys to tz-naive datetime64[ns] at
                # midnight so pandas accepts the merge — OHLCV timestamps
                # arrive from asyncpg as datetime64[us, UTC] while macro
                # dates come in as datetime64[s] without tz, which raises
                # "you are trying to merge on different dtypes" otherwise.
                df_macro["date"] = (
                    pd.to_datetime(df_macro["date"])
                    .dt.tz_localize(None)
                    .astype("datetime64[ns]")
                )
                ts_naive = pd.to_datetime(df["timestamp"])
                if getattr(ts_naive.dt, "tz", None) is not None:
                    ts_naive = ts_naive.dt.tz_localize(None)
                df["__date"] = ts_naive.dt.normalize().astype("datetime64[ns]")
                df = df.merge(df_macro, left_on="__date", right_on="date", how="left")
                df = df.drop(columns=["__date", "date"])
            else:
                # Macro table empty (backfill not run yet) — zero-fill so
                # training still produces a clean feature vector of the right
                # width.
                logger.warning(f"⚠️  {ticker}: fx_macro_features is empty — zero-filling macros")
                for col in FX_MACRO_COLS:
                    df[col] = 0.0

            # ── Per-pair realized-volatility features ────────────────────────
            # std of log returns over a 20-bar window, plus the 5-bar std of
            # that series (vol-of-vol). Captures volatility regime / regime
            # change — both well-documented FX persistence signals.
            log_ret = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
            df["realized_vol_20d"] = log_ret.rolling(20, min_periods=5).std().fillna(0.0)
            df["vol_of_vol_5d"]    = df["realized_vol_20d"].rolling(5, min_periods=2).std().fillna(0.0)

            # ── 4h multi-timeframe features ──────────────────────────────────
            # For each daily bar at timestamp T, snap to the most-recent 4h
            # indicator row at or before T. yfinance only gives ~2 years of 4h
            # history, so older daily samples will have NaNs here — they get
            # zero-padded by the model's nan_to_num path, equivalent to "no
            # MTF context".
            async with self.pool.acquire() as conn:
                rows_4h = await conn.fetch(
                    """
                    SELECT timestamp, rsi_14, macd_histogram, atr_14, roc_5
                    FROM indicators
                    WHERE ticker = $1 AND interval = '4h'
                    ORDER BY timestamp ASC
                    """,
                    ticker,
                )
            if rows_4h:
                df_4h = pd.DataFrame([dict(r) for r in rows_4h]).rename(columns={
                    "rsi_14":          "rsi_14_4h",
                    "macd_histogram":  "macd_hist_4h",
                    "atr_14":          "atr_14_4h",
                    "roc_5":           "roc_5_4h",
                })
                # Normalise timestamps to tz-naive ns so merge_asof accepts
                # both sides (mirrors the macro-feature merge fix earlier).
                ts_4h = pd.to_datetime(df_4h["timestamp"])
                if getattr(ts_4h.dt, "tz", None) is not None:
                    ts_4h = ts_4h.dt.tz_localize(None)
                df_4h["timestamp"] = ts_4h.astype("datetime64[ns]")
                df_4h = df_4h.sort_values("timestamp").reset_index(drop=True)

                ts_day = pd.to_datetime(df["timestamp"])
                if getattr(ts_day.dt, "tz", None) is not None:
                    ts_day = ts_day.dt.tz_localize(None)
                df["timestamp"] = ts_day.astype("datetime64[ns]")
                df = df.sort_values("timestamp").reset_index(drop=True)

                # Backward merge_asof: for each daily ts, take the latest 4h
                # row with ts <= daily ts. Direction='backward' is the default.
                df = pd.merge_asof(
                    df, df_4h,
                    on="timestamp",
                    direction="backward",
                    allow_exact_matches=True,
                )
            else:
                logger.warning(f"⚠️  {ticker}: no 4h indicators yet — zero-filling MTF cols")
                for col in FX_4H_COLS:
                    df[col] = 0.0

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

        # Use the asset-class-appropriate feature column list — FX adds 3 macros.
        feat_cols = FEATURE_COLS_FX if ticker_is_fx else FEATURE_COLS
        df[feat_cols] = df[feat_cols].ffill().bfill()

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

        prices     = df[feat_cols].values
        closes     = df["close"].values
        timestamps = df["timestamp"].values

        # ATR series — used by the volatility-normalised labelling branch.
        # When `barrier_atr_mult` is set, each bar's threshold becomes
        # `mult × atr(t) / close(t)`. This is the textbook fix for label
        # density inconsistency across pairs with different daily volatility.
        atr_series = df["atr_14"].astype(float).values if "atr_14" in df.columns else None

        X_price, X_sent, y_cls, y_reg, y_timing = [], [], [], [], []

        # Per-build overrides (asset-class-specific thresholds) win over
        # the global settings defaults.
        lookahead   = lookahead   if lookahead   is not None else settings.LOOKAHEAD_WINDOW
        buy_thresh  = buy_thresh  if buy_thresh  is not None else settings.BUY_THRESHOLD
        sell_thresh = sell_thresh if sell_thresh is not None else settings.SELL_THRESHOLD

        # Floor for ATR-normalised thresholds — prevents pegged pairs
        # (USDHKD-like) with near-zero ATR from getting a 0% barrier that
        # fires on every bar.
        _BARRIER_FLOOR = 0.003

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

            # ── Triple-barrier labeling ──────────────────────────────────────
            # Volatility-normalised when `barrier_atr_mult` is provided:
            # threshold(t) = mult × atr(t) / close(t), bounded below by the
            # floor to avoid 0% barriers on pegged pairs. Otherwise falls
            # back to the fixed buy_thresh / sell_thresh per-build overrides.
            if barrier_atr_mult is not None and atr_series is not None:
                atr_t = float(atr_series[i]) if not np.isnan(atr_series[i]) else 0.0
                if atr_t > 0 and current_close > 0:
                    bar_buy_thresh  = max(barrier_atr_mult * atr_t / current_close, _BARRIER_FLOOR)
                    bar_sell_thresh = bar_buy_thresh   # symmetric
                else:
                    bar_buy_thresh  = buy_thresh
                    bar_sell_thresh = sell_thresh
            else:
                bar_buy_thresh  = buy_thresh
                bar_sell_thresh = sell_thresh

            future_closes = closes[i + 1 : i + 1 + lookahead]
            label = 0  # HOLD default
            hit_idx = len(future_closes) - 1  # full window if no barrier
            for j, fc in enumerate(future_closes):
                ret = (fc - current_close) / current_close
                if ret >= bar_buy_thresh:
                    label = 1   # BUY — upper barrier hit first
                    hit_idx = j
                    break
                if ret <= -bar_sell_thresh:
                    label = 2   # SELL — lower barrier hit first
                    hit_idx = j
                    break

            # ── Regression targets — all as % deviations from current close ──
            # Truncate at the barrier-hit index so the regression head learns
            # SL/TP magnitudes consistent with the trade window the classifier
            # actually identified — not post-exit excursions that bloat the
            # noise in sl_pct labels.
            entry = current_close
            if entry <= 0:
                continue

            trade_window = future_closes[: hit_idx + 1] if len(future_closes) > 0 else future_closes

            if label == 1:  # BUY
                tp_abs  = float(np.max(trade_window)) if len(trade_window) > 0 else entry * (1 + buy_thresh)
                sl_abs  = float(np.min(trade_window)) if len(trade_window) > 0 else entry * (1 - sell_thresh)
                tp_pct  = (tp_abs - entry) / entry
                sl_pct  = (sl_abs - entry) / entry
                net_pct = tp_pct + sl_pct
            elif label == 2:  # SELL
                tp_abs  = float(np.min(trade_window)) if len(trade_window) > 0 else entry * (1 - sell_thresh)
                sl_abs  = float(np.max(trade_window)) if len(trade_window) > 0 else entry * (1 + buy_thresh)
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
