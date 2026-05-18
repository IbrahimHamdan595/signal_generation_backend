"""ATR-based stop-loss / take-profit level calculation.

Mechanical SL/TP derived from current ATR — `k_sl * ATR` below/above entry
for the stop, `k_tp * ATR` for the target. Default k_sl=1.5, k_tp=3.0 gives
a fixed 2:1 reward:risk geometry that matches the executor's MIN_RR_RATIO=2.0
filter, so every directional signal has favorable RR by construction.

Used by both the ML signal generator (in ml_service.predict_ticker) and the
rule-based Donchian strategy (donchian.py). The math is asset-agnostic and
the same constants work across stocks and FX/metals.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class AtrLevels(TypedDict):
    """Result of compute_atr_levels()."""
    entry:           float
    stop_loss:       float
    take_profit:     float
    abs_sl_pct:      float   # |SL distance| / entry, fraction
    abs_tp_pct:      float   # |TP distance| / entry, fraction
    used_atr:        bool    # True if ATR-derived, False if fallback used
    note:            str


# Public defaults — exported so strategy authors who want to tweak the RR
# geometry can do so consistently across the codebase.
DEFAULT_K_SL          = 1.5      # SL at 1.5× ATR
DEFAULT_K_TP          = 3.0      # TP at 3.0× ATR  (fixed 2:1 RR by construction)
DEFAULT_FALLBACK_SL   = 0.015    # 1.5% when ATR is missing/zero — approximates a typical S&P daily ATR fraction
DEFAULT_FALLBACK_TP   = 0.030    # 3.0% (same 2:1 geometry as the ATR path)
MIN_SL_FLOOR          = 0.005    # 0.5% — prevents pegged-currency pairs (USDHKD-like) from getting 0% SL


def compute_atr_levels(
    atr_14:        Optional[float],
    current_close: float,
    action:        str,
    k_sl:          float = DEFAULT_K_SL,
    k_tp:          float = DEFAULT_K_TP,
) -> AtrLevels:
    """Return entry/SL/TP dollars + their fractional magnitudes.

    `action` is one of "BUY", "SELL", or "HOLD" (case-insensitive). HOLD
    returns SL = TP = entry so the executor's "no-trade" branch fires.

    When `atr_14` is None or <= 0 (e.g. the indicator hasn't computed yet
    for a brand-new ticker), falls back to fixed 1.5% / 3.0% which keeps
    the RR ratio identical to the ATR path. The result still has
    `used_atr=False` so callers can log the fallback for diagnostics.
    """
    act = (action or "HOLD").upper()
    entry = float(current_close)

    if atr_14 and atr_14 > 0 and entry > 0:
        abs_sl_pct = max((k_sl * float(atr_14)) / entry, MIN_SL_FLOOR)
        abs_tp_pct = (k_tp * float(atr_14)) / entry
        used_atr = True
        note = f"k_sl={k_sl}x ATR, k_tp={k_tp}x ATR"
    else:
        abs_sl_pct = DEFAULT_FALLBACK_SL
        abs_tp_pct = DEFAULT_FALLBACK_TP
        used_atr = False
        note = "ATR unavailable — used fixed-pct fallback"

    if act == "BUY":
        sl = entry * (1 - abs_sl_pct)
        tp = entry * (1 + abs_tp_pct)
    elif act == "SELL":
        sl = entry * (1 + abs_sl_pct)
        tp = entry * (1 - abs_tp_pct)
    else:   # HOLD
        sl = entry
        tp = entry

    return {
        "entry":       round(entry, 6),
        "stop_loss":   round(sl,    6),
        "take_profit": round(tp,    6),
        "abs_sl_pct":  abs_sl_pct,
        "abs_tp_pct":  abs_tp_pct,
        "used_atr":    used_atr,
        "note":        note,
    }
