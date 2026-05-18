"""Unit tests for the ATR-mechanical SL/TP level helper.

`compute_atr_levels` is a pure function (no DB, no model) shared by both
the ML signal generator and the Donchian rule-based strategy. These tests
lock down its math so any future tweak that breaks the 2:1 RR geometry,
the asymmetric sign convention, or the fallback path fails CI immediately.
"""

import pytest

from app.strategies.atr_levels import (
    compute_atr_levels,
    DEFAULT_FALLBACK_SL,
    DEFAULT_FALLBACK_TP,
    DEFAULT_K_SL,
    DEFAULT_K_TP,
    MIN_SL_FLOOR,
)


# ── BUY path ───────────────────────────────────────────────────────────────────

def test_buy_with_atr_uses_atr_scaled_levels():
    # AAPL-ish: $200 close, $2.50 ATR → SL at 1.5×ATR=$3.75 below, TP at 3×ATR=$7.50 above
    r = compute_atr_levels(atr_14=2.5, current_close=200.0, action="BUY")
    assert r["used_atr"] is True
    assert r["entry"] == pytest.approx(200.0)
    assert r["stop_loss"] == pytest.approx(200.0 - 3.75)
    assert r["take_profit"] == pytest.approx(200.0 + 7.50)
    # 2:1 reward:risk by construction
    assert r["abs_tp_pct"] == pytest.approx(2.0 * r["abs_sl_pct"])


def test_buy_falls_back_to_fixed_pct_when_atr_missing():
    r = compute_atr_levels(atr_14=None, current_close=200.0, action="BUY")
    assert r["used_atr"] is False
    assert r["abs_sl_pct"] == DEFAULT_FALLBACK_SL
    assert r["abs_tp_pct"] == DEFAULT_FALLBACK_TP
    # Fallback should still produce 2:1 RR (3.0% / 1.5%)
    assert r["abs_tp_pct"] == pytest.approx(2.0 * r["abs_sl_pct"])
    assert r["stop_loss"] == pytest.approx(200.0 * (1 - DEFAULT_FALLBACK_SL))
    assert r["take_profit"] == pytest.approx(200.0 * (1 + DEFAULT_FALLBACK_TP))


# ── SELL path ──────────────────────────────────────────────────────────────────

def test_sell_inverts_sl_and_tp_relative_to_buy():
    buy  = compute_atr_levels(atr_14=2.5, current_close=200.0, action="BUY")
    sell = compute_atr_levels(atr_14=2.5, current_close=200.0, action="SELL")

    # Same magnitudes — only the sign flips
    assert sell["abs_sl_pct"] == buy["abs_sl_pct"]
    assert sell["abs_tp_pct"] == buy["abs_tp_pct"]

    # SELL: SL is ABOVE entry, TP is BELOW entry
    assert sell["stop_loss"] > sell["entry"]
    assert sell["take_profit"] < sell["entry"]
    # BUY: SL is BELOW, TP is ABOVE
    assert buy["stop_loss"] < buy["entry"]
    assert buy["take_profit"] > buy["entry"]


# ── HOLD path ─────────────────────────────────────────────────────────────────

def test_hold_returns_entry_for_both_sl_and_tp():
    r = compute_atr_levels(atr_14=2.5, current_close=200.0, action="HOLD")
    assert r["stop_loss"]   == pytest.approx(200.0)
    assert r["take_profit"] == pytest.approx(200.0)


def test_hold_case_insensitive():
    r = compute_atr_levels(atr_14=2.5, current_close=200.0, action="hold")
    assert r["stop_loss"]   == pytest.approx(200.0)
    assert r["take_profit"] == pytest.approx(200.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_pegged_pair_floor_applied():
    # USDHKD-like: peg keeps ATR tiny so k×ATR/close would be ~0.0001
    # The MIN_SL_FLOOR (~0.5%) prevents barriers that fire on every wiggle.
    r = compute_atr_levels(atr_14=0.0001, current_close=7.8, action="BUY")
    assert r["abs_sl_pct"] == pytest.approx(MIN_SL_FLOOR)


def test_zero_atr_treated_as_missing():
    # Defensive: ATR=0 (sometimes seen on fresh-listing tickers) should not
    # produce divide-by-zero or a 0% barrier. Falls back to fixed percentages.
    r = compute_atr_levels(atr_14=0.0, current_close=100.0, action="BUY")
    assert r["used_atr"] is False
    assert r["abs_sl_pct"] == DEFAULT_FALLBACK_SL


def test_negative_atr_treated_as_missing():
    r = compute_atr_levels(atr_14=-1.5, current_close=100.0, action="BUY")
    assert r["used_atr"] is False


def test_custom_k_multipliers_override_defaults():
    # Tighter stops + same target → higher RR
    r = compute_atr_levels(atr_14=2.0, current_close=100.0, action="BUY", k_sl=1.0, k_tp=4.0)
    assert r["abs_sl_pct"] == pytest.approx(2.0 / 100.0)        # 1.0 × 2.0 / 100 = 2%
    assert r["abs_tp_pct"] == pytest.approx(8.0 / 100.0)        # 4.0 × 2.0 / 100 = 8%
    # 4:1 RR with these multipliers
    assert r["abs_tp_pct"] == pytest.approx(4.0 * r["abs_sl_pct"])


def test_default_multipliers_produce_2to1_rr():
    # Sanity-check the documented invariant: defaults give 2:1 RR
    assert DEFAULT_K_TP / DEFAULT_K_SL == pytest.approx(2.0)


def test_returns_typed_keys():
    # Catches a regression if anyone renames keys without updating callers
    r = compute_atr_levels(atr_14=1.0, current_close=100.0, action="BUY")
    for key in ("entry", "stop_loss", "take_profit", "abs_sl_pct", "abs_tp_pct", "used_atr", "note"):
        assert key in r, f"missing key {key!r} in compute_atr_levels result"
