"""Unit tests for the comparison-report aggregation math.

`_aggregate_one_source` is the pure function behind GET /strategy-comparison —
takes a list of trade rows and returns per-source totals, win rate, equity
curve, Sharpe, and max drawdown. No DB involved at this layer.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.api.routes.reports import _aggregate_one_source, _cost_bps_for


# ── _cost_bps_for ─────────────────────────────────────────────────────────────

def test_cost_bps_uses_asset_class_lookup():
    # AAPL is an equity → 5 bps round-trip
    assert _cost_bps_for("AAPL") == 5.0
    # EURUSD is fx_major → 1 bps
    assert _cost_bps_for("EURUSD") == 1.0
    # XAUUSD is fx_metal → 3 bps
    assert _cost_bps_for("XAUUSD") == 3.0


def test_cost_bps_strips_broker_suffix():
    # Brokers add suffixes like `.r` or `.c` to symbol names; the lookup
    # should still resolve to the base ticker's asset class
    assert _cost_bps_for("EURUSD.r") == 1.0
    assert _cost_bps_for("XAUUSD.cent") == 3.0


def test_cost_bps_falls_back_to_equity_for_unknown_symbol():
    assert _cost_bps_for("???") == 5.0   # unknown asset class → conservative
    assert _cost_bps_for("") == 5.0


# ── _aggregate_one_source ─────────────────────────────────────────────────────

def _trade(pnl: float, *, ticker: str = "AAPL", volume: float = 1.0,
           fill_price: float = 100.0, closed_at: datetime | None = None) -> dict:
    """Make a fixture row matching the SELECT in the reports endpoint."""
    return {
        "pnl":        pnl,
        "volume":     volume,
        "fill_price": fill_price,
        "ticker":     ticker,
        "symbol":     ticker,
        "closed_at":  closed_at,
        "created_at": closed_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


def test_empty_input_returns_zeroed_aggregate():
    r = _aggregate_one_source([])
    assert r["total_trades"]      == 0
    assert r["wins"]              == 0
    assert r["losses"]            == 0
    assert r["win_rate"]          == 0.0
    assert r["total_pnl_usd"]     == 0.0
    assert r["total_pnl_usd_adj"] == 0.0
    assert r["sharpe_per_trade"]  == 0.0
    assert r["equity_curve"]      == []


def test_wins_and_losses_counted_correctly():
    # win/loss counts are computed on the COST-ADJUSTED PnL, so a $0-raw trade
    # still registers as a small loss after broker fees are subtracted.
    trades = [_trade(+10.0), _trade(-5.0), _trade(+7.0), _trade(0.0)]
    r = _aggregate_one_source(trades)
    assert r["total_trades"] == 4
    assert r["wins"]         == 2
    assert r["losses"]       == 2   # -5 raw and 0 raw (which becomes -cost adjusted)
    assert r["win_rate"]     == pytest.approx(0.5)


def test_total_pnl_sums_correctly_raw_and_adjusted():
    # 2 trades, AAPL (equity → 5 bps), each $100 notional → $0.05 cost per trade
    trades = [_trade(+10.0), _trade(-3.0)]
    r = _aggregate_one_source(trades)
    assert r["total_pnl_usd"] == pytest.approx(7.0)
    # cost = 100 * 1.0 * 5/10000 = 0.05 per trade → 0.10 total
    assert r["total_pnl_usd_adj"] == pytest.approx(7.0 - 0.10)


def test_fx_costs_lower_than_equity():
    # Same dollar PnL on equity vs FX, FX should be cheaper to trade
    eq_trades = [_trade(+10.0, ticker="AAPL",   fill_price=100.0, volume=1.0)]
    fx_trades = [_trade(+10.0, ticker="EURUSD", fill_price=1.0,   volume=100.0)]   # same notional
    r_eq = _aggregate_one_source(eq_trades)
    r_fx = _aggregate_one_source(fx_trades)
    cost_eq = r_eq["total_pnl_usd"] - r_eq["total_pnl_usd_adj"]
    cost_fx = r_fx["total_pnl_usd"] - r_fx["total_pnl_usd_adj"]
    assert cost_fx < cost_eq   # 1 bps FX < 5 bps equity


def test_equity_curve_is_chronological_and_cumulative():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        _trade(+5.0,  closed_at=t0 + timedelta(days=2)),
        _trade(-2.0,  closed_at=t0 + timedelta(days=1)),   # earlier
        _trade(+3.0,  closed_at=t0 + timedelta(days=3)),
    ]
    r = _aggregate_one_source(trades)
    curve = r["equity_curve"]
    assert len(curve) == 3
    # Sorted by timestamp ascending
    assert curve[0]["cum_pnl_raw"] == pytest.approx(-2.0)
    assert curve[1]["cum_pnl_raw"] == pytest.approx(-2.0 + 5.0)
    assert curve[2]["cum_pnl_raw"] == pytest.approx(-2.0 + 5.0 + 3.0)


def test_max_drawdown_walks_adjusted_curve():
    # Curve: +10 → +15 → +5 → +12 → drawdown of -10 from the +15 peak
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        _trade(+10.0, closed_at=t0 + timedelta(days=1)),
        _trade(+5.0,  closed_at=t0 + timedelta(days=2)),
        _trade(-10.0, closed_at=t0 + timedelta(days=3)),
        _trade(+7.0,  closed_at=t0 + timedelta(days=4)),
    ]
    r = _aggregate_one_source(trades)
    # max DD is the most-negative trough relative to the running peak.
    # Costs are tiny ($0.05/trade) so the raw geometry dominates.
    assert r["max_drawdown_usd"] < 0
    assert r["max_drawdown_usd"] >= -10.5   # within cost noise of -10


def test_sharpe_is_zero_when_all_pnls_equal():
    # Zero variance → Sharpe is 0 (we guard against div-by-zero)
    trades = [_trade(+5.0) for _ in range(4)]
    r = _aggregate_one_source(trades)
    assert r["sharpe_per_trade"] == pytest.approx(0.0)
