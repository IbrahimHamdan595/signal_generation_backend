"""Strategy comparison reports — backend.

Aggregates trade executions JOIN signals by source so the comparison tab
can show per-strategy metrics, equity curves, and per-trade drill-downs:

  GET /api/v1/reports/strategy-comparison?since={iso}
  GET /api/v1/reports/strategy-trades/{source}?since={iso}&limit={n}

Cost-aware accounting: each trade's PnL is also reported net of an
asset-class-specific round-trip cost so the comparison reflects what
the user would actually keep after spreads/commission, not the gross
broker fill.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.core.asset_class import asset_class_for


router = APIRouter(prefix="/reports", tags=["Reports"])


# Round-trip cost in basis points, applied to position notional.
# Values reflect typical retail demo broker conditions (AmanaCapital):
#   - equity:    ~5 bps (commission + small spread)
#   - fx_major:  ~1 bps (tight major spreads, no commission)
#   - fx_metal:  ~3 bps (gold/silver spreads wider than majors)
COST_BPS_PER_ASSET_CLASS = {
    "equity":   5.0,
    "fx_major": 1.0,
    "fx_metal": 3.0,
}

# Friendly display names for the source codes.
SOURCE_DISPLAY = {
    "ml_equities":   "Equities ML",
    "ml_fx":         "FX ML",
    "rule_donchian": "FX Donchian (rule)",
}


def _cost_bps_for(symbol_or_ticker: str) -> float:
    """Best-effort lookup — the symbol may carry a broker suffix (`.r`)."""
    base = symbol_or_ticker.split(".")[0].upper() if symbol_or_ticker else ""
    return COST_BPS_PER_ASSET_CLASS.get(asset_class_for(base), 5.0)


def _aggregate_one_source(trades: list[dict]) -> dict:
    """Compute per-source metrics from a list of closed trade rows.

    Each row should have: pnl (float), volume, fill_price, ticker, symbol,
    closed_at, action. We compute both raw PnL and cost-adjusted PnL.
    """
    if not trades:
        return {
            "total_trades":        0,
            "wins":                0,
            "losses":              0,
            "win_rate":            0.0,
            "total_pnl_usd":       0.0,
            "total_pnl_usd_adj":   0.0,
            "avg_pnl_per_trade":   0.0,
            "sharpe_per_trade":    0.0,
            "max_drawdown_usd":    0.0,
            "max_drawdown_pct":    0.0,
            "equity_curve":        [],
        }

    pnls_raw: list[float]  = []
    pnls_adj: list[float]  = []
    curve: list[dict]      = []

    # Walk trades in chronological order (closed_at) for the equity curve
    sorted_trades = sorted(trades, key=lambda r: r.get("closed_at") or r.get("created_at"))
    cum_raw = 0.0
    cum_adj = 0.0
    peak_adj = 0.0
    max_dd_usd = 0.0

    for t in sorted_trades:
        pnl_raw = float(t.get("pnl") or 0.0)
        volume    = float(t.get("volume") or 0.0)
        fill      = float(t.get("fill_price") or 0.0)
        # Approximate notional. For FX this is rough (true notional needs
        # contract_size from MT5 symbol_info) but it's directionally
        # correct for cost ranking across strategies and assets.
        notional_usd = abs(volume * fill)
        bps = _cost_bps_for(t.get("ticker") or t.get("symbol") or "")
        cost_usd = notional_usd * (bps / 10000.0)
        pnl_adj = pnl_raw - cost_usd

        pnls_raw.append(pnl_raw)
        pnls_adj.append(pnl_adj)

        cum_raw += pnl_raw
        cum_adj += pnl_adj
        peak_adj = max(peak_adj, cum_adj)
        dd = cum_adj - peak_adj   # always <= 0
        if dd < max_dd_usd:
            max_dd_usd = dd

        closed_at = t.get("closed_at") or t.get("created_at")
        curve.append({
            "timestamp":     closed_at.isoformat() if closed_at else None,
            "cum_pnl_raw":   round(cum_raw, 4),
            "cum_pnl_adj":   round(cum_adj, 4),
        })

    arr_adj = np.array(pnls_adj, dtype=float)
    mean    = float(arr_adj.mean()) if len(arr_adj) else 0.0
    std     = float(arr_adj.std())  if len(arr_adj) > 1 else 0.0
    sharpe  = mean / std if std > 1e-9 else 0.0

    wins = sum(1 for p in pnls_adj if p > 0)
    losses = sum(1 for p in pnls_adj if p < 0)
    n = len(pnls_adj)

    # Drawdown % uses peak adjusted equity as the base. When peak is zero
    # (everything underwater from start), report 100% — informative.
    dd_pct = (max_dd_usd / peak_adj * 100.0) if peak_adj > 0 else (100.0 if max_dd_usd < 0 else 0.0)

    return {
        "total_trades":        n,
        "wins":                wins,
        "losses":              losses,
        "win_rate":            round(wins / n, 4) if n else 0.0,
        "total_pnl_usd":       round(cum_raw, 4),
        "total_pnl_usd_adj":   round(cum_adj, 4),
        "avg_pnl_per_trade":   round(mean, 4),
        "sharpe_per_trade":    round(sharpe, 4),
        "max_drawdown_usd":    round(max_dd_usd, 4),
        "max_drawdown_pct":    round(dd_pct, 4),
        "equity_curve":        curve,
    }


@router.get("/strategy-comparison")
async def strategy_comparison(
    since: Optional[str] = Query(None, description="ISO date — only consider trades closed_at >= this"),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Return per-source aggregates so the comparison tab can render side-by-side.

    Response shape:
        {
          "since": "2026-04-15T00:00:00+00:00",
          "sources": [
            {
              "source": "ml_equities",
              "display_name": "Equities ML",
              "total_trades": 17,
              "wins": 9, "losses": 8, "win_rate": 0.529,
              "total_pnl_usd": 142.50, "total_pnl_usd_adj": 121.30,
              "avg_pnl_per_trade": 8.38,
              "sharpe_per_trade": 0.42,
              "max_drawdown_usd": -34.50, "max_drawdown_pct": -19.4,
              "equity_curve": [ {timestamp, cum_pnl_raw, cum_pnl_adj}, ... ]
            },
            ...
          ]
        }
    """
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(400, f"`since` must be ISO date — got {since!r}")
    else:
        # Default to last 30 days, matching the comparison tab default
        since_dt = datetime.now(timezone.utc) - timedelta(days=30)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT te.id, te.signal_id, te.symbol, te.volume, te.fill_price,
                   te.pnl, te.status, te.closed_at, te.created_at,
                   s.ticker, s.source, s.action, s.confidence
            FROM trade_executions te
            JOIN signals s ON s.id = te.signal_id
            WHERE te.status = 'closed'
              AND COALESCE(te.closed_at, te.created_at) >= $1
            ORDER BY te.closed_at ASC NULLS LAST
            """,
            since_dt,
        )

    # Bucket by source
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        src = r["source"] or "unknown"
        by_source.setdefault(src, []).append(dict(r))

    # Always surface the 3 canonical sources even if a bucket is empty so
    # the frontend can render zeros / empty curves consistently.
    canonical = ["ml_equities", "ml_fx", "rule_donchian"]
    for src in canonical:
        by_source.setdefault(src, [])

    sources_out: list[dict] = []
    for src in sorted(by_source.keys()):
        agg = _aggregate_one_source(by_source[src])
        sources_out.append({
            "source":       src,
            "display_name": SOURCE_DISPLAY.get(src, src),
            **agg,
        })

    return {
        "since":   since_dt.isoformat(),
        "until":   datetime.now(timezone.utc).isoformat(),
        "sources": sources_out,
        "cost_model": COST_BPS_PER_ASSET_CLASS,
    }


@router.get("/strategy-trades/{source}")
async def strategy_trades(
    source: str,
    since: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Per-trade drill-down for a single strategy. Used by the comparison
    tab when the user expands a row in the metrics table."""
    if source not in SOURCE_DISPLAY:
        raise HTTPException(400, f"unknown source {source!r}; expected one of {list(SOURCE_DISPLAY)}")

    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(400, f"`since` must be ISO date — got {since!r}")
    else:
        since_dt = datetime.now(timezone.utc) - timedelta(days=30)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT te.id, te.signal_id, te.symbol, te.mt5_ticket, te.order_type,
                   te.volume, te.fill_price, te.stop_loss, te.take_profit,
                   te.pnl, te.status, te.closed_at, te.created_at,
                   s.ticker, s.action, s.confidence, s.predicted_rr, s.expected_value
            FROM trade_executions te
            JOIN signals s ON s.id = te.signal_id
            WHERE s.source = $1
              AND COALESCE(te.closed_at, te.created_at) >= $2
            ORDER BY te.closed_at DESC NULLS LAST, te.created_at DESC
            LIMIT $3
            """,
            source, since_dt, limit,
        )

    trades = []
    for r in rows:
        pnl_raw = float(r["pnl"] or 0.0)
        volume    = float(r["volume"] or 0.0)
        fill      = float(r["fill_price"] or 0.0)
        notional_usd = abs(volume * fill)
        bps = _cost_bps_for(r["ticker"] or r["symbol"] or "")
        cost_usd = notional_usd * (bps / 10000.0)
        outcome = "win" if pnl_raw > 0 else ("loss" if pnl_raw < 0 else "flat")

        trades.append({
            "id":              r["id"],
            "signal_id":       r["signal_id"],
            "ticker":          r["ticker"],
            "symbol":          r["symbol"],
            "mt5_ticket":      r["mt5_ticket"],
            "action":          r["action"],
            "order_type":      r["order_type"],
            "volume":          volume,
            "fill_price":      fill,
            "stop_loss":       float(r["stop_loss"] or 0.0),
            "take_profit":     float(r["take_profit"] or 0.0),
            "confidence":      float(r["confidence"] or 0.0) if r["confidence"] is not None else None,
            "predicted_rr":    float(r["predicted_rr"] or 0.0) if r["predicted_rr"] is not None else None,
            "expected_value":  float(r["expected_value"] or 0.0) if r["expected_value"] is not None else None,
            "pnl_raw_usd":     round(pnl_raw, 4),
            "pnl_adj_usd":     round(pnl_raw - cost_usd, 4),
            "cost_usd":        round(cost_usd, 4),
            "outcome":         outcome,
            "status":          r["status"],
            "closed_at":       r["closed_at"].isoformat() if r["closed_at"] else None,
            "created_at":      r["created_at"].isoformat() if r["created_at"] else None,
        })

    return {
        "source":       source,
        "display_name": SOURCE_DISPLAY[source],
        "since":        since_dt.isoformat(),
        "count":        len(trades),
        "trades":       trades,
    }
