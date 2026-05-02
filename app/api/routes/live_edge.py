"""
Live Edge Dashboard
===================
Honest production metrics: what the model actually achieved on real trades,
not what it predicted.

Endpoints
---------
GET  /live-edge/summary        — overall live performance vs predicted
GET  /live-edge/daily          — per-day breakdown
GET  /live-edge/calibration    — confidence vs actual win rate buckets
GET  /live-edge/slippage       — distribution of fill price vs signal price
POST /live-edge/backtest       — run real-fill backtest on historical signals
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional, List
import logging

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.realfill_backtest_service import RealFillBacktestService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/live-edge", tags=["Live Edge"])


@router.get("/summary")
async def summary(
    days: int = Query(30, ge=1, le=365),
    current_user=Depends(get_current_active_user),
):
    """
    Overall live trading performance over the last N days.
    Compares predicted confidence to actual win rate, computes real PnL.
    """
    pool = await get_db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
              COUNT(*)                                                     AS total_trades,
              COUNT(*) FILTER (WHERE te.pnl > 0)                            AS wins,
              COUNT(*) FILTER (WHERE te.pnl < 0)                            AS losses,
              COALESCE(SUM(te.pnl), 0)                                      AS realized_pnl,
              COALESCE(AVG(te.pnl), 0)                                      AS avg_pnl,
              COALESCE(AVG(s.confidence), 0)                                AS avg_confidence,
              COALESCE(AVG(CASE WHEN te.pnl > 0 THEN 1.0 ELSE 0.0 END), 0)  AS actual_win_rate,
              COALESCE(AVG(ABS(te.fill_price - s.entry_price) / NULLIF(s.entry_price,0)), 0) AS slippage_pct
            FROM trade_executions te
            JOIN signals s ON s.id = te.signal_id
            WHERE te.status = 'closed'
              AND te.closed_at >= NOW() - ($1 || ' days')::interval
            """,
            str(days),
        )

    total = int(row["total_trades"] or 0)
    if total == 0:
        return {"days": days, "total_trades": 0, "no_data": True}

    return {
        "days":                  days,
        "total_trades":          total,
        "wins":                  int(row["wins"] or 0),
        "losses":                int(row["losses"] or 0),
        "actual_win_rate":       round(float(row["actual_win_rate"]), 4),
        "avg_predicted_conf":    round(float(row["avg_confidence"]), 4),
        "calibration_gap":       round(float(row["avg_confidence"] - row["actual_win_rate"]), 4),
        "realized_pnl":          round(float(row["realized_pnl"]), 2),
        "avg_pnl":               round(float(row["avg_pnl"]), 4),
        "avg_slippage_pct":      round(float(row["slippage_pct"]) * 100, 4),
    }


@router.get("/daily")
async def daily(
    days: int = Query(30, ge=1, le=365),
    current_user=Depends(get_current_active_user),
):
    """Per-day P&L, trade count, and win rate."""
    pool = await get_db()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
              DATE(te.closed_at)                                            AS day,
              COUNT(*)                                                      AS trades,
              SUM(te.pnl)                                                   AS pnl,
              AVG(CASE WHEN te.pnl > 0 THEN 1.0 ELSE 0.0 END)               AS win_rate,
              AVG(s.confidence)                                             AS avg_conf,
              AVG(ABS(te.fill_price - s.entry_price) / NULLIF(s.entry_price,0)) AS slippage
            FROM trade_executions te
            JOIN signals s ON s.id = te.signal_id
            WHERE te.status = 'closed'
              AND te.closed_at >= NOW() - ($1 || ' days')::interval
            GROUP BY 1
            ORDER BY 1 DESC
            """,
            str(days),
        )

    return [
        {
            "day":          r["day"].isoformat(),
            "trades":       int(r["trades"]),
            "pnl":          round(float(r["pnl"] or 0), 2),
            "win_rate":     round(float(r["win_rate"] or 0), 4),
            "avg_conf":     round(float(r["avg_conf"] or 0), 4),
            "slippage_pct": round(float(r["slippage"] or 0) * 100, 4),
        }
        for r in rows
    ]


@router.get("/calibration")
async def calibration(current_user=Depends(get_current_active_user)):
    """
    Reliability diagram: bucket signals by predicted confidence,
    compare to actual win rate. A well-calibrated model has
    avg_confidence ≈ actual_win_rate per bucket.
    """
    pool = await get_db()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
              WIDTH_BUCKET(s.confidence, 0.5, 1.0, 5)        AS bucket,
              COUNT(*)                                       AS n,
              AVG(s.confidence)                              AS avg_conf,
              AVG(CASE WHEN te.pnl > 0 THEN 1.0 ELSE 0.0 END) AS actual_win_rate
            FROM trade_executions te
            JOIN signals s ON s.id = te.signal_id
            WHERE te.status = 'closed'
            GROUP BY 1
            ORDER BY 1
            """
        )

    return [
        {
            "bucket":          int(r["bucket"]),
            "n":               int(r["n"]),
            "avg_confidence":  round(float(r["avg_conf"] or 0), 4),
            "actual_win_rate": round(float(r["actual_win_rate"] or 0), 4),
            "calibration_gap": round(float((r["avg_conf"] or 0) - (r["actual_win_rate"] or 0)), 4),
        }
        for r in rows
    ]


@router.get("/slippage")
async def slippage(current_user=Depends(get_current_active_user)):
    """
    Slippage distribution: how far is fill_price from signal entry_price?
    Returns percentiles. > 0.2% suggests execution-quality issues.
    """
    pool = await get_db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
              PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY s_pct) AS p50,
              PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY s_pct) AS p95,
              PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY s_pct) AS p99,
              MAX(s_pct) AS worst,
              AVG(s_pct) AS avg
            FROM (
              SELECT ABS(te.fill_price - s.entry_price) / NULLIF(s.entry_price,0) AS s_pct
              FROM trade_executions te
              JOIN signals s ON s.id = te.signal_id
              WHERE te.status IN ('filled', 'closed')
                AND te.fill_price IS NOT NULL
                AND s.entry_price IS NOT NULL
            ) sub
            """
        )

    if not row or row["avg"] is None:
        return {"no_data": True}

    return {
        "avg_pct":   round(float(row["avg"]) * 100, 4),
        "p50_pct":   round(float(row["p50"]) * 100, 4),
        "p95_pct":   round(float(row["p95"]) * 100, 4),
        "p99_pct":   round(float(row["p99"]) * 100, 4),
        "worst_pct": round(float(row["worst"]) * 100, 4),
    }


@router.post("/backtest")
async def run_backtest(
    tickers: List[str],
    interval: str = "1d",
    current_user=Depends(get_current_active_user),
):
    """
    Run real-fill backtest on historical signals stored in DB. Walks actual
    OHLCV bars forward from each signal — gives the honest Sharpe / win rate.
    """
    pool = await get_db()
    svc = RealFillBacktestService(pool)
    return await svc.backtest_portfolio(tickers, interval)
