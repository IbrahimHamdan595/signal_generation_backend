"""
Trading routes — MetaTrader 5 demo/live integration.

Endpoints
---------
GET  /trading/status            — MT5 connection + account info
POST /trading/connect           — initialise MT5 terminal and log in
POST /trading/disconnect        — shut down MT5 connection
GET  /trading/config            — read per-user trading settings
PUT  /trading/config            — update settings (risk %, limits, MT5 creds)
POST /trading/execute/{signal_id} — manually execute a specific signal
GET  /trading/positions         — open positions from MT5
POST /trading/positions/{ticket}/close — close a position by ticket
GET  /trading/executions        — execution history from trade_executions table
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.broker_service import BrokerService
from app.services.execution_service import ExecutionService
from app.services.signal_service import SignalService
from app.services.ohlcv_service import OHLCVService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/trading", tags=["Trading"])


# ── Request / response models ─────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    account:  int    = Field(..., description="MT5 account number")
    password: str    = Field(..., description="MT5 account password")
    server:   str    = Field(..., description="MT5 broker server name, e.g. 'MetaQuotes-Demo'")
    path: Optional[str] = Field(
        None,
        description="Optional: full path to terminal64.exe if auto-detection fails"
    )


class TradingConfigUpdate(BaseModel):
    enabled:            Optional[bool]  = None
    auto_trade:         Optional[bool]  = None
    risk_per_trade_pct: Optional[float] = Field(None, ge=0.1, le=10.0)
    max_open_positions: Optional[int]   = Field(None, ge=1, le=50)
    max_daily_loss_pct: Optional[float] = Field(None, ge=0.5, le=20.0)
    min_confidence:     Optional[float] = Field(None, ge=0.3, le=1.0)
    allowed_actions:    Optional[List[str]] = None
    mt5_account:        Optional[int]   = None
    mt5_server:         Optional[str]   = None
    mt5_path:           Optional[str]   = None
    symbol_suffix:      Optional[str]   = Field(None, description="e.g. '.US' for US stocks on some brokers")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_status(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Returns the current MT5 connection state and live account metrics.
    Does not attempt to reconnect — call POST /trading/connect first.
    """
    broker = BrokerService()
    connected = await broker.is_connected()

    if not connected:
        return {
            "connected": False,
            "account": None,
            "message": "MT5 not connected. Call POST /trading/connect to initialise.",
        }

    account = await broker.account_info()
    exec_svc = ExecutionService(pool, broker)
    config = await exec_svc.get_config(current_user["id"])

    return {
        "connected":   True,
        "account":     account,
        "auto_trade":  config.get("auto_trade", False) if config else False,
        "enabled":     config.get("enabled", False) if config else False,
    }


@router.post("/connect")
async def connect(
    body: ConnectRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Initialise the MT5 terminal and log in with the given credentials.
    The account number and server are also saved in trading_config so the
    scheduler can reconnect automatically after a server restart.
    """
    broker = BrokerService()
    ok, error = await broker.connect(
        account=body.account,
        password=body.password,
        server=body.server,
        path=body.path,
    )
    if not ok:
        raise HTTPException(400, error)

    # Persist connection details (password is NOT stored — re-enter on reconnect)
    exec_svc = ExecutionService(pool, broker)
    config = await exec_svc.upsert_config(current_user["id"], {
        "mt5_account": body.account,
        "mt5_server":  body.server,
        "mt5_path":    body.path,
        "enabled":     True,
    })

    account = await broker.account_info()
    return {
        "connected": True,
        "account":   account,
        "config":    config,
        "message":   f"Connected to {'DEMO' if account['is_demo'] else 'LIVE'} account #{account['login']}",
    }


@router.post("/disconnect")
async def disconnect(current_user=Depends(get_current_active_user)):
    """Shut down the MT5 terminal connection."""
    broker = BrokerService()
    await broker.disconnect()
    return {"connected": False, "message": "MT5 disconnected"}


@router.get("/config")
async def get_config(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Return the current per-user trading configuration."""
    exec_svc = ExecutionService(pool, BrokerService())
    config = await exec_svc.get_config(current_user["id"])
    if not config:
        return {
            "configured": False,
            "message": "No trading config yet. Connect via POST /trading/connect first.",
        }
    # Don't expose raw DB row — strip internal fields
    return {
        "configured":         True,
        "enabled":            config["enabled"],
        "auto_trade":         config["auto_trade"],
        "risk_per_trade_pct": config["risk_per_trade_pct"],
        "max_open_positions": config["max_open_positions"],
        "max_daily_loss_pct": config["max_daily_loss_pct"],
        "min_confidence":     config["min_confidence"],
        "allowed_actions":    config["allowed_actions"],
        "mt5_account":        config["mt5_account"],
        "mt5_server":         config["mt5_server"],
        "symbol_suffix":      config["symbol_suffix"],
        "updated_at":         config["updated_at"],
    }


@router.put("/config")
async def update_config(
    body: TradingConfigUpdate,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Update any subset of the trading configuration."""
    exec_svc = ExecutionService(pool, BrokerService())
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    config = await exec_svc.upsert_config(current_user["id"], updates)
    return config


class ExecuteRequest(BaseModel):
    volume: Optional[float] = Field(
        None, gt=0, description="Override lot size. Omit to use auto position-sizing."
    )


@router.post("/execute/{signal_id}")
async def execute_signal(
    signal_id: int,
    body: ExecuteRequest = ExecuteRequest(),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Manually execute a specific signal.
    Runs all risk checks (daily loss limit, max positions, stop-loss required)
    and places a market order via MT5.
    Pass `volume` in the request body to override auto position-sizing.
    """
    broker = BrokerService()
    if not await broker.is_connected():
        raise HTTPException(400, "MT5 not connected — call POST /trading/connect first")

    exec_svc = ExecutionService(pool, broker)
    result = await exec_svc.execute_signal(signal_id, current_user["id"], volume_override=body.volume)

    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


@router.get("/positions")
async def get_positions(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Return all open positions currently held in MT5 (filtered by app magic number)."""
    broker = BrokerService()
    if not await broker.is_connected():
        raise HTTPException(400, "MT5 not connected")
    positions = await broker.get_open_positions()
    if not positions:
        return positions
    tickets = [p["ticket"] for p in positions if p.get("ticket")]
    if tickets and pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT te.mt5_ticket, s.source
                FROM trade_executions te
                LEFT JOIN signals s ON s.id = te.signal_id
                WHERE te.mt5_ticket = ANY($1::bigint[])
                """,
                tickets,
            )
        source_map = {r["mt5_ticket"]: r["source"] for r in rows}
        for p in positions:
            p["source"] = source_map.get(p.get("ticket"))
    return positions


@router.post("/positions/{ticket}/close")
async def close_position(
    ticket: int,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Close a specific MT5 position by ticket number at market price."""
    broker = BrokerService()
    if not await broker.is_connected():
        raise HTTPException(400, "MT5 not connected")

    exec_svc = ExecutionService(pool, broker)
    result = await exec_svc.close_position_by_ticket(ticket, current_user["id"])
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


@router.post("/run-now")
async def run_now(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Manually trigger signal regeneration then immediately auto-execute
    any BUY/SELL signals for the current user.
    Requires MT5 to be connected and auto_trade = TRUE.
    """
    broker = BrokerService()
    if not await broker.is_connected():
        raise HTTPException(400, "MT5 not connected — call POST /trading/connect first")

    exec_svc = ExecutionService(pool, broker)
    config = await exec_svc.get_config(current_user["id"])
    if not config or not config.get("enabled"):
        raise HTTPException(400, "Trading not enabled — configure it first")
    if not config.get("auto_trade"):
        raise HTTPException(400, "Auto-trade is off — enable it in Risk Configuration")

    # 1. Get tracked tickers
    ohlcv_svc = OHLCVService(pool)
    tickers = await ohlcv_svc.get_available_tickers()
    if not tickers:
        raise HTTPException(400, "No tickers in database — ingest OHLCV data first")

    # 2. Validate which tickers exist on this broker
    suffix = config.get("symbol_suffix") or ""
    valid_tickers, invalid_tickers = await broker.validate_symbols(tickers, suffix)
    if not valid_tickers:
        raise HTTPException(400, "None of your tracked tickers are available on this MT5 broker")

    # 3. Regenerate signals across every enabled pipeline on the broker-valid
    # universe. _run_pipeline_inner handles asset-class filtering (equities vs
    # FX) and routes ML calls to the right checkpoint (daily vs 1h). Disabled
    # pipelines are skipped silently and their last_run_at is not updated.
    from app.api.routes.pipelines import (
        get_enabled_sources, record_run, _run_pipeline_inner, SOURCE_META,
    )

    enabled = await get_enabled_sources(pool)
    # Re-read the latest signals after each pipeline so we can tally per-source.
    pipeline_results: dict[str, dict] = {}
    for src in sorted(enabled):
        if src not in SOURCE_META:
            continue
        try:
            n = await _run_pipeline_inner(pool, src, tickers=valid_tickers)
            await record_run(pool, src, signals_count=n, error=None)
            pipeline_results[src] = {"signals_generated": n, "ok": True}
            logger.info(f"✅ pipeline {src}: {n} signals generated")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            try:
                await record_run(pool, src, signals_count=None, error=err)
            except Exception:
                pass
            pipeline_results[src] = {"signals_generated": 0, "ok": False, "error": err}
            logger.error(f"❌ pipeline {src} failed: {err}")

    # Pull the just-generated signals back from the DB so the per-source
    # breakdown and filter-reason tally stay accurate. We use the last
    # ~5 minutes window — generous enough for slow runs, narrow enough to
    # avoid mixing in stale rows from the previous scheduler tick.
    async with pool.acquire() as conn:
        signal_rows = await conn.fetch(
            """
            SELECT source, action, expected_value, reject_reasons
            FROM signals
            WHERE created_at >= NOW() - INTERVAL '5 minutes'
            """
        )
    signals = [dict(r) for r in signal_rows]

    # Diagnostic breakdown of the new profit-filter gates: how many signals
    # were downgraded to HOLD and the reasons. Surfaced to the UI so users
    # can see which gates are killing trades.
    actionable     = [s for s in signals if s.get("action") in ("BUY", "SELL")]
    filtered_hold  = [s for s in signals if s.get("action") == "HOLD" and s.get("reject_reasons")]
    reason_counts: dict[str, int] = {}
    for s in filtered_hold:
        for r in (s.get("reject_reasons") or []):
            # First token is the gate name (e.g. "conf=", "uncertainty=", "EV=")
            key = (r.split("=")[0] if "=" in r else r.split("<")[0]).strip()
            reason_counts[key] = reason_counts.get(key, 0) + 1

    # Per-source counts so the toast can show "12 Equities ML, 4 FX ML, 2 Donchian"
    per_source_counts: dict[str, int] = {}
    for s in signals:
        src = s.get("source") or "ml_equities"
        per_source_counts[src] = per_source_counts.get(src, 0) + 1

    # 4. Auto-execute new signals (across all three sources — auto_execute reads
    # the signals table and ranks BUY/SELL rows by EV regardless of source).
    results = await exec_svc.auto_execute(current_user["id"])
    filled = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    # Aggregate expected value across actionable signals (rough P&L forecast
    # — sums confidence-weighted edge across all BUY/SELL signals generated)
    total_ev = sum(float(s.get("expected_value") or 0.0) for s in actionable)

    return {
        "signals_generated":  len(signals),
        "signals_actionable": len(actionable),
        "signals_filtered":   len(filtered_hold),
        "signals_by_source":  per_source_counts,
        "filter_breakdown":   reason_counts,
        "total_expected_value": round(total_ev, 6),
        "orders_attempted":   len(results),
        "orders_filled":      len(filled),
        "orders_failed":      len(failed),
        "skipped_tickers":    invalid_tickers,
        "filled":             filled,
        "failed":             failed,
    }


@router.post("/backfill-pnl")
async def backfill_pnl(
    include_zero: bool = False,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    One-off recovery: walk every closed trade with NULL pnl (or pnl=0 when
    include_zero=true) and pull realised pnl from MT5's 90-day deal history.

    Pass include_zero=true to also re-process rows recorded as pnl=0 by an
    earlier reconciler version that returned the opening deal's profit
    (which is always 0) instead of summing all deals for the position.
    """
    broker = BrokerService()
    if not await broker.is_connected():
        raise HTTPException(400, "MT5 not connected — call POST /trading/connect first")

    from app.services.reconciliation_service import ReconciliationService
    recon = ReconciliationService(pool, broker)

    where_clause = "pnl IS NULL" if not include_zero else "(pnl IS NULL OR pnl = 0)"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, mt5_ticket
            FROM trade_executions
            WHERE status = 'closed'
              AND {where_clause}
              AND mt5_ticket IS NOT NULL
            ORDER BY closed_at DESC NULLS LAST
            """
        )

    recovered, missing = [], []
    for r in rows:
        ticket = int(r["mt5_ticket"])
        pnl = await recon._lookup_closed_pnl(ticket)
        if pnl is None:
            missing.append(ticket)
            continue
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE trade_executions SET pnl = $1 WHERE id = $2",
                float(pnl), int(r["id"]),
            )
        recovered.append({"ticket": ticket, "pnl": float(pnl)})

    return {
        "total_null":       len(rows),
        "recovered_count":  len(recovered),
        "missing_count":    len(missing),
        "recovered":        recovered[:50],
        "missing_tickets":  missing[:50],
    }


@router.get("/executions")
async def get_executions(
    limit: int = 20,
    offset: int = 0,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Return paginated execution history for the current user."""
    exec_svc = ExecutionService(pool, BrokerService())
    return await exec_svc.get_executions(
        current_user["id"], limit=min(limit, 100), offset=offset
    )


@router.get("/activity")
async def get_activity(
    limit: int = 20,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Unified activity feed for the dashboard. UNIONs four virtual event streams
    from the last 24h:
      - signal_generation   (per-source, grouped to the minute)
      - order_fill          (trade_executions status='filled')
      - order_close         (trade_executions status='closed')
      - order_reject        (trade_executions status='rejected_commission')
    Returns the most recent `limit` events DESC by timestamp.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH gens AS (
              SELECT
                date_trunc('minute', created_at) AS ts,
                'signal_generation'::text       AS kind,
                source                           AS source,
                NULL::text                       AS symbol,
                COUNT(*)::int                    AS count,
                NULL::float                      AS pnl
              FROM signals
              WHERE created_at >= NOW() - INTERVAL '24 hours'
              GROUP BY 1, source
            ),
            fills AS (
              SELECT te.created_at, 'order_fill'::text, s.source, te.symbol, 1, NULL::float
              FROM trade_executions te LEFT JOIN signals s ON s.id = te.signal_id
              WHERE te.status = 'filled' AND te.created_at >= NOW() - INTERVAL '24 hours'
            ),
            closes AS (
              SELECT te.closed_at, 'order_close'::text, s.source, te.symbol, 1, te.pnl
              FROM trade_executions te LEFT JOIN signals s ON s.id = te.signal_id
              WHERE te.status = 'closed'
                AND te.closed_at IS NOT NULL
                AND te.closed_at >= NOW() - INTERVAL '24 hours'
            ),
            rejects AS (
              SELECT te.created_at, 'order_reject'::text, s.source, te.symbol, 1, NULL::float
              FROM trade_executions te LEFT JOIN signals s ON s.id = te.signal_id
              WHERE te.status = 'rejected_commission'
                AND te.created_at >= NOW() - INTERVAL '24 hours'
            )
            SELECT * FROM gens
            UNION ALL SELECT * FROM fills
            UNION ALL SELECT * FROM closes
            UNION ALL SELECT * FROM rejects
            ORDER BY ts DESC NULLS LAST
            LIMIT $1
            """,
            limit,
        )
    return [
        {
            "ts":     r["ts"].isoformat() if r["ts"] else None,
            "kind":   r["kind"],
            "source": r["source"],
            "symbol": r["symbol"],
            "count":  r["count"],
            "pnl":    float(r["pnl"]) if r["pnl"] is not None else None,
        }
        for r in rows
    ]


@router.get("/commission-stats")
async def get_commission_stats(
    days: int = 7,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Total broker commission saved by the Path A commission gate over the
    last `days` days. Sums COMMISSION_USD[asset_class] per rejected signal.
    """
    from app.services.execution_service import COMMISSION_USD
    from app.core.asset_class import asset_class_for

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT te.symbol, s.ticker
            FROM trade_executions te
            LEFT JOIN signals s ON s.id = te.signal_id
            WHERE te.status = 'rejected_commission'
              AND te.created_at >= NOW() - ($1 || ' days')::interval
            """,
            str(days),
        )
    saved = 0.0
    for r in rows:
        ac = asset_class_for(r["ticker"] or r["symbol"] or "")
        saved += COMMISSION_USD.get(ac, 15.0)
    return {
        "days":                days,
        "signals_filtered":    len(rows),
        "commission_saved_usd": round(saved, 2),
    }
