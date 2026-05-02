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
    return await broker.get_open_positions()


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
        raise HTTPException(400, "No tickers in database — add tickers to your watchlist first")

    # 2. Validate which tickers exist on this broker
    suffix = config.get("symbol_suffix") or ""
    valid_tickers, invalid_tickers = await broker.validate_symbols(tickers, suffix)
    if not valid_tickers:
        raise HTTPException(400, "None of your tracked tickers are available on this MT5 broker")

    # 3. Regenerate signals only for broker-valid tickers
    signal_svc = SignalService(pool)
    signals = await signal_svc.generate_batch(valid_tickers, "1d")

    # 4. Auto-execute new signals
    results = await exec_svc.auto_execute(current_user["id"])
    filled = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    return {
        "signals_generated": len(signals),
        "orders_attempted": len(results),
        "orders_filled": len(filled),
        "orders_failed": len(failed),
        "skipped_tickers": invalid_tickers,
        "filled": filled,
        "failed": failed,
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
