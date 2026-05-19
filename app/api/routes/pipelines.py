"""Pipeline orchestration API.

Three pipelines (`ml_equities`, `ml_fx`, `rule_donchian`) each have a row
in `pipeline_config` carrying their enable flag, last-run status, and
strategy-specific config. These endpoints expose that state to the UI:

  GET    /api/v1/pipelines               — list all pipelines + status
  PATCH  /api/v1/pipelines/{source}      — toggle enabled, edit config
  POST   /api/v1/pipelines/{source}/run  — trigger one pipeline immediately
  POST   /api/v1/pipelines/run-all       — trigger all enabled pipelines

The scheduler reads `pipeline_config.enabled` before running each pipeline
so disabled strategies are skipped silently until the user re-enables them
via PATCH.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import get_current_active_user
from app.db.database import get_db


router = APIRouter(prefix="/pipelines", tags=["Pipelines"])

logger = logging.getLogger(__name__)


# Display metadata for each source — kept in code (not DB) because it's UI-only
# and won't ever vary per user.
SOURCE_META: dict[str, dict[str, str]] = {
    "ml_equities": {
        "display_name": "Equities ML",
        "description":  "Transformer+MLP model on S&P 500 equities (daily bars).",
        "kind":         "ml",
    },
    "ml_fx": {
        "display_name": "FX ML",
        "description":  "Transformer+MLP model on FX/metal pairs (daily bars, vol-normalised labels).",
        "kind":         "ml",
    },
    "rule_donchian": {
        "display_name": "FX Donchian (rule)",
        "description":  "Classic 20-day breakout, ATR-based stops. No ML.",
        "kind":         "rule",
    },
}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PipelineConfigPatch(BaseModel):
    """Partial update — None fields are left unchanged."""
    enabled: Optional[bool]              = None
    config:  Optional[dict[str, Any]]    = Field(default=None, description="Strategy-specific tuning JSON")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_one(pool, source: str) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM pipeline_config WHERE source = $1",
            source,
        )
    return dict(row) if row else None


async def _load_all(pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM pipeline_config ORDER BY source"
        )
    return [dict(r) for r in rows]


async def get_enabled_sources(pool) -> set[str]:
    """Helper for the scheduler — returns the set of source codes that are
    currently enabled. Falls back to all-enabled when the table is unreachable
    so a DB hiccup never silently kills the whole pipeline."""
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT source FROM pipeline_config WHERE enabled = TRUE")
        return {r["source"] for r in rows}
    except Exception as e:
        logger.warning(f"pipeline_config unreachable, defaulting to all enabled: {e}")
        return set(SOURCE_META.keys())


async def record_run(
    pool,
    source: str,
    *,
    signals_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Persist a run summary back to the table. Called by the scheduler
    after each pipeline cycle (whether the pipeline ran or errored)."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE pipeline_config
            SET last_run_at        = NOW(),
                last_signals_count = $2,
                last_error         = $3,
                updated_at         = NOW()
            WHERE source = $1
            """,
            source, signals_count, error,
        )


def _decorate(row: dict) -> dict:
    """Add display metadata to a raw DB row."""
    meta = SOURCE_META.get(row["source"], {})
    cfg = row.get("config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            cfg = {}
    return {
        "source":             row["source"],
        "display_name":       meta.get("display_name", row["source"]),
        "description":        meta.get("description", ""),
        "kind":               meta.get("kind", "ml"),
        "enabled":            bool(row["enabled"]),
        "last_run_at":        row["last_run_at"].isoformat() if row.get("last_run_at") else None,
        "last_signals_count": row.get("last_signals_count"),
        "last_error":         row.get("last_error"),
        "config":             cfg or {},
        "updated_at":         row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
async def list_pipelines(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Return all configured pipelines with status. Used by the /pipeline UI."""
    rows = await _load_all(pool)
    return {"pipelines": [_decorate(r) for r in rows]}


@router.patch("/{source}")
async def update_pipeline(
    source: str,
    body: PipelineConfigPatch,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Patch a pipeline's enabled flag and/or its strategy-specific config."""
    if source not in SOURCE_META:
        raise HTTPException(404, f"unknown source {source!r}")

    row = await _load_one(pool, source)
    if not row:
        raise HTTPException(404, f"pipeline_config row missing for {source!r}")

    new_enabled = body.enabled if body.enabled is not None else row["enabled"]
    if body.config is not None:
        new_config = body.config
    else:
        existing = row.get("config")
        new_config = json.loads(existing) if isinstance(existing, str) else (existing or {})

    async with pool.acquire() as conn:
        updated = await conn.fetchrow(
            """
            UPDATE pipeline_config
            SET enabled    = $2,
                config     = $3::jsonb,
                updated_at = NOW()
            WHERE source = $1
            RETURNING *
            """,
            source, new_enabled, json.dumps(new_config),
        )
    logger.info(f"pipeline_config updated: {source} enabled={new_enabled}")
    return _decorate(dict(updated))


@router.post("/{source}/run")
async def run_one(
    source: str,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Trigger one pipeline immediately (regardless of its enabled flag).
    Returns 202 — the actual generation runs in the background and the
    result is reflected in `last_run_at` / `last_signals_count` once it
    completes."""
    if source not in SOURCE_META:
        raise HTTPException(404, f"unknown source {source!r}")

    background_tasks.add_task(_run_pipeline, pool, source)
    return {"status": "scheduled", "source": source}


@router.post("/run-all")
async def run_all(
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Trigger every ENABLED pipeline immediately. Disabled ones are skipped."""
    enabled = await get_enabled_sources(pool)
    if not enabled:
        return {"status": "noop", "reason": "no pipelines enabled"}
    for src in enabled:
        background_tasks.add_task(_run_pipeline, pool, src)
    return {"status": "scheduled", "sources": sorted(enabled)}


@router.post("/run-all-and-trade")
async def run_all_and_trade(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Run every enabled pipeline (signal generation), then immediately
    auto-execute with the per-pipeline quota policy.

    Combines /pipelines/run-all (generate signals from each enabled pipeline)
    and the auto-execute step from /trading/run-now into one call. Unlike
    /trading/run-now this runs on every tracked ticker (not just broker-valid);
    signals for non-broker tickers fail at per-trade broker validation and
    are skipped without polluting the executions table.

    Synchronous — the HTTP request stays open until generation + execution
    both complete. Typical wall-clock: 1–3 minutes depending on universe
    size and DB latency. The caller should expect a long timeout.
    """
    # ── Guards ────────────────────────────────────────────────────────────
    from app.services.broker_service import BrokerService
    from app.services.execution_service import ExecutionService

    broker = BrokerService()
    if not await broker.is_connected():
        raise HTTPException(400, "MT5 not connected — call POST /trading/connect first")

    exec_svc = ExecutionService(pool, broker)
    cfg = await exec_svc.get_config(current_user["id"])
    if not cfg or not cfg.get("enabled"):
        raise HTTPException(400, "Trading not enabled — configure it first")
    if not cfg.get("auto_trade"):
        raise HTTPException(400, "Auto-trade is off — enable it in Risk Configuration")

    enabled = await get_enabled_sources(pool)
    if not enabled:
        return {"status": "noop", "reason": "no pipelines enabled"}

    # ── Phase 1: generate signals from every enabled pipeline (sequential) ─
    # Sequential so the three pipelines don't contend for the GPU and the DB
    # pool. record_run is called after each so /strategies cards show fresh
    # last_run_at as the run progresses.
    pipeline_results: dict[str, dict] = {}
    for src in sorted(enabled):
        try:
            n = await _run_pipeline_inner(pool, src)
            await record_run(pool, src, signals_count=n, error=None)
            logger.info(f"✅ pipeline {src}: {n} signals generated")
            pipeline_results[src] = {"signals_generated": n, "ok": True}
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            try:
                await record_run(pool, src, signals_count=None, error=err)
            except Exception:
                pass
            logger.error(f"❌ pipeline {src} failed: {err}")
            pipeline_results[src] = {"signals_generated": 0, "ok": False, "error": err}

    # ── Phase 2: auto-execute with per-pipeline quota ──────────────────────
    # auto_execute already applies: top-conf within each pipeline, even quota
    # across enabled pipelines, redistribute leftover slots, interleaved order.
    results = await exec_svc.auto_execute(current_user["id"])
    filled = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    return {
        "status":           "complete",
        "pipelines":        pipeline_results,
        "orders_attempted": len(results),
        "orders_filled":    len(filled),
        "orders_failed":    len(failed),
        "filled":           filled,
        "failed":           failed,
    }


@router.post("/reset")
async def reset_pipeline_state(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Admin reset — clears transient state so the system starts fresh.

    Touches four tables in a single transaction:

    - `trade_executions`: stale `pending` rows flipped to `failed` so blocked
      signal_ids can be retried on the next auto-execute cycle.
    - `pipeline_config`: `last_run_at` / `last_signals_count` / `last_error`
      reset to NULL so the /strategies cards show "awaiting first run".
    - `equity_snapshots`: today's row is deleted so the next health check
      rebases the equity guardrail at the current MT5 equity (avoids a
      stale peak triggering EQUITY_HALT).
    - `trading_config`: `auto_trade` flipped back to TRUE for every user
      that had it (lets execution resume after a guardrail halt).

    Does NOT delete historical signals, executions history, or model
    checkpoints. Does NOT kill in-flight stuck APScheduler jobs — the
    Python process must be restarted for that.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Count pending orders before clearing so we can report it.
            # trade_executions uses status values: pending | filled | closed |
            # cancelled | error (per migration 0002). We mark stale pending
            # rows as 'cancelled' — it's the closest terminal state and falls
            # outside auto_execute's NOT-EXISTS filter (which excludes only
            # 'filled' and 'pending'), so the same signal_id can be retried.
            # Audit trail goes into mt5_comment, not 'error' (no such column).
            pending_count = await conn.fetchval(
                "SELECT COUNT(*) FROM trade_executions WHERE status = 'pending'"
            ) or 0
            await conn.execute(
                """
                UPDATE trade_executions
                SET status      = 'cancelled',
                    mt5_comment = COALESCE(mt5_comment, '') ||
                                  CASE WHEN mt5_comment IS NULL OR mt5_comment = ''
                                       THEN 'cleared by /pipelines/reset'
                                       ELSE ' | cleared by /pipelines/reset'
                                  END
                WHERE status = 'pending'
                """
            )

            # 2. Wipe per-pipeline last-run state
            await conn.execute(
                """
                UPDATE pipeline_config
                SET last_run_at        = NULL,
                    last_signals_count = NULL,
                    last_error         = NULL,
                    updated_at         = NOW()
                """
            )

            # 3. Rebase today's equity guardrail
            equity_deleted = await conn.fetchval(
                """
                WITH d AS (DELETE FROM equity_snapshots WHERE date = CURRENT_DATE RETURNING 1)
                SELECT COUNT(*) FROM d
                """
            ) or 0

            # 4. Re-enable auto_trade everywhere it was off
            auto_trade_reenabled = await conn.fetchval(
                """
                WITH u AS (
                    UPDATE trading_config SET auto_trade = TRUE
                    WHERE auto_trade = FALSE
                    RETURNING 1
                )
                SELECT COUNT(*) FROM u
                """
            ) or 0

    logger.info(
        f"Pipeline state reset by user {current_user.get('id')}: "
        f"pending_cleared={pending_count}, "
        f"equity_rebased={equity_deleted}, "
        f"auto_trade_reenabled={auto_trade_reenabled}"
    )
    return {
        "status":                 "reset",
        "pending_orders_cleared": pending_count,
        "pipeline_cards_reset":   True,
        "equity_snapshot_rebased": bool(equity_deleted),
        "auto_trade_reenabled":   auto_trade_reenabled,
        "note": (
            "If a job is hung mid-execution, restart the backend — this endpoint "
            "cannot interrupt running coroutines."
        ),
    }


# ── Background runner ─────────────────────────────────────────────────────────

async def _run_pipeline_inner(pool, source: str) -> int:
    """Inner implementation — runs the pipeline, returns signals_count.

    Imports the heavy services inline to avoid pulling them at module-import
    time (which would slow down startup for every other route). Raises on
    any failure; the wrappers below handle persistence of last_error."""
    from app.core.asset_class import is_fx
    from app.services.signal_service import SignalService
    from app.services.ohlcv_service import OHLCVService
    from app.strategies.donchian import DonchianService

    ohlcv = OHLCVService(pool)
    all_tickers = await ohlcv.get_available_tickers() or []
    if source == "ml_equities":
        tickers = [t for t in all_tickers if not is_fx(t)]
        svc     = SignalService(pool)
        results = await svc.generate_batch(tickers, "1d")
    elif source == "ml_fx":
        tickers = [t for t in all_tickers if is_fx(t)]
        svc     = SignalService(pool)
        results = await svc.generate_batch(tickers, "1d")
    elif source == "rule_donchian":
        tickers = [t for t in all_tickers if is_fx(t)]
        donch   = DonchianService(pool)
        results = await donch.generate_batch(tickers, "1d")
    else:
        raise ValueError(f"no runner for source {source!r}")
    return len(results)


async def _run_pipeline(pool, source: str) -> None:
    """Background-task wrapper around _run_pipeline_inner. Catches and
    records errors so a fire-and-forget BackgroundTask never explodes."""
    try:
        n = await _run_pipeline_inner(pool, source)
        await record_run(pool, source, signals_count=n, error=None)
        logger.info(f"✅ pipeline {source}: {n} signals generated")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        try:
            await record_run(pool, source, signals_count=None, error=err)
        except Exception as e2:
            logger.error(f"could not record pipeline failure for {source}: {e2}")
        logger.error(f"❌ pipeline {source} failed: {err}")
