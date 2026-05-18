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


# ── Background runner ─────────────────────────────────────────────────────────

async def _run_pipeline(pool, source: str) -> None:
    """Generate signals for a single source on the tracked-ticker set.

    Imports the heavy services inline to avoid pulling them at module-import
    time (which would slow down startup for every other route)."""
    from app.core.asset_class import is_fx
    from app.services.signal_service import SignalService
    from app.services.ohlcv_service import OHLCVService
    from app.strategies.donchian import DonchianService

    try:
        ohlcv = OHLCVService(pool)
        all_tickers = await ohlcv.get_available_tickers() or []
        if source == "ml_equities":
            tickers = [t for t in all_tickers if not is_fx(t)]
            svc     = SignalService(pool)
            results = await svc.generate_batch(tickers, "1d")
            n = len(results)
        elif source == "ml_fx":
            tickers = [t for t in all_tickers if is_fx(t)]
            svc     = SignalService(pool)
            results = await svc.generate_batch(tickers, "1d")
            n = len(results)
        elif source == "rule_donchian":
            tickers = [t for t in all_tickers if is_fx(t)]
            donch   = DonchianService(pool)
            results = await donch.generate_batch(tickers, "1d")
            n = len(results)
        else:
            raise ValueError(f"no runner for source {source!r}")

        await record_run(pool, source, signals_count=n, error=None)
        logger.info(f"✅ pipeline {source}: {n} signals generated")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        try:
            await record_run(pool, source, signals_count=None, error=err)
        except Exception as e2:
            logger.error(f"could not record pipeline failure for {source}: {e2}")
        logger.error(f"❌ pipeline {source} failed: {err}")
