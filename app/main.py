from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

from app.core.config import settings
from app.core.asset_class import fx_pair_count   # import fails fast if fx_majors.json is malformed
from app.db.database import connect_db, close_db
from app.api.router import api_router
from app.ml.models.registry import load_model
from app.services.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"FX registry loaded: {fx_pair_count()} pairs")
    await connect_db()
    from app.services.storage_service import sync_from_cloud
    sync_from_cloud()
    # Eagerly load both per-asset-class models so the first request doesn't
    # pay the disk-read latency. Either may be absent (untrained) — `load_model`
    # handles missing files gracefully.
    load_model("equities")
    load_model("fx")
    start_scheduler()
    yield
    stop_scheduler()
    await close_db()
    logger.info("Server shut down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Phases 1-4: Data + Sentiment + AI Model + Real-Time Signal Engine",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_cors_origins = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Per-pipeline operational health.

    Returns enabled flag, training status, last-run timestamp, last error,
    and last-signals-count for each of the three pipelines. External
    monitors (uptime checks, dashboards) get a single endpoint that
    reflects the whole multi-pipeline system's state.
    """
    from app.ml.models.registry import is_model_trained
    from app.db.database import get_db
    from app.api.routes.pipelines import get_enabled_sources

    out = {"status": "ok", "scheduler": "running", "pipelines": {}}

    # Per-pipeline section — degrades gracefully if the DB is unreachable
    try:
        pool = await get_db()
        enabled = await get_enabled_sources(pool)
        async with pool.acquire() as conn:
            cfg_rows = await conn.fetch(
                "SELECT source, enabled, last_run_at, last_signals_count, last_error "
                "FROM pipeline_config ORDER BY source"
            )
        for r in cfg_rows:
            src = r["source"]
            if src == "ml_equities":
                trained = is_model_trained("equities")
            elif src == "ml_fx":
                trained = is_model_trained("fx")
            else:   # rule_donchian — no training step
                trained = True
            out["pipelines"][src] = {
                "enabled":            src in enabled,
                "trained":            trained,
                "last_run_at":        r["last_run_at"].isoformat() if r["last_run_at"] else None,
                "last_signals_count": r["last_signals_count"],
                "last_error":         r["last_error"],
            }
    except Exception as e:
        out["status"] = "degraded"
        out["error"]  = f"pipeline_config unreachable: {type(e).__name__}: {e}"

    # Legacy field kept for backwards-compat with anything still consuming the
    # old /health shape — `True` iff *any* ML pipeline has a trained checkpoint.
    out["model_trained"] = bool(
        out["pipelines"].get("ml_equities", {}).get("trained")
        or out["pipelines"].get("ml_fx", {}).get("trained")
    )
    return out
