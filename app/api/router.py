from fastapi import APIRouter
from app.api.routes import (
    auth, users, ingest, market, sentiment,
    ml, signals, alerts, outcomes, backtest, confluence, ws,
    portfolio, price_alerts, jobs,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(ingest.router)
api_router.include_router(market.router)
api_router.include_router(sentiment.router)
api_router.include_router(ml.router)
api_router.include_router(signals.router)
api_router.include_router(alerts.router)
api_router.include_router(outcomes.router)
api_router.include_router(backtest.router)
api_router.include_router(confluence.router)
api_router.include_router(ws.router)
api_router.include_router(portfolio.router)
api_router.include_router(price_alerts.router)
api_router.include_router(jobs.router)
