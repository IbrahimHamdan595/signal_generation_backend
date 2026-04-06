from fastapi import APIRouter
from app.api.routes import auth, users, ingest, market, sentiment, ml, signals

# NOTE: signals router added in Phase 4

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(ingest.router)
api_router.include_router(market.router)
api_router.include_router(sentiment.router)
api_router.include_router(ml.router)
api_router.include_router(signals.router)