from fastapi import APIRouter, Depends
from typing import List
from app.db.database import get_db
from app.services.confluence_service import ConfluenceService
from app.core.security import get_current_user

router = APIRouter(prefix="/confluence", tags=["confluence"])


def _svc(pool=Depends(get_db)) -> ConfluenceService:
    return ConfluenceService(pool)


@router.get("/{ticker}")
async def get_confluence(
    ticker: str,
    svc: ConfluenceService = Depends(_svc),
    _=Depends(get_current_user),
):
    return await svc.get_confluence(ticker)


@router.post("/batch")
async def get_confluence_batch(
    tickers: List[str],
    svc: ConfluenceService = Depends(_svc),
    _=Depends(get_current_user),
):
    return await svc.get_confluence_batch(tickers)
