from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.ohlcv_service import OHLCVService
from app.schemas.schemas import OHLCVResponse, IndicatorsResponse

router = APIRouter(prefix="/market", tags=["Market Data"])


@router.get("/ohlcv/{ticker}", response_model=List[OHLCVResponse])
async def get_ohlcv(
    ticker: str,
    interval: str = Query(default="1d", description="1d | 1h | 5m"),
    limit: int = Query(default=100, le=1000),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = OHLCVService(pool)
    data = await svc.get_ohlcv(ticker.upper(), interval, limit)

    if not data:
        raise HTTPException(
            404,
            f"No OHLCV data found for {ticker.upper()}. "
            "Use POST /api/v1/ingest to load data first.",
        )
    return data


@router.get("/indicators/{ticker}", response_model=List[IndicatorsResponse])
async def get_indicators(
    ticker: str,
    interval: str = Query(default="1d"),
    limit: int = Query(default=100, le=1000),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = OHLCVService(pool)
    data = await svc.get_indicators(ticker.upper(), interval, limit)

    if not data:
        raise HTTPException(404, f"No indicator data found for {ticker.upper()}")
    return data


@router.get("/indicators/{ticker}/latest", response_model=IndicatorsResponse)
async def get_latest_indicators(
    ticker: str,
    interval: str = Query(default="1d"),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = OHLCVService(pool)
    doc = await svc.get_latest_indicator(ticker.upper(), interval)

    if not doc:
        raise HTTPException(404, f"No indicator data found for {ticker.upper()}")
    return doc


@router.get("/summary/{ticker}")
async def get_ticker_summary(
    ticker: str,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = OHLCVService(pool)

    ohlcv = await svc.get_ohlcv(ticker.upper(), interval="1d", limit=1)
    indicators = await svc.get_latest_indicator(ticker.upper(), interval="1d")

    if not ohlcv:
        raise HTTPException(404, f"No data for {ticker.upper()}")

    latest_bar = ohlcv[0] if ohlcv else {}

    return {
        "ticker": ticker.upper(),
        "latest_bar": latest_bar,
        "indicators": indicators,
    }
