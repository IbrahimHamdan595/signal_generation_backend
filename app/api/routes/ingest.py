from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.ohlcv_service import OHLCVService, VALID_INTERVALS, VALID_PERIODS
from app.schemas.schemas import IngestRequest, IngestResponse, MessageResponse

router = APIRouter(prefix="/ingest", tags=["Data Ingestion"])


@router.post("/", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if body.interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Choose from: {VALID_INTERVALS}")
    if body.period not in VALID_PERIODS:
        raise HTTPException(400, f"Invalid period. Choose from: {VALID_PERIODS}")
    if len(body.tickers) > 50:
        raise HTTPException(400, "Max 50 tickers per request")

    svc = OHLCVService(pool)
    success, failed, total = await svc.ingest_tickers(
        body.tickers, body.interval, body.period
    )

    return IngestResponse(
        success=success,
        failed=failed,
        total_records=total,
        message=f"Ingested {total} records for {len(success)} tickers",
    )


@router.post("/background", response_model=MessageResponse)
async def ingest_background(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if len(body.tickers) > 200:
        raise HTTPException(400, "Max 200 tickers for background ingestion")

    async def _run():
        await OHLCVService(pool).ingest_tickers(
            body.tickers, body.interval, body.period
        )

    background_tasks.add_task(_run)
    return MessageResponse(
        message=f"Background ingestion started for {len(body.tickers)} tickers"
    )


@router.get("/tickers", response_model=List[str])
async def available_tickers(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    return await OHLCVService(pool).get_available_tickers()


@router.get("/sp500", response_model=dict)
async def get_sp500_tickers(
    current_user=Depends(get_current_active_user),
):
    from app.services.news_service import get_sp500_map

    sp500 = get_sp500_map()
    return {
        "total": len(sp500),
        "tickers": sp500,
    }
