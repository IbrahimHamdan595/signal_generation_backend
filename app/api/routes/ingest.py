from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.ohlcv_service import OHLCVService, VALID_INTERVALS, VALID_PERIODS
from app.services.job_service import JobService
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
    success, failed, total, failed_details = await svc.ingest_tickers(
        body.tickers, body.interval, body.period
    )

    return IngestResponse(
        success=success,
        failed=failed,
        total_records=total,
        message=f"Ingested {total} records for {len(success)} tickers",
        failed_details=failed_details,
    )


@router.post("/background")
async def ingest_background(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Fire-and-forget ingest that immediately returns a job_id.
    Poll GET /jobs/{job_id} to track progress.
    """
    if len(body.tickers) > 500:
        raise HTTPException(400, "Max 500 tickers for background ingestion")

    job_svc = JobService(pool)
    job_id = await job_svc.create("ingest")

    async def _run():
        import logging
        log = logging.getLogger(__name__)
        ohlcv_svc = OHLCVService(pool)
        success, failed = [], []
        failed_details: list[dict] = []
        total_records = 0

        try:
            for i, ticker in enumerate(body.tickers):
                t = ticker.upper()
                try:
                    # ON CONFLICT DO UPDATE — same bar overwrites itself, no duplicates
                    count = await ohlcv_svc._ingest_single(t, body.interval, body.period)
                    success.append(t)
                    total_records += count
                    log.info(f"✅ {t}: {count} bars upserted")

                except Exception as e:
                    log.error(f"❌ Ingest failed for {t}: {e}", exc_info=True)
                    failed.append(t)
                    failed_details.append({"ticker": t, "error": str(e)})

                await job_svc.update_progress(job_id, {
                    "done": len(success),
                    "failed": len(failed),
                    "total": len(body.tickers),
                    "total_records": total_records,
                    "current": i + 1,
                })

            await job_svc.complete(job_id, {
                "done": len(success),
                "failed": len(failed),
                "total": len(body.tickers),
                "total_records": total_records,
                "success": success,
                "failed_tickers": failed,
                "failed_details": failed_details,
            })
        except Exception as e:
            await job_svc.fail(job_id, str(e))

    background_tasks.add_task(_run)
    return {"job_id": job_id, "message": f"Background ingestion started for {len(body.tickers)} tickers"}


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
