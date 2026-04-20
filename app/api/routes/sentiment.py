import logging
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List
from pydantic import BaseModel, Field

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.sentiment_service import SentimentService
from app.services.job_service import JobService
from app.schemas.sentiment import (
    SentimentFetchRequest,
    SentimentArticleResponse,
    SentimentSnapshotResponse,
    SentimentSummaryResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


class EnrichRequest(BaseModel):
    tickers: List[str] = Field(..., description="Tickers to enrich with AV historical sentiment")
    period_years: int = Field(default=2, ge=1, le=5, description="Years of history to fetch (2 AV calls per year per ticker)")


class BuildDailyRequest(BaseModel):
    tickers: List[str] = Field(..., description="Tickers to build daily sentiment for")
    years: int = Field(default=5, ge=1, le=10, description="Years of news history to fetch per ticker")


@router.post("/enrich", response_model=dict)
async def enrich_social_sentiment(
    body: EnrichRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Fetch historical Alpha Vantage sentiment for each ticker and write
    per-bar social_sentiment values into the indicators table.

    Returns a job_id immediately. Poll GET /jobs/{job_id} for progress.
    Requires ALPHAVANTAGE_KEY in .env. Free tier: 25 calls/day.
    """
    if len(body.tickers) > 10:
        raise HTTPException(400, "Max 10 tickers per enrich request (AV rate limit)")

    job_svc = JobService(pool)
    job_id = await job_svc.create("sentiment_enrich")

    async def _run():
        svc = SentimentService(pool)
        success, failed = [], []
        try:
            for i, ticker in enumerate(body.tickers):
                try:
                    # 1. Write per-bar social_sentiment into indicators table (AV daily series)
                    s, f = await svc.enrich_social_sentiment([ticker], body.period_years)
                    if s:
                        success.extend(s)
                        # 2. Also store recent articles + snapshots in sentiment_articles
                        try:
                            await svc.ingest_articles([ticker], recent_limit=50)
                            logger.info(f"✅ Articles stored for {ticker}")
                        except Exception as ae:
                            logger.warning(f"⚠️  Article storage failed for {ticker}: {ae}")
                    else:
                        failed.extend(f or [ticker.upper()])
                except Exception as e:
                    failed.append(ticker.upper())
                    logger.error(f"Enrich failed for {ticker}: {e}")

                await job_svc.update_progress(job_id, {
                    "done": len(success),
                    "failed": len(failed),
                    "total": len(body.tickers),
                    "current": i + 1,
                })

            await job_svc.complete(job_id, {
                "done": len(success),
                "failed": len(failed),
                "total": len(body.tickers),
                "success": success,
                "failed_tickers": failed,
            })
            logger.info(f"Enrich done — success: {success}, failed: {failed}")
        except Exception as e:
            await job_svc.fail(job_id, str(e))

    background_tasks.add_task(_run)
    return {
        "job_id": job_id,
        "message": f"Enrichment started for {len(body.tickers)} tickers ({body.period_years}y each)",
        "av_calls_needed": len(body.tickers) * body.period_years,
        "note": "Requires ALPHAVANTAGE_KEY. Free tier: 25 calls/day.",
    }


@router.post("/build-daily", response_model=dict)
async def build_daily_sentiment(
    body: BuildDailyRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Fetch company news from Finnhub (or AV fallback) for each ticker over
    `years` years, score every article with FinBERT, aggregate by trading day,
    and write the results into the daily_sentiment table.

    Also updates indicators.social_sentiment per bar for backward compatibility.

    Returns a job_id — poll GET /jobs/{job_id} for progress.
    Requires FINNHUB_API_KEY (preferred) or ALPHAVANTAGE_KEY in .env.
    """
    if len(body.tickers) > 20:
        raise HTTPException(400, "Max 20 tickers per request")

    job_svc = JobService(pool)
    job_id = await job_svc.create("daily_sentiment_build")

    async def _run():
        svc = SentimentService(pool)
        success, failed = [], []

        async def _progress(current: int, total: int, n_ok: int, n_fail: int):
            await job_svc.update_progress(job_id, {
                "current": current,
                "total": total,
                "done": n_ok,
                "failed": n_fail,
            })

        try:
            success, failed, _ = await svc.ingest_articles(
                body.tickers,
                years=body.years,
                progress_cb=_progress,
            )
            await job_svc.complete(job_id, {
                "done": len(success),
                "failed": len(failed),
                "total": len(body.tickers),
                "success": success,
                "failed_tickers": failed,
            })
            logger.info(f"Daily sentiment build done — success: {success}, failed: {failed}")
        except Exception as e:
            await job_svc.fail(job_id, str(e))
            logger.error(f"Daily sentiment build failed: {e}")

    background_tasks.add_task(_run)
    return {
        "job_id": job_id,
        "message": f"Daily sentiment build started for {len(body.tickers)} tickers ({body.years}y each)",
        "note": "Uses Finnhub (preferred) or Alpha Vantage. Runs FinBERT per article.",
    }


@router.post("/fetch", response_model=dict)
async def fetch_sentiment(
    body: SentimentFetchRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if len(body.tickers) > 20:
        raise HTTPException(400, "Max 20 tickers per request")
    svc = SentimentService(pool)
    success, failed, total = await svc.ingest_articles(body.tickers, recent_limit=body.limit)
    return {
        "success": success,
        "failed": failed,
        "total_articles": total,
        "message": f"Processed {total} articles for {len(success)} tickers",
    }


@router.get("/articles/{ticker}", response_model=List[SentimentArticleResponse])
async def get_articles(
    ticker: str,
    limit: int = Query(default=20, le=100),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    docs = await SentimentService(pool).get_articles(ticker.upper(), limit)
    if not docs:
        raise HTTPException(
            404, f"No articles for {ticker.upper()}. Run POST /sentiment/fetch first."
        )
    return [SentimentArticleResponse(**d) for d in docs]


@router.get("/snapshot/{ticker}", response_model=SentimentSnapshotResponse)
async def get_snapshot(
    ticker: str,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    doc = await SentimentService(pool).get_latest_snapshot(ticker.upper())
    if not doc:
        raise HTTPException(404, f"No snapshot for {ticker.upper()}")
    return SentimentSnapshotResponse(**doc)


@router.get(
    "/snapshot/{ticker}/history", response_model=List[SentimentSnapshotResponse]
)
async def get_snapshot_history(
    ticker: str,
    limit: int = Query(default=30, le=100),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    docs = await SentimentService(pool).get_sentiment_history(ticker.upper(), limit)
    if not docs:
        raise HTTPException(404, f"No history for {ticker.upper()}")
    return [SentimentSnapshotResponse(**d) for d in docs]


@router.get("/summary", response_model=List[SentimentSummaryResponse])
async def get_all_summaries(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    docs = await SentimentService(pool).get_all_snapshots()
    return [
        SentimentSummaryResponse(
            ticker=d["ticker"],
            avg_compound=d.get("avg_compound", 0.0),
            avg_positive=d.get("avg_positive", 0.0),
            avg_negative=d.get("avg_negative", 0.0),
            avg_neutral=d.get("avg_neutral", 0.0),
        )
        for d in docs
    ]


@router.get("/summary/{ticker}", response_model=SentimentSummaryResponse)
async def get_ticker_summary(
    ticker: str,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    doc = await SentimentService(pool).get_latest_snapshot(ticker.upper())
    if not doc:
        raise HTTPException(404, f"No sentiment data for {ticker.upper()}")
    return SentimentSummaryResponse(
        ticker=doc["ticker"],
        avg_compound=doc.get("avg_compound", 0.0),
        avg_positive=doc.get("avg_positive", 0.0),
        avg_negative=doc.get("avg_negative", 0.0),
        avg_neutral=doc.get("avg_neutral", 0.0),
    )
