from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List
from pydantic import BaseModel, Field

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.sentiment_service import SentimentService
from app.schemas.sentiment import (
    SentimentFetchRequest,
    SentimentArticleResponse,
    SentimentSnapshotResponse,
    SentimentSummaryResponse,
)

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


class EnrichRequest(BaseModel):
    tickers: List[str] = Field(..., description="Tickers to enrich with AV historical sentiment")
    period_years: int = Field(default=2, ge=1, le=5, description="Years of history to fetch (2 AV calls per year per ticker)")


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

    Runs in the background. Requires ALPHAVANTAGE_KEY in .env.
    Free tier: 25 calls/day — each ticker uses `period_years` calls.
    """
    if len(body.tickers) > 10:
        raise HTTPException(400, "Max 10 tickers per enrich request (AV rate limit)")

    async def _run():
        svc = SentimentService(pool)
        success, failed = await svc.enrich_social_sentiment(
            body.tickers, body.period_years
        )
        import logging
        logging.getLogger(__name__).info(
            f"Enrich done — success: {success}, failed: {failed}"
        )

    background_tasks.add_task(_run)
    return {
        "message": f"Enrichment started for {len(body.tickers)} tickers ({body.period_years}y each)",
        "tickers": body.tickers,
        "av_calls_needed": len(body.tickers) * body.period_years,
        "note": "Requires ALPHAVANTAGE_KEY. Free tier: 25 calls/day.",
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
    success, failed, total = await svc.run_pipeline(body.tickers, body.limit)
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
