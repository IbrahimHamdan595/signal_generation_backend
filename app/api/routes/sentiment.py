from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List

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
