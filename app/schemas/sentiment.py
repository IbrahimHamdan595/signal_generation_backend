from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── Request ───────────────────────────────────────────────────────────────────

class SentimentFetchRequest(BaseModel):
    tickers: List[str] = Field(..., description="e.g. ['AAPL', 'MSFT']")
    limit: int = Field(default=10, ge=1, le=50, description="Articles per ticker")


# ── Article Response ──────────────────────────────────────────────────────────

class SentimentArticleResponse(BaseModel):
    id: str
    ticker: str
    title: str
    description: Optional[str] = None
    url: str
    source: str
    published_at: datetime
    sentiment_label: str
    positive_score: float
    negative_score: float
    neutral_score: float
    compound_score: float
    ingested_at: datetime


# ── Snapshot Response ─────────────────────────────────────────────────────────

class SentimentSnapshotResponse(BaseModel):
    ticker: str
    window_start: datetime
    window_end: datetime
    article_count: int
    avg_positive: float
    avg_negative: float
    avg_neutral: float
    avg_compound: float
    dominant_sentiment: str
    computed_at: datetime


# ── Summary (lightweight, for dashboard) ─────────────────────────────────────

class SentimentSummaryResponse(BaseModel):
    ticker: str
    dominant_sentiment: str
    avg_compound: float                 # –1.0 (very negative) → +1.0 (very positive)
    article_count: int
    latest_article_at: Optional[datetime] = None