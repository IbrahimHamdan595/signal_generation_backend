from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone
from bson import ObjectId
from app.models.models import PyObjectId


# ── Sentiment Article ─────────────────────────────────────────────────────────


class SentimentArticleModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    ticker: str  # e.g. "AAPL"

    # Article metadata
    title: str
    description: Optional[str] = None
    url: str
    source: str  # e.g. "Reuters"
    published_at: datetime

    # Raw FinBERT output
    sentiment_label: str  # "positive" | "negative" | "neutral"
    positive_score: float
    negative_score: float
    neutral_score: float

    # Derived compound score  →  positive - negative  ∈ [-1, +1]
    compound_score: float

    # h_CLS embedding vector from FinBERT (768-dim) — stored for Phase 3 fusion
    embedding: Optional[List[float]] = None

    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# ── Aggregated Sentiment Snapshot ─────────────────────────────────────────────


class SentimentSnapshotModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    ticker: str
    window_start: datetime  # earliest article in this window
    window_end: datetime  # latest article in this window
    article_count: int

    avg_positive: float
    avg_negative: float
    avg_neutral: float
    avg_compound: float  # overall sentiment score for this window

    dominant_sentiment: str  # "positive" | "negative" | "neutral"
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
