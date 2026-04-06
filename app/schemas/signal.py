from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


class SignalResponse(BaseModel):
    id: str
    ticker: str
    interval: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)

    entry_price: Optional[float] = Field(default=None, gt=0)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    net_profit: Optional[float] = None

    bars_to_entry: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    entry_time: Optional[datetime] = None

    probabilities: Optional[dict[str, float]] = None
    prob_buy: Optional[float] = None
    prob_sell: Optional[float] = None
    prob_hold: Optional[float] = None

    source: str = "ml_model"
    created_at: datetime


class SignalGenerateRequest(BaseModel):
    ticker: str
    interval: str = Field(default="1d", description="1d | 1h")


class SignalBatchRequest(BaseModel):
    tickers: List[str] = Field(
        ..., description="List of tickers to generate signals for"
    )
    interval: str = Field(default="1d")
