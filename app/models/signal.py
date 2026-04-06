from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from bson import ObjectId
from app.models.models import PyObjectId


class SignalModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    ticker: str
    interval: str  # "1d" | "1h"
    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: float

    # Price levels
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    net_profit: Optional[float] = None

    # Entry time prediction
    bars_to_entry: Optional[float] = None
    entry_time: Optional[datetime] = None  # next candle open timestamp
    entry_time_label: Optional[str] = None  # human-readable

    # Signal probabilities
    prob_buy: Optional[float] = None
    prob_sell: Optional[float] = None
    prob_hold: Optional[float] = None

    source: str = "ml_model"  # "ml_model" | "rule_based"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
