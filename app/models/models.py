from pydantic import BaseModel, Field, EmailStr, GetCoreSchemaHandler
from typing import Optional, List
from datetime import datetime, timezone
from bson import ObjectId
from pydantic_core import CoreSchema, core_schema


class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls.validate,
            serialization=core_schema.to_string_ser_schema(),
        )


# ── User ──────────────────────────────────────────────────────────────────────

class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    full_name: str
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    watchlist: List[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# ── OHLCV ─────────────────────────────────────────────────────────────────────

class OHLCVModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    ticker: str
    interval: str          # "1d" | "1h" | "5m"
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# ── Technical Indicators ──────────────────────────────────────────────────────

class IndicatorsModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    ticker: str
    interval: str
    timestamp: datetime

    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr_14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_bandwidth: Optional[float] = None
    obv: Optional[float] = None
    mfi_14: Optional[float] = None
    volume_roc: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    day_of_week: Optional[int] = None
    day_of_month: Optional[int] = None
    month: Optional[int] = None
    is_trading_day: Optional[bool] = None
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    pivot: Optional[float] = None
    resistance_1: Optional[float] = None
    support_1: Optional[float] = None
    resistance_2: Optional[float] = None
    support_2: Optional[float] = None
    price_sma20_dist: Optional[float] = None
    price_sma50_dist: Optional[float] = None
    high_vol_regime: Optional[bool] = None
    above_sma50: Optional[bool] = None
    above_sma200: Optional[bool] = None
    normalized_volatility: Optional[float] = None
    bb_position: Optional[float] = None
    roc_5: Optional[float] = None
    roc_10: Optional[float] = None
    higher_high: Optional[bool] = None
    lower_low: Optional[bool] = None
    price_change_pct: Optional[float] = None
    volume_above_avg: Optional[bool] = None
    vix_level: Optional[float] = None
    vix_change: Optional[float] = None
    earnings_days: Optional[int] = None
    social_sentiment: Optional[float] = None
    options_put_call_ratio: Optional[float] = None
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

