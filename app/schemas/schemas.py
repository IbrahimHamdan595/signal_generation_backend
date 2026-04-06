from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=64)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


# ── User ──────────────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: str
    full_name: str
    email: str
    is_active: bool
    is_admin: bool
    watchlist: List[str]
    created_at: datetime


class WatchlistUpdate(BaseModel):
    tickers: List[str] = Field(..., min_length=1)


# ── OHLCV ─────────────────────────────────────────────────────────────────────

class OHLCVResponse(BaseModel):
    ticker: str
    interval: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Indicators ────────────────────────────────────────────────────────────────

class IndicatorsResponse(BaseModel):
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


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    tickers: List[str] = Field(..., description="e.g. ['AAPL', 'MSFT']")
    interval: str = Field(default="1d", description="1d | 1h | 5m")
    period: str = Field(default="1y", description="1y | 6mo | 3mo")


class IngestResponse(BaseModel):
    success: List[str]
    failed: List[str]
    total_records: int
    message: str


# ── Generic ───────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    success: bool = True

# NOTE: SignalResponse/SignalRequest schemas live in Phase 4.