from pydantic import model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = ""
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Supabase — project URL + service-role key for Storage access
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    NEWSAPI_KEY: str = ""
    HF_API_TOKEN: str = ""
    HF_FINBERT_URL: str = (
        "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
    )
    NEWS_FETCH_LIMIT: int = 10

    # Alpha Vantage — free tier: 25 calls/day, historical news with built-in sentiment
    ALPHAVANTAGE_KEY: str = ""

    # Finnhub — free tier: 60 req/min, company news with historical date ranges
    FINNHUB_API_KEY: str = ""

    # FRED (St. Louis Fed) — free API key at fred.stlouisfed.org/docs/api/api_key.html
    # Used for CPI and NFP release date calendars. Falls back to algorithmic if unset.
    FRED_API_KEY: str = ""

    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8501"

    APP_NAME: str = "Trading Signal API"
    APP_VERSION: str = "2.0.0"
    APP_DEBUG: bool = False

    TICKER_LIST_PATH: str = "data/sp500.json"
    MAX_BARS_TO_ENTRY: int = 30
    # Triple-barrier labeling — symmetric 2%/2% over 15 bars.
    # Previous asymmetric +2.5/-1.5 @ 10d skewed the class distribution
    # toward SELL (53% of labels) because the smaller bear barrier fired
    # more often, producing a model that predicted SELL 60% of the time
    # even on true BUY samples. Symmetric thresholds with a longer window
    # rebalance the classes and push HOLD toward majority so the model is
    # forced to learn high-conviction directional moves, not regime drift.
    LOOKAHEAD_WINDOW: int = 15
    BUY_THRESHOLD: float = 0.020
    SELL_THRESHOLD: float = 0.020

    # ── Signal execution risk parameters ──────────────────────────────────────
    # All applied at signal generation time in predict_ticker. The model's
    # raw prediction is filtered through these gates before being surfaced.
    UNCERTAINTY_MAX:      float = 0.15  # Skip if MC-dropout std > this (untrustworthy signal)
    MIN_RR_RATIO:         float = 2.0   # Require predicted_tp / predicted_sl >= this
    MIN_EXPECTED_VALUE:   float = 0.005 # 0.5% — minimum positive EV to take the trade
    EVENT_DAYS_MIN:       int   = 1     # Skip trades within N days of FOMC/CPI/NFP
    EARNINGS_DAYS_MIN:    int   = 2     # Skip trades within N days of earnings (gap risk)
    KELLY_FRACTION:       float = 0.25  # Quarter-Kelly bet sizing — conservative
    MAX_RISK_PER_TRADE:   float = 0.01  # 1% of capital risked per trade max
    MIN_PREDICTED_SL_PCT: float = 0.005 # 0.5% — floor on predicted SL (was 0.3% — too tight vs spread+vol)

    @model_validator(mode="after")
    def validate_required(self) -> "Settings":
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL is required in .env")
        if not self.DATABASE_URL.startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")

        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY is required in .env")
        if len(self.SECRET_KEY) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")

        return self

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
