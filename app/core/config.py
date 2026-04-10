from pydantic import model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = ""
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    NEWSAPI_KEY: str = ""
    HF_API_TOKEN: str = ""
    HF_FINBERT_URL: str = (
        "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
    )
    NEWS_FETCH_LIMIT: int = 10

    # Alpha Vantage — free tier: 25 calls/day, historical news with built-in sentiment
    ALPHAVANTAGE_KEY: str = ""

    APP_NAME: str = "Trading Signal API"
    APP_VERSION: str = "2.0.0"
    APP_DEBUG: bool = False

    TICKER_LIST_PATH: str = "data/sp500.json"
    MAX_BARS_TO_ENTRY: int = 30
    # Triple-barrier labeling
    LOOKAHEAD_WINDOW: int = 10   # max bars to look ahead for a barrier touch
    BUY_THRESHOLD: float = 0.02  # +2% triggers BUY label
    SELL_THRESHOLD: float = 0.01 # -1% triggers SELL label (asymmetric: 2:1 RR)

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
