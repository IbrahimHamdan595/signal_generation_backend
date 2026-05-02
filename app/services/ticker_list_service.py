import json
import logging
from typing import List, Optional
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

class TickerListService:
    def __init__(self):
        self._cached_tickers: Optional[List[str]] = None
        self._custom_tickers: Optional[List[str]] = None

    async def get_tickers(self) -> List[str]:
        if self._custom_tickers:
            return self._custom_tickers
        if self._cached_tickers:
            return self._cached_tickers
        return await self.load_ticker_list()

    async def load_ticker_list(self) -> List[str]:
        path = Path(settings.TICKER_LIST_PATH)

        if not path.exists():
            logger.warning(f"⚠️  Ticker list not found: {path}")
            self._cached_tickers = []
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                tickers = [
                    item["ticker"] if isinstance(item, dict) else item for item in data
                ]
            elif isinstance(data, dict) and "tickers" in data:
                tickers = [
                    item["ticker"] if isinstance(item, dict) else item
                    for item in data["tickers"]
                ]
            else:
                logger.error("❌ Invalid sp500.json format")
                self._cached_tickers = []
                return []

            self._cached_tickers = tickers
            logger.info(f"✅ Loaded {len(tickers)} tickers from {path}")
            return tickers

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse {path}: {e}")
            self._cached_tickers = []
            return []
        except Exception as e:
            logger.error(f"❌ Failed to load ticker list: {e}")
            self._cached_tickers = []
            return []

    def set_custom_tickers(self, tickers: List[str]) -> None:
        self._custom_tickers = tickers
        logger.info(f"✅ Set custom ticker list: {len(tickers)} tickers")

    def clear_custom_tickers(self) -> None:
        self._custom_tickers = None
        logger.info("🗑 Cleared custom ticker list")

    def reload(self) -> None:
        self._cached_tickers = None
        logger.info("🔄 Cleared ticker cache")


ticker_list_service = TickerListService()
