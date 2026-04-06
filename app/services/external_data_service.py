import yfinance as yf
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExternalDataService:
    """
    Fetch external market data (VIX, options, earnings, sentiment).
    These enhance the signal generation with real-time market context.
    """

    @staticmethod
    async def get_vix_data() -> Dict[str, float]:
        """
        Fetch current VIX level and change from past 5 days.
        VIX = Volatility Index (fear gauge)
        """
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_history = vix_ticker.history(period="5d")

            if not vix_history.empty:
                current_vix = vix_history["Close"].iloc[-1]
                previous_vix = (
                    vix_history["Close"].iloc[-2]
                    if len(vix_history) > 1
                    else current_vix
                )
                vix_change = current_vix - previous_vix

                logger.info(f"✅ VIX: {current_vix:.2f} (change: {vix_change:.2f})")
                return {
                    "vix_level": float(current_vix),
                    "vix_change": float(vix_change),
                }
        except Exception as e:
            logger.warning(f"⚠️ VIX fetch error: {e}")

        return {"vix_level": 20.0, "vix_change": 0.0}

    @staticmethod
    async def get_sp500_correlation(ticker: str) -> Dict[str, float]:
        """
        Fetch S&P 500 performance for beta/correlation analysis.
        Useful to detect if a stock moves independently or with market.
        """
        try:
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="1y")

            if not sp500_data.empty:
                sp500_return = (
                    sp500_data["Close"].iloc[-1] - sp500_data["Close"].iloc[0]
                ) / sp500_data["Close"].iloc[0]
                logger.info(f"✅ S&P 500 YTD: {sp500_return * 100:.2f}%")
                return {
                    "sp500_ytd_return": float(sp500_return * 100),
                    "market_up": 1 if sp500_return > 0 else 0,
                }
        except Exception as e:
            logger.warning(f"⚠️ S&P 500 fetch error: {e}")

        return {"sp500_ytd_return": 0.0, "market_up": 0}

    @staticmethod
    async def get_earnings_data(ticker: str) -> Dict[str, Optional[int]]:
        """
        Placeholder for earnings calendar data.
        TODO: Integrate with Alpha Vantage, Polygon.io, or IEX Cloud API

        These APIs provide:
        - Days until next earnings
        - Next earnings date
        - EPS surprise history
        """
        # Future integration points:
        # - Alpha Vantage: earnings_calendar endpoint
        # - Polygon.io: /vX/reference/financials endpoint
        # - IEX Cloud: earnings endpoint

        logger.info(
            f"⏳ Earnings data: {ticker} (placeholder - integrate with earnings API)"
        )
        return {
            "earnings_days": 30,  # Default placeholder
            "next_earnings_date": None,
        }

    @staticmethod
    async def get_social_sentiment(ticker: str) -> Dict[str, float]:
        """
        Placeholder for social media sentiment analysis.
        TODO: Integrate with Twitter API, Reddit, StockTwits, Seeking Alpha

        Options:
        - Twitter API: #ticker mentions, sentiment
        - PRAW (Reddit): r/investing, r/stocks analysis
        - StockTwits API: Bullish/Bearish sentiment
        - Newspaper3k + TextBlob: News sentiment
        """
        # Future integration:
        # - tweepy for Twitter
        # - praw for Reddit
        # - requests for StockTwits API

        logger.info(
            f"⏳ Social sentiment: {ticker} (placeholder - integrate with social APIs)"
        )
        return {
            "social_sentiment": 0.0,  # -1 (bearish) to 1 (bullish)
            "social_volume": 0,
        }

    @staticmethod
    async def get_options_sentiment(ticker: str) -> Dict[str, float]:
        """
        Placeholder for options market sentiment.
        TODO: Integrate with options data providers (Polygon.io, IEX Cloud, yfinance)

        Key metrics:
        - Put/Call Ratio: >1 = bearish, <1 = bullish
        - IV (Implied Volatility): High = uncertainty, Low = confidence
        - Max Pain: Options prices suggest likely price target
        """
        # Future integration:
        # - yfinance options chains (basic)
        # - Polygon.io options data (advanced)
        # - cboe.com data (official put/call ratios)

        logger.info(
            f"⏳ Options sentiment: {ticker} (placeholder - integrate with options APIs)"
        )
        return {
            "options_put_call_ratio": 1.0,  # 1.0 = neutral
            "implied_volatility": 0.0,
        }

    @staticmethod
    async def get_macro_data() -> Dict[str, float]:
        """
        Fetch macro indicators that affect all stocks.
        Placeholder for deeper market analysis.

        Would fetch:
        - Yield curve (10Y - 2Y spread)
        - Treasury rates
        - USD Index
        - Commodity prices (Oil, Gold)
        """
        logger.info("⏳ Macro data: (placeholder - integrate with macro APIs)")
        return {
            "yield_curve_spread": 0.0,
            "usd_index": 100.0,
            "oil_price": 0.0,
            "gold_price": 0.0,
        }

    @staticmethod
    async def enrich_indicators(ticker: str, indicators: Dict) -> Dict:
        """
        Enrich a ticker's indicators with all external data.
        """
        async_tasks = {
            "vix": await ExternalDataService.get_vix_data(),
            "sp500": await ExternalDataService.get_sp500_correlation(ticker),
            "earnings": await ExternalDataService.get_earnings_data(ticker),
            "social": await ExternalDataService.get_social_sentiment(ticker),
            "options": await ExternalDataService.get_options_sentiment(ticker),
            "macro": await ExternalDataService.get_macro_data(),
        }

        # Merge all external data into indicators
        enriched = {
            **indicators,
            **async_tasks["vix"],
            **async_tasks["sp500"],
            **async_tasks["earnings"],
            **async_tasks["social"],
            **async_tasks["options"],
            **async_tasks["macro"],
        }

        logger.info(f"✅ Indicators enriched for {ticker}")
        return enriched
