import asyncpg
from datetime import datetime, timezone
from typing import List, Tuple
import logging
from app.services.news_service import NewsService
from app.services.finbert_service import FinBERTService

logger = logging.getLogger(__name__)


class SentimentService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.news_svc = NewsService()
        self.finbert_svc = FinBERTService()

    async def run_pipeline(
        self, tickers: List[str], limit: int = 10
    ) -> Tuple[List[str], List[str], int]:
        success, failed, total = [], [], 0
        for ticker in tickers:
            try:
                count = await self._process_ticker(ticker.upper(), limit)
                success.append(ticker.upper())
                total += count
                logger.info(f"✅ Sentiment done for {ticker}: {count} articles")
            except Exception as e:
                failed.append(ticker.upper())
                logger.error(f"❌ Sentiment failed for {ticker}: {e}")
        return success, failed, total

    async def _process_ticker(self, ticker: str, limit: int) -> int:
        raw_articles = await self.news_svc.fetch_articles(ticker, limit)
        if not raw_articles:
            return 0

        parsed = [self.news_svc.parse_article(a, ticker) for a in raw_articles]
        texts = [f"{a['title']}. {a['description']}".strip() for a in parsed]

        sentiments = await self.finbert_svc.classify_batch(texts)

        async with self.pool.acquire() as conn:
            for article, sentiment in zip(parsed, sentiments):
                published_at = article.get("published_at")
                await conn.execute(
                    """
                    INSERT INTO sentiment_articles (ticker, title, content, url, published_at, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (url) DO UPDATE SET
                        title = EXCLUDED.title, content = EXCLUDED.content, published_at = EXCLUDED.published_at
                    """,
                    ticker,
                    article.get("title"),
                    article.get("content"),
                    article.get("url"),
                    published_at,
                    datetime.now(timezone.utc),
                )

        await self._compute_snapshot(ticker, parsed, sentiments)
        return len(parsed)

    async def _compute_snapshot(self, ticker: str, articles: list, sentiments: list):
        if not sentiments:
            return
        n = len(sentiments)
        avg_pos = sum(s["positive_score"] for s in sentiments) / n
        avg_neg = sum(s["negative_score"] for s in sentiments) / n
        avg_neu = sum(s["neutral_score"] for s in sentiments) / n
        avg_cmp = sum(s["compound_score"] for s in sentiments) / n

        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for s in sentiments:
            counts[s["sentiment_label"]] += 1
        _ = max(counts, key=counts.get)

        dates = [a["published_at"] for a in articles if a.get("published_at")]
        _ = min(dates) if dates else datetime.now(timezone.utc)
        _ = max(dates) if dates else datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sentiment_snapshots (
                    ticker, avg_positive, avg_negative, avg_neutral, avg_compound, computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (ticker) DO UPDATE SET
                    avg_positive = EXCLUDED.avg_positive,
                    avg_negative = EXCLUDED.avg_negative,
                    avg_neutral = EXCLUDED.avg_neutral,
                    avg_compound = EXCLUDED.avg_compound,
                    computed_at = EXCLUDED.computed_at
                """,
                ticker,
                round(avg_pos, 6),
                round(avg_neg, 6),
                round(avg_neu, 6),
                round(avg_cmp, 6),
                datetime.now(timezone.utc),
            )

    async def get_articles(self, ticker: str, limit: int = 20) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT title, content, url, published_at, positive_score, negative_score, neutral_score, compound_score, sentiment_label
                FROM sentiment_articles
                WHERE ticker = $1
                ORDER BY published_at DESC
                LIMIT $2
                """,
                ticker.upper(),
                limit,
            )
            return [dict(r) for r in rows]

    async def get_latest_snapshot(self, ticker: str) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM sentiment_snapshots
                WHERE ticker = $1
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                ticker.upper(),
            )
            return dict(row) if row else None

    async def get_all_snapshots(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (ticker) * FROM sentiment_snapshots
                ORDER by ticker, computed_at DESC
                """
            )
            return [dict(r) for r in rows]

    async def get_sentiment_history(self, ticker: str, limit: int = 30) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM sentiment_snapshots
                WHERE ticker = $1
                ORDER BY computed_at DESC
                LIMIT $2
                """,
                ticker.upper(),
                limit,
            )
            return [dict(r) for r in rows]
