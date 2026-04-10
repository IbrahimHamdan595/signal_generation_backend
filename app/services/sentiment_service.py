import asyncpg
from datetime import datetime, timezone
from typing import List, Tuple
import logging

from app.services.news_service import NewsService, AlphaVantageNewsService
from app.services.finbert_service import FinBERTService
from app.core.config import settings

logger = logging.getLogger(__name__)


class SentimentService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.news_svc = NewsService()
        self.finbert_svc = FinBERTService()
        self.av_svc = AlphaVantageNewsService()

    # ── Public pipeline ───────────────────────────────────────────────────────

    async def run_pipeline(
        self, tickers: List[str], limit: int = 10
    ) -> Tuple[List[str], List[str], int]:
        """
        For each ticker:
          • If ALPHAVANTAGE_KEY is set → use Alpha Vantage (articles include
            pre-computed sentiment, no FinBERT call needed).
          • Otherwise → fall back to NewsAPI + FinBERT.
        """
        success, failed, total = [], [], 0
        for ticker in tickers:
            try:
                if self.av_svc.is_configured():
                    count = await self._process_ticker_av(ticker.upper(), limit)
                else:
                    count = await self._process_ticker_finbert(ticker.upper(), limit)
                success.append(ticker.upper())
                total += count
                logger.info(f"✅ Sentiment done for {ticker}: {count} articles")
            except Exception as e:
                failed.append(ticker.upper())
                logger.error(f"❌ Sentiment failed for {ticker}: {e}")
        return success, failed, total

    async def enrich_social_sentiment(
        self, tickers: List[str], period_years: int = 2
    ) -> Tuple[List[str], List[str]]:
        """
        Fetch a per-day compound-sentiment series from Alpha Vantage for each
        ticker and write the values into the `social_sentiment` column of the
        `indicators` table (matched by ticker + date).

        This replaces the hardcoded 0.0 stub with real historical sentiment
        aligned to every price bar — the key missing piece for training.

        Requires ALPHAVANTAGE_KEY. Free tier: 25 calls/day.
        Each ticker uses `period_years` calls (default 2 → 2 calls/ticker).
        """
        if not self.av_svc.is_configured():
            logger.warning("ALPHAVANTAGE_KEY not set — skipping social sentiment enrichment")
            return [], list(tickers)

        success, failed = [], []
        for ticker in tickers:
            try:
                daily_scores = await self.av_svc.fetch_daily_sentiment_series(
                    ticker.upper(), period_years=period_years
                )
                if not daily_scores:
                    logger.warning(f"⚠️  No AV sentiment data for {ticker}")
                    failed.append(ticker.upper())
                    continue

                rows_updated = await self._write_social_sentiment(
                    ticker.upper(), daily_scores
                )
                logger.info(
                    f"✅ Social sentiment written for {ticker}: "
                    f"{rows_updated} bars updated from {len(daily_scores)} scored days"
                )
                success.append(ticker.upper())
            except Exception as e:
                logger.error(f"❌ Social sentiment enrichment failed for {ticker}: {e}")
                failed.append(ticker.upper())

        return success, failed

    # ── Internal: Alpha Vantage path ──────────────────────────────────────────

    async def _process_ticker_av(self, ticker: str, limit: int) -> int:
        articles = await self.av_svc.fetch_articles(ticker, limit=limit)
        if not articles:
            return 0
        await self._store_articles_with_sentiment(ticker, articles)
        await self._compute_snapshot_from_articles(ticker, articles)
        return len(articles)

    # ── Internal: NewsAPI + FinBERT path ──────────────────────────────────────

    async def _process_ticker_finbert(self, ticker: str, limit: int) -> int:
        raw_articles = await self.news_svc.fetch_articles(ticker, limit)
        if not raw_articles:
            return 0

        parsed = [self.news_svc.parse_article(a, ticker) for a in raw_articles]
        texts = [f"{a['title']}. {a['description']}".strip() for a in parsed]
        sentiments = await self.finbert_svc.classify_batch(texts)

        enriched = [{**article, **sentiment}
                    for article, sentiment in zip(parsed, sentiments)]

        await self._store_articles_with_sentiment(ticker, enriched)
        await self._compute_snapshot_from_articles(ticker, enriched)
        return len(enriched)

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _store_articles_with_sentiment(
        self, ticker: str, articles: List[dict]
    ):
        async with self.pool.acquire() as conn:
            for a in articles:
                published_at = a.get("published_at")
                if isinstance(published_at, datetime) and published_at.tzinfo:
                    published_at = published_at.replace(tzinfo=None)

                await conn.execute(
                    """
                    INSERT INTO sentiment_articles (
                        ticker, title, description, url, source, published_at,
                        sentiment_label, positive_score, negative_score,
                        neutral_score, compound_score, ingested_at
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    ON CONFLICT (url) DO UPDATE SET
                        sentiment_label = EXCLUDED.sentiment_label,
                        positive_score  = EXCLUDED.positive_score,
                        negative_score  = EXCLUDED.negative_score,
                        neutral_score   = EXCLUDED.neutral_score,
                        compound_score  = EXCLUDED.compound_score
                    """,
                    ticker,
                    a.get("title", ""),
                    a.get("description", ""),
                    a.get("url", ""),
                    a.get("source", "Unknown"),
                    published_at,
                    a.get("sentiment_label", "neutral"),
                    float(a.get("positive_score", 0.0)),
                    float(a.get("negative_score", 0.0)),
                    float(a.get("neutral_score",  1.0)),
                    float(a.get("compound_score", 0.0)),
                    datetime.now(timezone.utc),
                )

    async def _compute_snapshot_from_articles(
        self, ticker: str, articles: List[dict]
    ):
        if not articles:
            return
        n = len(articles)
        avg_pos = sum(float(a.get("positive_score", 0)) for a in articles) / n
        avg_neg = sum(float(a.get("negative_score", 0)) for a in articles) / n
        avg_neu = sum(float(a.get("neutral_score",  1)) for a in articles) / n
        avg_cmp = sum(float(a.get("compound_score", 0)) for a in articles) / n

        counts: dict[str, int] = {}
        for a in articles:
            lbl = a.get("sentiment_label", "neutral")
            counts[lbl] = counts.get(lbl, 0) + 1
        dominant = max(counts, key=counts.get)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sentiment_snapshots (
                    ticker, avg_positive, avg_negative, avg_neutral,
                    avg_compound, dominant_sentiment, article_count, computed_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (ticker) DO UPDATE SET
                    avg_positive       = EXCLUDED.avg_positive,
                    avg_negative       = EXCLUDED.avg_negative,
                    avg_neutral        = EXCLUDED.avg_neutral,
                    avg_compound       = EXCLUDED.avg_compound,
                    dominant_sentiment = EXCLUDED.dominant_sentiment,
                    article_count      = EXCLUDED.article_count,
                    computed_at        = EXCLUDED.computed_at
                """,
                ticker,
                round(avg_pos, 6),
                round(avg_neg, 6),
                round(avg_neu, 6),
                round(avg_cmp, 6),
                dominant,
                n,
                datetime.now(timezone.utc),
            )

    async def _write_social_sentiment(
        self, ticker: str, daily_scores: dict
    ) -> int:
        """
        Update social_sentiment in the indicators table bar-by-bar.
        daily_scores: {"2023-10-15": 0.32, ...}
        """
        rows_updated = 0
        async with self.pool.acquire() as conn:
            for date_str, score in daily_scores.items():
                result = await conn.execute(
                    """
                    UPDATE indicators
                       SET social_sentiment = $1
                     WHERE ticker = $2
                       AND DATE(timestamp) = $3::date
                    """,
                    round(float(score), 6),
                    ticker,
                    date_str,
                )
                try:
                    rows_updated += int(result.split()[-1])
                except Exception:
                    pass
        return rows_updated

    # ── Read helpers ──────────────────────────────────────────────────────────

    async def get_articles(self, ticker: str, limit: int = 20) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, ticker, title, description, url, source,
                       published_at, sentiment_label, positive_score,
                       negative_score, neutral_score, compound_score, ingested_at
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
                ORDER BY ticker, computed_at DESC
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
