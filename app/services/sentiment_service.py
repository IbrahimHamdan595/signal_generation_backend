import asyncpg
from collections import defaultdict
from datetime import datetime, timezone, timedelta, date as date_type
from typing import List, Tuple, Dict, Optional
import logging

from app.services.news_service import NewsService, AlphaVantageNewsService, FinnhubNewsService, EdgarNewsService
from app.services.finbert_service import FinBERTService
from app.core.asset_class import is_fx

logger = logging.getLogger(__name__)


class SentimentService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.news_svc = NewsService()
        self.finbert_svc = FinBERTService()
        self.av_svc = AlphaVantageNewsService()
        self.finnhub_svc = FinnhubNewsService()
        self.edgar_svc = EdgarNewsService()

    # ── Public pipeline ───────────────────────────────────────────────────────

    async def ingest_articles(
        self,
        tickers: List[str],
        *,
        years: float = 5.0,
        recent_limit: Optional[int] = None,
        progress_cb=None,
    ) -> Tuple[List[str], List[str], int]:
        """
        Unified article ingestion pipeline.

        recent_limit=N  (6h scheduler refresh / POST /sentiment/fetch):
            Fetch the last N articles per ticker from Alpha Vantage (if
            configured) or NewsAPI, store raw, score unscored with FinBERT,
            aggregate into daily_sentiment, sync indicators, update snapshot.

        recent_limit=None (Pipeline page / POST /sentiment/build-daily):
            Full ``years``-year history via Finnhub + EDGAR (preferred) or
            Alpha Vantage daily-series fallback.

        In both modes articles are stored with ON CONFLICT DO NOTHING, then
        only NULL-scored articles are sent to FinBERT — so a re-run never
        overwrites existing scores and is safe to resume after a crash.

        Returns (success_tickers, failed_tickers, total_count) where
        total_count is articles newly stored (recent mode) or days written
        (full-history mode).
        """
        success, failed, total_count = [], [], 0
        n = len(tickers)

        for i, ticker in enumerate(tickers):
            try:
                t = ticker.upper()
                # FX/metal pairs don't have per-pair news coverage in any of
                # the configured sources (Finnhub company-news, EDGAR filings,
                # Alpha Vantage equity news). Skip them entirely — the
                # dataset builder falls back to zero-vectors when
                # daily_sentiment has no rows for a ticker.
                if is_fx(t):
                    success.append(t)
                    if progress_cb is not None:
                        progress_cb(i + 1, n, t)
                    continue
                if recent_limit is not None:
                    count = await self._ingest_recent(t, recent_limit)
                else:
                    # articles: ON CONFLICT DO NOTHING — same URL never inserted twice
                    # FinBERT: only scores rows where sentiment_label IS NULL — no re-scoring
                    count = await self._ingest_full_history(t, years)
                success.append(t)
                total_count += count
                unit = "articles" if recent_limit is not None else "days"
                logger.info(f"✅ Sentiment done for {ticker}: {count} {unit}")
            except Exception as e:
                failed.append(ticker.upper())
                logger.error(f"❌ Sentiment failed for {ticker}: {e}")

            if progress_cb:
                await progress_cb(i + 1, n, len(success), len(failed))

        return success, failed, total_count

    async def enrich_social_sentiment(
        self, tickers: List[str], period_years: int = 2
    ) -> Tuple[List[str], List[str]]:
        """
        Pull Alpha Vantage's pre-aggregated daily compound-sentiment series
        and write per-bar social_sentiment values into the indicators table.

        This is a separate data path from article ingestion: AV provides one
        compound score per date (no individual articles), so FinBERT is not
        involved. Requires ALPHAVANTAGE_KEY. Free tier: 25 calls/day;
        each ticker uses ``period_years`` API calls.
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

                rows_updated = await self._write_social_sentiment(ticker.upper(), daily_scores)
                logger.info(
                    f"✅ Social sentiment written for {ticker}: "
                    f"{rows_updated} bars updated from {len(daily_scores)} scored days"
                )
                success.append(ticker.upper())
            except Exception as e:
                logger.error(f"❌ Social sentiment enrichment failed for {ticker}: {e}")
                failed.append(ticker.upper())

        return success, failed

    # ── Private: ingest paths ─────────────────────────────────────────────────

    async def _ingest_recent(self, ticker: str, limit: int) -> int:
        """
        Fetch the last ``limit`` articles from AV (preferred) or NewsAPI.
        Store raw → score unscored with FinBERT → aggregate → sync → snapshot.
        Returns the number of newly stored articles.
        """
        if self.av_svc.is_configured():
            articles = await self.av_svc.fetch_articles(ticker, limit=limit)
        else:
            raw = await self.news_svc.fetch_articles(ticker, limit)
            articles = [self.news_svc.parse_article(a, ticker) for a in raw]

        if not articles:
            return 0

        new_count = await self._store_articles_raw(ticker, articles)
        await self._score_unscored_articles(ticker)
        await self._aggregate_daily_from_db(ticker)
        await self._sync_social_sentiment_from_daily(ticker)
        await self._compute_snapshot_from_db(ticker)
        return new_count

    async def _ingest_full_history(self, ticker: str, years: float) -> int:
        """
        Full history pipeline: Finnhub + EDGAR (preferred) or AV daily-series
        fallback when Finnhub is not configured.
        Returns the number of days written to daily_sentiment.
        """
        if self.finnhub_svc.is_configured():
            return await self._ingest_finnhub_edgar(ticker, years)
        elif self.av_svc.is_configured():
            return await self._build_daily_av(ticker, min(int(years), 2))
        else:
            raise ValueError(
                "No news API key configured — set FINNHUB_API_KEY or ALPHAVANTAGE_KEY in .env"
            )

    async def _ingest_finnhub_edgar(self, ticker: str, years: float) -> int:
        """
        Hybrid pipeline with crash recovery:
          1. Finnhub  → recent news (~1 year on free tier)
          2. EDGAR    → 8-K / 10-Q filings for the historical gap
          3. Store ALL raw articles (ON CONFLICT DO NOTHING) — idempotent
          4. Score unscored articles with FinBERT, persisting after each chunk
          5. Aggregate scored articles → daily_sentiment
          6. Sync avg_compound → indicators.social_sentiment
          7. Update sentiment_snapshots

        Returns the number of daily_sentiment rows written.
        """
        now = datetime.now(timezone.utc)
        full_start = (now.replace(tzinfo=None) - timedelta(days=365 * years)).strftime("%Y-%m-%d")

        all_articles: list = []

        # Phase 1: Finnhub
        logger.info(f"📰 [{ticker}] Phase 1: Finnhub news fetch (up to {years}y)")
        finnhub_articles, finnhub_cutoff = await self.finnhub_svc.fetch_date_range(
            ticker, years=years
        )
        all_articles.extend(finnhub_articles)
        logger.info(
            f"📰 [{ticker}] Finnhub: {len(finnhub_articles)} articles, "
            f"coverage back to {finnhub_cutoff}"
        )

        # Phase 2: EDGAR gap-fill
        if finnhub_cutoff > full_start:
            logger.info(
                f"📄 [{ticker}] Phase 2: EDGAR filings for gap "
                f"{full_start} → {finnhub_cutoff}"
            )
            edgar_articles = await self.edgar_svc.fetch_date_range(
                ticker, from_date=full_start, to_date=finnhub_cutoff,
            )
            all_articles.extend(edgar_articles)
            logger.info(f"📄 [{ticker}] EDGAR: {len(edgar_articles)} filings collected")
        else:
            logger.info(f"📄 [{ticker}] Finnhub covered full {years}y — EDGAR not needed")

        if not all_articles:
            logger.warning(f"⚠️  [{ticker}] No articles from Finnhub or EDGAR")
            # Aggregate from any previously stored articles (restart recovery)
            days_written = await self._aggregate_daily_from_db(ticker)
            if days_written:
                await self._sync_social_sentiment_from_daily(ticker)
                await self._compute_snapshot_from_db(ticker)
            return days_written

        # Phase 3: Store raw (idempotent)
        new_count = await self._store_articles_raw(ticker, all_articles)
        logger.info(
            f"💾 [{ticker}] {new_count} new articles stored "
            f"({len(all_articles) - new_count} already existed)"
        )

        # Phase 4: Score unscored (resumable)
        scored_count = await self._score_unscored_articles(ticker)
        logger.info(f"🔬 [{ticker}] FinBERT scored {scored_count} articles")

        # Phase 5: Aggregate → daily_sentiment
        days_written = await self._aggregate_daily_from_db(ticker)

        # Phase 6: Sync → indicators.social_sentiment
        await self._sync_social_sentiment_from_daily(ticker)

        # Phase 7: Update sentiment_snapshots
        await self._compute_snapshot_from_db(ticker)

        edgar_count = len(all_articles) - len(finnhub_articles)
        logger.info(
            f"✅ [{ticker}] Daily sentiment complete: "
            f"{len(finnhub_articles)} Finnhub + {edgar_count} EDGAR "
            f"→ {days_written} trading days written"
        )
        return days_written

    async def _build_daily_av(self, ticker: str, years: int) -> int:
        """
        AV fallback (no Finnhub key): fetch the historical compound-sentiment
        series, derive pos/neg/neu, write to daily_sentiment and sync to
        indicators. No individual articles are stored.
        """
        daily_scores = await self.av_svc.fetch_daily_sentiment_series(ticker, period_years=years)
        if not daily_scores:
            return 0

        by_date: Dict[str, List[dict]] = {}
        for day, compound in daily_scores.items():
            pos = max(0.0, compound)
            neg = max(0.0, -compound)
            neu = round(1.0 - abs(compound), 6)
            by_date[day] = [{
                "positive_score": pos,
                "negative_score": neg,
                "neutral_score":  neu,
                "compound_score": compound,
            }]

        days_written = await self._write_daily_sentiment(ticker, by_date)
        await self._write_social_sentiment(ticker, daily_scores)
        return days_written

    # ── DB write helpers ──────────────────────────────────────────────────────

    async def _store_articles_raw(self, ticker: str, articles: list) -> int:
        """
        Insert articles with NULL sentiment scores (not yet scored).
        Uses executemany (one DB round-trip) + ON CONFLICT DO NOTHING on url.
        Returns the number of newly inserted rows.
        """
        now = datetime.now(timezone.utc)
        records = []
        for a in articles:
            published_at = a.get("published_at")
            if isinstance(published_at, datetime) and published_at.tzinfo:
                published_at = published_at.replace(tzinfo=None)
            records.append((
                ticker,
                a.get("title", ""),
                a.get("description", ""),
                a.get("url", "") or f"__no_url_{ticker}_{len(records)}__",
                a.get("source", "Unknown"),
                published_at,
                now,
            ))

        if not records:
            return 0

        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO sentiment_articles
                    (ticker, title, description, url, source, published_at, ingested_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (url) DO NOTHING
                """,
                records,
            )
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM sentiment_articles "
                "WHERE ticker = $1 AND ingested_at >= $2",
                ticker, now,
            )
        return row["n"] if row else 0

    async def _score_unscored_articles(self, ticker: str) -> int:
        """
        Fetch articles with NULL sentiment_label for this ticker, score them
        with FinBERT in chunks of BATCH_SIZE, and write scores to DB after
        each chunk — so a server restart resumes from where it left off.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, description
                FROM sentiment_articles
                WHERE ticker = $1 AND sentiment_label IS NULL
                ORDER BY published_at ASC
                """,
                ticker,
            )

        if not rows:
            logger.info(f"[{ticker}] All articles already scored — nothing to do")
            return 0

        total = len(rows)
        CHUNK = self.finbert_svc.BATCH_SIZE
        logger.info(
            f"🔬 [{ticker}] Scoring {total} unscored articles with FinBERT "
            f"(batch size={CHUNK}, "
            f"~{(total + CHUNK - 1) // CHUNK} API calls)"
        )

        scored = 0
        for chunk_start in range(0, total, CHUNK):
            chunk = rows[chunk_start : chunk_start + CHUNK]
            texts = [
                (
                    f"{r['title']}. {r['description']}".strip()
                    if r["description"]
                    else (r["title"] or "")
                )[:512]
                for r in chunk
            ]

            sentiments = await self.finbert_svc.classify_batch(texts)

            async with self.pool.acquire() as conn:
                for row, sent in zip(chunk, sentiments):
                    await conn.execute(
                        """
                        UPDATE sentiment_articles SET
                            sentiment_label = $1,
                            positive_score  = $2,
                            negative_score  = $3,
                            neutral_score   = $4,
                            compound_score  = $5
                        WHERE id = $6
                        """,
                        sent["sentiment_label"],
                        sent["positive_score"],
                        sent["negative_score"],
                        sent["neutral_score"],
                        sent["compound_score"],
                        row["id"],
                    )

            scored += len(chunk)
            logger.info(f"🔬 [{ticker}] Progress: {scored}/{total} articles scored")

        return scored

    async def _aggregate_daily_from_db(self, ticker: str) -> int:
        """
        Read all scored articles for this ticker from the DB, aggregate by
        calendar date, and upsert into daily_sentiment.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT published_at, positive_score, negative_score,
                       neutral_score, compound_score
                FROM sentiment_articles
                WHERE ticker = $1
                  AND sentiment_label IS NOT NULL
                  AND published_at IS NOT NULL
                ORDER BY published_at ASC
                """,
                ticker,
            )

        by_date: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            pub = r["published_at"]
            if pub:
                day = pub.strftime("%Y-%m-%d") if hasattr(pub, "strftime") else str(pub)[:10]
                by_date[day].append({
                    "positive_score": r["positive_score"] or 0.0,
                    "negative_score": r["negative_score"] or 0.0,
                    "neutral_score":  r["neutral_score"]  or 1.0,
                    "compound_score": r["compound_score"] or 0.0,
                })

        return await self._write_daily_sentiment(ticker, by_date)

    async def _sync_social_sentiment_from_daily(self, ticker: str) -> int:
        """Read daily_sentiment for this ticker and back-write to indicators.social_sentiment."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT date, avg_compound FROM daily_sentiment WHERE ticker = $1",
                ticker,
            )
        daily_scores = {str(r["date"]): r["avg_compound"] or 0.0 for r in rows}
        return await self._write_social_sentiment(ticker, daily_scores)

    async def _compute_snapshot_from_db(self, ticker: str):
        """Compute and upsert sentiment_snapshots from all scored articles for this ticker."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sentiment_label, positive_score, negative_score,
                       neutral_score, compound_score
                FROM sentiment_articles
                WHERE ticker = $1 AND sentiment_label IS NOT NULL
                """,
                ticker,
            )

        if not rows:
            return

        articles = [dict(r) for r in rows]
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

    async def _write_daily_sentiment(
        self, ticker: str, by_date: Dict[str, List[dict]]
    ) -> int:
        """Upsert aggregated daily sentiment rows into daily_sentiment table."""
        rows_written = 0
        async with self.pool.acquire() as conn:
            for day_str, arts in by_date.items():
                if not arts:
                    continue
                try:
                    parsed_date = date_type.fromisoformat(day_str[:10])
                except ValueError:
                    continue

                n = len(arts)
                avg_pos = sum(float(a.get("positive_score", 0)) for a in arts) / n
                avg_neg = sum(float(a.get("negative_score", 0)) for a in arts) / n
                avg_neu = sum(float(a.get("neutral_score",  1)) for a in arts) / n
                avg_cmp = sum(float(a.get("compound_score", 0)) for a in arts) / n

                await conn.execute(
                    """
                    INSERT INTO daily_sentiment
                        (ticker, date, avg_positive, avg_negative, avg_neutral,
                         avg_compound, article_count, computed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        avg_positive   = EXCLUDED.avg_positive,
                        avg_negative   = EXCLUDED.avg_negative,
                        avg_neutral    = EXCLUDED.avg_neutral,
                        avg_compound   = EXCLUDED.avg_compound,
                        article_count  = daily_sentiment.article_count + EXCLUDED.article_count,
                        computed_at    = NOW()
                    """,
                    ticker,
                    parsed_date,
                    round(avg_pos, 6),
                    round(avg_neg, 6),
                    round(avg_neu, 6),
                    round(avg_cmp, 6),
                    n,
                )
                rows_written += 1

        return rows_written

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
                try:
                    parsed_date = date_type.fromisoformat(str(date_str)[:10])
                except ValueError:
                    continue
                result = await conn.execute(
                    """
                    UPDATE indicators
                       SET social_sentiment = $1
                     WHERE ticker = $2
                       AND DATE(timestamp) = $3
                    """,
                    round(float(score), 6),
                    ticker,
                    parsed_date,
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
