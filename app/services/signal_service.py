import asyncio
import asyncpg
from datetime import datetime, timezone
from typing import List, Optional
import logging

from app.services.ml_service import MLService
from app.services.alert_service import AlertService
from app.ml.models.registry import is_model_trained, get_model

logger = logging.getLogger(__name__)


class SignalService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.ml_svc = MLService(pool)

    async def generate_and_store(
        self, ticker: str, interval: str = "1d"
    ) -> Optional[dict]:
        ticker = ticker.upper()

        if not is_model_trained():
            logger.warning("⚠️  Signal requested but model not trained yet.")
            return None

        result = await self.ml_svc.predict_ticker(ticker, interval)

        if "error" in result:
            logger.error(f"Signal error for {ticker}: {result['error']}")
            return None

        entry_time = None
        if result.get("entry_time"):
            try:
                entry_time = datetime.fromisoformat(result["entry_time"])
            except Exception:
                pass

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals (
                    ticker, interval, action, confidence,
                    entry_price, stop_loss, take_profit, net_profit,
                    bars_to_entry, entry_time, entry_time_label,
                    prob_buy, prob_sell, prob_hold, source, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                RETURNING *
                """,
                ticker,
                interval,
                result["action"],
                result["confidence"],
                result.get("entry_price"),
                result.get("stop_loss"),
                result.get("take_profit"),
                result.get("net_profit"),
                result.get("bars_to_entry"),
                entry_time,
                result.get("entry_time_label"),
                result.get("probabilities", {}).get("buy"),
                result.get("probabilities", {}).get("sell"),
                result.get("probabilities", {}).get("hold"),
                "ml_model",
                datetime.now(timezone.utc),
            )

            doc = dict(row)
            logger.info(
                f"✅ Signal stored: {ticker} → {result['action']} @ {result.get('entry_time')}"
            )

        # Fire alert for high-confidence directional signals
        try:
            alert_svc = AlertService(self.pool)
            await alert_svc.maybe_create(
                ticker=ticker,
                action=result["action"],
                confidence=result["confidence"],
                signal_id=doc["id"],
            )
        except Exception as e:
            logger.warning(f"⚠️  Alert creation failed for {ticker}: {e}")

        return doc

    async def generate_signal(self, ticker: str, interval: str = "1d") -> dict:
        ticker = ticker.upper()
        model = get_model()

        if model is None:
            return {
                "ticker": ticker,
                "interval": interval,
                "action": "HOLD",
                "confidence": 0.0,
                "source": "no_model",
            }

        result = await self.ml_svc.predict_ticker(ticker, interval)

        if "error" in result:
            return {
                "ticker": ticker,
                "interval": interval,
                "action": "HOLD",
                "confidence": 0.0,
                "source": "error",
                "error": result["error"],
            }

        return {
            "ticker": ticker,
            "interval": interval,
            "action": result.get("action", "HOLD"),
            "confidence": result.get("confidence", 0.0),
            "source": "ml_model",
            **{
                k: result[k]
                for k in (
                    "entry_price",
                    "stop_loss",
                    "take_profit",
                    "net_profit",
                    "bars_to_entry",
                    "entry_time",
                )
                if k in result
            },
        }

    async def generate_batch(
        self, tickers: List[str], interval: str = "1d"
    ) -> List[dict]:
        tasks = [self.generate_and_store(ticker, interval) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def generate_batch_with_skip_report(
        self, tickers: List[str], interval: str = "1d"
    ) -> dict:
        tasks = [self.generate_and_store(ticker, interval) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        generated = []
        skipped = []

        for ticker, result in zip(tickers, results):
            if result is None or isinstance(result, Exception):
                skipped.append(ticker)
            else:
                generated.append(result)

        return {
            "generated": len(generated),
            "skipped": len(skipped),
            "skipped_tickers": skipped,
            "signals": generated,
        }

    async def get_latest(self, ticker: str, interval: str = "1d") -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM signals
                WHERE ticker = $1 AND interval = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                ticker.upper(),
                interval,
            )
            return dict(row) if row else None

    async def get_history(
        self, ticker: str, interval: str = "1d", limit: int = 50
    ) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM signals
                WHERE ticker = $1 AND interval = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                ticker.upper(),
                interval,
                limit,
            )
            return [dict(r) for r in rows]

    async def get_all_latest(self) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (ticker) * FROM signals
                ORDER BY ticker, created_at DESC
                """
            )
            return [dict(r) for r in rows]

    async def get_by_action(
        self, action: str, interval: str = "1d", limit: int = 50
    ) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM signals
                WHERE action = $1 AND interval = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                action.upper(),
                interval,
                limit,
            )
            return [dict(r) for r in rows]

    async def get_high_confidence(
        self, min_confidence: float = 0.75, interval: str = "1d", limit: int = 20
    ) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM signals
                WHERE confidence >= $1 AND interval = $2
                AND action IN ('BUY', 'SELL')
                ORDER BY created_at DESC
                LIMIT $3
                """,
                min_confidence,
                interval,
                limit,
            )
            return [dict(r) for r in rows]
