"""
Alert Service
=============
Writes an alert row whenever a high-confidence BUY/SELL signal is generated.
The frontend polls GET /alerts to show the bell-icon notification count.
"""

import asyncpg
import logging
from typing import Optional

logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_THRESHOLD = 0.70


class AlertService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def maybe_create(
        self,
        ticker: str,
        action: str,
        confidence: float,
        signal_id: Optional[int] = None,
    ) -> bool:
        """Create an alert if the signal meets the confidence threshold."""
        if action == "HOLD" or confidence < HIGH_CONFIDENCE_THRESHOLD:
            return False
        msg = (
            f"{action} signal for {ticker} — "
            f"{round(confidence * 100)}% confidence"
        )
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO alerts (ticker, action, confidence, signal_id, message)
                VALUES ($1, $2, $3, $4, $5)
            """, ticker.upper(), action, confidence, signal_id, msg)
        logger.info(f"🔔 Alert created: {msg}")
        return True

    async def get_alerts(self, unread_only: bool = False, limit: int = 50) -> list[dict]:
        async with self.pool.acquire() as conn:
            if unread_only:
                rows = await conn.fetch("""
                    SELECT * FROM alerts
                    WHERE is_read = FALSE
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM alerts
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)
        return [dict(r) for r in rows]

    async def mark_read(self, alert_id: int) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE alerts SET is_read = TRUE WHERE id = $1
            """, alert_id)
        return result == "UPDATE 1"

    async def mark_all_read(self) -> int:
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE alerts SET is_read = TRUE WHERE is_read = FALSE
            """)
        count = int(result.split()[-1])
        return count

    async def unread_count(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM alerts WHERE is_read = FALSE"
            )
        return row["n"] if row else 0
