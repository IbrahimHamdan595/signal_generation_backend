"""
Signal Outcome Tracker
======================
After a BUY/SELL signal is generated, this service checks—N bars later—
whether price hit the take_profit (WIN) or stop_loss (LOSS), and records
the actual return.  HOLD signals are skipped (no directional bet).

Scheduler calls `check_pending_outcomes()` every hour.
"""

import asyncpg
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# How many bars we wait before resolving a signal as EXPIRED
MAX_BARS_HELD = 10


class OutcomeService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    # ── Public API ────────────────────────────────────────────────────────────

    async def check_pending_outcomes(self) -> dict:
        """
        Fetch all unresolved BUY/SELL signals that have enough subsequent
        OHLCV bars, resolve each one, and store the result.
        Returns counts of resolved / skipped.
        """
        resolved = skipped = 0
        async with self.pool.acquire() as conn:
            # Signals with no outcome yet, not HOLD, with entry_price/sl/tp set
            rows = await conn.fetch("""
                SELECT s.id, s.ticker, s.interval, s.action,
                       s.entry_price, s.stop_loss, s.take_profit, s.created_at
                FROM   signals s
                LEFT JOIN signal_outcomes o ON o.signal_id = s.id
                WHERE  o.id IS NULL
                  AND  s.action IN ('BUY', 'SELL')
                  AND  s.entry_price IS NOT NULL
                  AND  s.stop_loss   IS NOT NULL
                  AND  s.take_profit IS NOT NULL
                ORDER BY s.created_at ASC
                LIMIT  500
            """)

        for row in rows:
            ok = await self._resolve(row)
            if ok:
                resolved += 1
            else:
                skipped += 1

        logger.info(f"✅ Outcomes: {resolved} resolved, {skipped} pending/skipped")
        return {"resolved": resolved, "skipped": skipped}

    async def get_outcomes(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        async with self.pool.acquire() as conn:
            if ticker:
                rows = await conn.fetch("""
                    SELECT o.*, s.action as signal_action, s.created_at as signal_at,
                           s.confidence
                    FROM signal_outcomes o
                    JOIN signals s ON s.id = o.signal_id
                    WHERE o.ticker = $1
                    ORDER BY o.checked_at DESC
                    LIMIT $2
                """, ticker.upper(), limit)
            else:
                rows = await conn.fetch("""
                    SELECT o.*, s.action as signal_action, s.created_at as signal_at,
                           s.confidence
                    FROM signal_outcomes o
                    JOIN signals s ON s.id = o.signal_id
                    ORDER BY o.checked_at DESC
                    LIMIT $1
                """, limit)
        return [dict(r) for r in rows]

    async def get_accuracy_summary(self) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*)                                          AS total,
                    COUNT(*) FILTER (WHERE outcome = 'WIN')          AS wins,
                    COUNT(*) FILTER (WHERE outcome = 'LOSS')         AS losses,
                    COUNT(*) FILTER (WHERE outcome = 'EXPIRED')      AS expired,
                    AVG(actual_return)                               AS avg_return,
                    AVG(actual_return) FILTER (WHERE outcome='WIN')  AS avg_win,
                    AVG(actual_return) FILTER (WHERE outcome='LOSS') AS avg_loss
                FROM signal_outcomes
            """)
        if not row or row["total"] == 0:
            return {"total": 0, "win_rate": 0.0, "avg_return": 0.0}

        total = row["total"]
        wins  = row["wins"] or 0
        return {
            "total":      total,
            "wins":       wins,
            "losses":     row["losses"] or 0,
            "expired":    row["expired"] or 0,
            "win_rate":   wins / total if total else 0.0,
            "avg_return": float(row["avg_return"] or 0.0),
            "avg_win":    float(row["avg_win"]    or 0.0),
            "avg_loss":   float(row["avg_loss"]   or 0.0),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _resolve(self, signal: asyncpg.Record) -> bool:
        """
        Walk the OHLCV bars after the signal timestamp.
        First bar to breach TP or SL determines the outcome.
        If neither is breached within MAX_BARS_HELD bars, mark EXPIRED.
        Returns True if an outcome was written.
        """
        async with self.pool.acquire() as conn:
            bars = await conn.fetch("""
                SELECT high, low, close, timestamp
                FROM   ohlcv_data
                WHERE  ticker   = $1
                  AND  interval = $2
                  AND  timestamp > $3
                ORDER  BY timestamp ASC
                LIMIT  $4
            """, signal["ticker"], signal["interval"],
                signal["created_at"], MAX_BARS_HELD)

        if not bars:
            return False  # no subsequent bars yet

        action     = signal["action"]
        entry      = float(signal["entry_price"])
        stop_loss  = float(signal["stop_loss"])
        take_profit = float(signal["take_profit"])

        outcome     = "EXPIRED"
        exit_price  = float(bars[-1]["close"])
        exit_time   = bars[-1]["timestamp"]
        bars_held   = len(bars)

        for i, bar in enumerate(bars):
            hi = float(bar["high"])
            lo = float(bar["low"])

            if action == "BUY":
                if hi >= take_profit:
                    outcome    = "WIN"
                    exit_price = take_profit
                    exit_time  = bar["timestamp"]
                    bars_held  = i + 1
                    break
                if lo <= stop_loss:
                    outcome    = "LOSS"
                    exit_price = stop_loss
                    exit_time  = bar["timestamp"]
                    bars_held  = i + 1
                    break
            else:  # SELL
                if lo <= take_profit:
                    outcome    = "WIN"
                    exit_price = take_profit
                    exit_time  = bar["timestamp"]
                    bars_held  = i + 1
                    break
                if hi >= stop_loss:
                    outcome    = "LOSS"
                    exit_price = stop_loss
                    exit_time  = bar["timestamp"]
                    bars_held  = i + 1
                    break

        if action == "BUY":
            actual_return = (exit_price - entry) / entry
        else:
            actual_return = (entry - exit_price) / entry

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO signal_outcomes
                    (signal_id, ticker, action, entry_price, stop_loss, take_profit,
                     outcome, actual_return, bars_held, exit_price, exit_time, checked_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (signal_id) DO UPDATE SET
                    outcome       = EXCLUDED.outcome,
                    actual_return = EXCLUDED.actual_return,
                    bars_held     = EXCLUDED.bars_held,
                    exit_price    = EXCLUDED.exit_price,
                    exit_time     = EXCLUDED.exit_time,
                    checked_at    = EXCLUDED.checked_at
            """,
                signal["id"], signal["ticker"], action,
                entry, stop_loss, take_profit,
                outcome, actual_return, bars_held, exit_price,
                exit_time, datetime.now(timezone.utc),
            )
        return True
