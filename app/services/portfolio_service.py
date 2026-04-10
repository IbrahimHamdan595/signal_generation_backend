"""
Portfolio Service
=================
Paper-trading portfolio tracker.
Users can open/close positions from signals and track unrealized P&L.
"""

import asyncpg
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PortfolioService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_positions(self, user_id: int, open_only: bool = True) -> list[dict]:
        async with self.pool.acquire() as conn:
            query = """
                SELECT p.*,
                    (SELECT close FROM ohlcv_data
                     WHERE ticker = p.ticker AND interval = '1d'
                     ORDER BY timestamp DESC LIMIT 1) AS current_price
                FROM positions p
                WHERE p.user_id = $1
            """
            if open_only:
                query += " AND p.is_open = TRUE"
            query += " ORDER BY p.opened_at DESC"
            rows = await conn.fetch(query, user_id)
        result = []
        for r in rows:
            d = dict(r)
            cp = d.get("current_price")
            if cp and d["avg_cost"] and d["quantity"]:
                d["unrealized_pnl"] = round(
                    (cp - d["avg_cost"]) * d["quantity"], 2
                )
                d["unrealized_pct"] = round(
                    (cp - d["avg_cost"]) / d["avg_cost"] * 100, 2
                ) if d["avg_cost"] else 0
            else:
                d["unrealized_pnl"] = None
                d["unrealized_pct"] = None
            result.append(d)
        return result

    async def get_summary(self, user_id: int) -> dict:
        positions = await self.get_positions(user_id, open_only=True)
        total_cost = sum(p["avg_cost"] * p["quantity"] for p in positions)
        total_value = sum(
            (p.get("current_price") or p["avg_cost"]) * p["quantity"]
            for p in positions
        )
        total_unrealized = total_value - total_cost
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(SUM(realized_pnl), 0) AS realized FROM positions WHERE user_id = $1",
                user_id,
            )
        realized = float(row["realized"]) if row else 0.0
        return {
            "open_positions": len(positions),
            "total_cost": round(total_cost, 2),
            "total_value": round(total_value, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "realized_pnl": round(realized, 2),
            "total_pnl": round(total_unrealized + realized, 2),
        }

    async def open_position(
        self,
        user_id: int,
        ticker: str,
        quantity: float,
        price: float,
        signal_id: Optional[int] = None,
    ) -> dict:
        ticker = ticker.upper()
        async with self.pool.acquire() as conn:
            # Check if an open position already exists for this ticker
            existing = await conn.fetchrow(
                "SELECT * FROM positions WHERE user_id = $1 AND ticker = $2 AND is_open = TRUE",
                user_id, ticker,
            )
            if existing:
                # Average down: update qty and avg cost
                old_qty = existing["quantity"]
                old_cost = existing["avg_cost"]
                new_qty = old_qty + quantity
                new_avg = (old_qty * old_cost + quantity * price) / new_qty
                row = await conn.fetchrow("""
                    UPDATE positions
                    SET quantity = $1, avg_cost = $2
                    WHERE id = $3 RETURNING *
                """, new_qty, new_avg, existing["id"])
                pos_id = existing["id"]
            else:
                row = await conn.fetchrow("""
                    INSERT INTO positions (user_id, ticker, quantity, avg_cost)
                    VALUES ($1, $2, $3, $4) RETURNING *
                """, user_id, ticker, quantity, price)
                pos_id = row["id"]

            await conn.execute("""
                INSERT INTO position_trades (position_id, signal_id, action, quantity, price)
                VALUES ($1, $2, 'BUY', $3, $4)
            """, pos_id, signal_id, quantity, price)

        return dict(row)

    async def close_position(
        self,
        user_id: int,
        position_id: int,
        price: float,
    ) -> dict:
        async with self.pool.acquire() as conn:
            pos = await conn.fetchrow(
                "SELECT * FROM positions WHERE id = $1 AND user_id = $2 AND is_open = TRUE",
                position_id, user_id,
            )
            if not pos:
                raise ValueError("Position not found or already closed")

            pnl = round((price - pos["avg_cost"]) * pos["quantity"], 2)
            row = await conn.fetchrow("""
                UPDATE positions
                SET is_open = FALSE, closed_at = NOW(), realized_pnl = realized_pnl + $1
                WHERE id = $2 RETURNING *
            """, pnl, position_id)

            await conn.execute("""
                INSERT INTO position_trades (position_id, action, quantity, price, pnl)
                VALUES ($1, 'SELL', $2, $3, $4)
            """, position_id, pos["quantity"], price, pnl)

        d = dict(row)
        d["realized_pnl_this_close"] = pnl
        return d
