import asyncpg
from datetime import datetime, timezone
from typing import Optional

from app.core.security import hash_password
from app.schemas.schemas import RegisterRequest


class UserService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create_user(self, data: RegisterRequest) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (full_name, email, password_hash, is_active, is_admin, watchlist, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
                """,
                data.full_name,
                data.email,
                hash_password(data.password),
                True,
                False,
                [],
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            )
            return dict(row)

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email = $1",
                email,
            )
            return dict(row) if row else None

    async def get_user_by_id(self, user_id: int) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1",
                user_id,
            )
            return dict(row) if row else None

    async def update_watchlist(self, user_id: int, tickers: list) -> dict:
        tickers = [t.upper() for t in tickers]
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET watchlist = $1, updated_at = $2 WHERE id = $3",
                tickers,
                datetime.now(timezone.utc),
                user_id,
            )
        return await self.get_user_by_id(user_id)

    async def add_to_watchlist(self, user_id: int, ticker: str) -> dict:
        ticker = ticker.upper()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT watchlist FROM users WHERE id = $1",
                user_id,
            )
            if row:
                watchlist = list(row["watchlist"]) if row["watchlist"] else []
                if ticker not in watchlist:
                    watchlist.append(ticker)
                await conn.execute(
                    "UPDATE users SET watchlist = $1, updated_at = $2 WHERE id = $3",
                    watchlist,
                    datetime.now(timezone.utc),
                    user_id,
                )
        return await self.get_user_by_id(user_id)

    async def remove_from_watchlist(self, user_id: int, ticker: str) -> dict:
        ticker = ticker.upper()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT watchlist FROM users WHERE id = $1",
                user_id,
            )
            if row:
                watchlist = list(row["watchlist"]) if row["watchlist"] else []
                if ticker in watchlist:
                    watchlist.remove(ticker)
                await conn.execute(
                    "UPDATE users SET watchlist = $1, updated_at = $2 WHERE id = $3",
                    watchlist,
                    datetime.now(timezone.utc),
                    user_id,
                )
        return await self.get_user_by_id(user_id)
