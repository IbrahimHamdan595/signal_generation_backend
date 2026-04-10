"""
Job Service
===========
Tracks background task status (training, walk-forward, etc.) in the jobs table.
"""

import asyncpg
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create(self, job_type: str) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO jobs (job_type, status, started_at)
                VALUES ($1, 'running', NOW())
                RETURNING id
            """, job_type)
        return row["id"]

    async def update_progress(self, job_id: int, progress: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE jobs SET progress = $1::jsonb WHERE id = $2
            """, str(progress).replace("'", '"'), job_id)

    async def complete(self, job_id: int, progress: Optional[dict] = None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE jobs
                SET status = 'completed', finished_at = NOW(),
                    progress = COALESCE($1::jsonb, progress)
                WHERE id = $2
            """, str(progress).replace("'", '"') if progress else None, job_id)

    async def fail(self, job_id: int, error: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE jobs
                SET status = 'failed', finished_at = NOW(), error = $1
                WHERE id = $2
            """, error, job_id)

    async def get(self, job_id: int) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
        return dict(row) if row else None

    async def get_latest(self, job_type: str) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM jobs
                WHERE job_type = $1
                ORDER BY created_at DESC LIMIT 1
            """, job_type)
        return dict(row) if row else None
