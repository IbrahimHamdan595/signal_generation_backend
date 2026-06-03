import logging
import asyncpg
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

pool: Optional[asyncpg.Pool] = None


async def connect_db():
    global pool
    pool = await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=2,
        max_size=20,
        statement_cache_size=0,
        # Recycle idle connections after 5 min so stale sockets from network
        # blips are replaced before the next query tries to use them.
        max_inactive_connection_lifetime=300,
        # Hard timeout per query — surfaces slow queries instead of hanging
        # the connection. Browser cancels around 30s anyway.
        command_timeout=25,
    )
    logger.info("✅ Connected to PostgreSQL")


async def close_db():
    global pool
    if pool:
        await pool.close()
        logger.info("🔌 PostgreSQL connection closed")


async def get_db():
    return pool
