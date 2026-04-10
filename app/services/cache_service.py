"""
Cache Service
=============
In-memory TTL cache with optional Redis backend.
Falls back to in-memory if REDIS_URL is not configured.
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── In-memory fallback ────────────────────────────────────────────────────────

class _MemoryCache:
    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}

    async def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int = 300):
        self._store[key] = (value, time.monotonic() + ttl)

    async def delete(self, key: str):
        self._store.pop(key, None)

    async def clear_pattern(self, prefix: str):
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]


# ── Redis backend ─────────────────────────────────────────────────────────────

class _RedisCache:
    def __init__(self, redis):
        self._r = redis

    async def get(self, key: str) -> Optional[Any]:
        try:
            raw = await self._r.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        try:
            await self._r.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"Redis set error: {e}")

    async def delete(self, key: str):
        try:
            await self._r.delete(key)
        except Exception:
            pass

    async def clear_pattern(self, prefix: str):
        try:
            keys = await self._r.keys(f"{prefix}*")
            if keys:
                await self._r.delete(*keys)
        except Exception:
            pass


# ── Singleton factory ─────────────────────────────────────────────────────────

_cache_instance = None


async def get_cache():
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    try:
        from app.core.config import settings
        redis_url = getattr(settings, "REDIS_URL", None)
        if redis_url:
            import redis.asyncio as aioredis
            r = aioredis.from_url(redis_url, decode_responses=True)
            await r.ping()
            _cache_instance = _RedisCache(r)
            logger.info("✅ Using Redis cache")
            return _cache_instance
    except Exception as e:
        logger.info(f"Redis not available ({e}), using in-memory cache")

    _cache_instance = _MemoryCache()
    return _cache_instance
