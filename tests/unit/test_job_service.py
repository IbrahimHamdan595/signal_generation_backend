"""Unit tests for JobService."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from app.services.job_service import JobService


def _make_pool(fetchrow_return=None, execute_return=None):
    conn = MagicMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.execute = AsyncMock(return_value=execute_return)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=conn)
    return pool


_SAMPLE_ROW = {
    "id": 1,
    "job_type": "train",
    "status": "pending",
    "progress": {},
    "error": None,
    "started_at": None,
    "finished_at": None,
    "created_at": datetime.now(timezone.utc),
}


class TestJobService:
    @pytest.mark.asyncio
    async def test_create_returns_job_id(self):
        pool = _make_pool(fetchrow_return=_SAMPLE_ROW)
        svc = JobService(pool)
        job_id = await svc.create("train")
        assert job_id == 1

    @pytest.mark.asyncio
    async def test_get_returns_none_when_missing(self):
        pool = _make_pool(fetchrow_return=None)
        svc = JobService(pool)
        result = await svc.get(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_dict_when_found(self):
        pool = _make_pool(fetchrow_return=_SAMPLE_ROW)
        svc = JobService(pool)
        result = await svc.get(1)
        assert result is not None
        assert result["job_type"] == "train"
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_complete_updates_status(self):
        pool = _make_pool()
        svc = JobService(pool)
        # Should not raise
        await svc.complete(1)
        pool.acquire.return_value.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_fail_updates_status_with_error(self):
        pool = _make_pool()
        svc = JobService(pool)
        await svc.fail(1, "Something went wrong")
        pool.acquire.return_value.execute.assert_called_once()
