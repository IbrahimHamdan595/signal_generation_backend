"""Contract tests for jobs routes."""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient


async def _get_token(client: AsyncClient) -> str:
    await client.post(
        "/api/v1/auth/register",
        json={"full_name": "Jobs User", "email": "jobs@example.com", "password": "pass1234"},
    )
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": "jobs@example.com", "password": "pass1234"},
    )
    return resp.json()["access_token"]


_SAMPLE_JOB = {
    "id": 42,
    "job_type": "train",
    "status": "running",
    "progress": {},
    "error": None,
    "started_at": "2025-01-01T10:00:00",
    "finished_at": None,
    "created_at": "2025-01-01T09:59:00",
}


@pytest.mark.asyncio
async def test_get_job(client: AsyncClient):
    token = await _get_token(client)

    with patch("app.api.routes.jobs.JobService") as MockSvc:
        instance = MockSvc.return_value
        instance.get = AsyncMock(return_value=_SAMPLE_JOB)

        resp = await client.get(
            "/api/v1/jobs/42",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 42
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_get_job_not_found(client: AsyncClient):
    token = await _get_token(client)

    with patch("app.api.routes.jobs.JobService") as MockSvc:
        instance = MockSvc.return_value
        instance.get = AsyncMock(return_value=None)

        resp = await client.get(
            "/api/v1/jobs/9999",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_latest_job(client: AsyncClient):
    token = await _get_token(client)

    completed_job = {**_SAMPLE_JOB, "status": "completed", "finished_at": "2025-01-01T11:00:00"}

    with patch("app.api.routes.jobs.JobService") as MockSvc:
        instance = MockSvc.return_value
        instance.get_latest = AsyncMock(return_value=completed_job)

        resp = await client.get(
            "/api/v1/jobs/latest/train",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["job_type"] == "train"


@pytest.mark.asyncio
async def test_get_latest_job_not_found(client: AsyncClient):
    token = await _get_token(client)

    with patch("app.api.routes.jobs.JobService") as MockSvc:
        instance = MockSvc.return_value
        instance.get_latest = AsyncMock(return_value=None)

        resp = await client.get(
            "/api/v1/jobs/latest/nonexistent_type",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 404
