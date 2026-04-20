"""Contract tests for portfolio routes."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


def _make_auth_header(client_fixture):
    """Helper – not used directly, tests create their own tokens via register+login."""
    return {}


async def _get_token(client: AsyncClient) -> str:
    await client.post(
        "/api/v1/auth/register",
        json={"full_name": "Portfolio User", "email": "portfolio@example.com", "password": "pass1234"},
    )
    resp = await client.post(
        "/api/v1/auth/login",
        data={"username": "portfolio@example.com", "password": "pass1234"},
    )
    return resp.json()["access_token"]


@pytest.mark.asyncio
async def test_get_positions_empty(client: AsyncClient):
    token = await _get_token(client)

    with patch("app.api.routes.portfolio.PortfolioService") as MockSvc:
        instance = MockSvc.return_value
        instance.get_positions = AsyncMock(return_value=[])

        resp = await client.get(
            "/api/v1/portfolio/positions",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_summary(client: AsyncClient):
    token = await _get_token(client)

    summary = {
        "open_positions": 2,
        "total_cost": 5000.0,
        "total_value": 5200.0,
        "unrealized_pnl": 200.0,
        "realized_pnl": 0.0,
        "total_pnl": 200.0,
    }

    with patch("app.api.routes.portfolio.PortfolioService") as MockSvc:
        instance = MockSvc.return_value
        instance.get_summary = AsyncMock(return_value=summary)

        resp = await client.get(
            "/api/v1/portfolio/summary",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["open_positions"] == 2
    assert data["unrealized_pnl"] == 200.0


@pytest.mark.asyncio
async def test_open_position(client: AsyncClient):
    token = await _get_token(client)

    new_pos = {
        "id": 1,
        "user_id": 1,
        "ticker": "AAPL",
        "quantity": 10.0,
        "avg_cost": 180.0,
        "opened_at": "2025-01-01T00:00:00",
        "closed_at": None,
        "realized_pnl": 0.0,
        "is_open": True,
        "current_price": None,
        "unrealized_pnl": None,
        "unrealized_pct": None,
    }

    with patch("app.api.routes.portfolio.PortfolioService") as MockSvc:
        instance = MockSvc.return_value
        instance.open_position = AsyncMock(return_value=new_pos)

        resp = await client.post(
            "/api/v1/portfolio/positions",
            json={"ticker": "AAPL", "quantity": 10, "price": 180.0},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert data["is_open"] is True


@pytest.mark.asyncio
async def test_close_position(client: AsyncClient):
    token = await _get_token(client)

    closed = {
        "id": 1,
        "user_id": 1,
        "ticker": "AAPL",
        "quantity": 10.0,
        "avg_cost": 180.0,
        "opened_at": "2025-01-01T00:00:00",
        "closed_at": "2025-06-01T00:00:00",
        "realized_pnl": 50.0,
        "realized_pnl_this_close": 50.0,
        "is_open": False,
        "current_price": None,
        "unrealized_pnl": None,
        "unrealized_pct": None,
    }

    with patch("app.api.routes.portfolio.PortfolioService") as MockSvc:
        instance = MockSvc.return_value
        instance.close_position = AsyncMock(return_value=closed)

        resp = await client.post(
            "/api/v1/portfolio/positions/1/close",
            json={"price": 185.0},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["is_open"] is False
    assert data["realized_pnl"] == 50.0
