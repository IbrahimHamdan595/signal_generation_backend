import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings
from app.db.database import get_db


_users: dict = {}


class MockRecord(dict):
    """Mock asyncpg Record that can be converted to dict"""

    def __init__(self, data: dict):
        super().__init__(data)


class MockConnection:
    def __init__(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def fetchrow(self, query, *args):
        # Handle INSERT RETURNING
        if "insert into users" in query.lower() and "returning" in query.lower():
            user_dict = {
                "id": 1,
                "full_name": args[0] if len(args) > 0 else None,
                "email": args[1] if len(args) > 1 else None,
                "password_hash": args[2] if len(args) > 2 else None,
                "is_active": args[3] if len(args) > 3 else True,
                "is_admin": args[4] if len(args) > 4 else False,
                "watchlist": args[5] if len(args) > 5 else [],
                "created_at": args[6] if len(args) > 6 else None,
                "updated_at": args[7] if len(args) > 7 else None,
            }
            # Store for later retrieval
            if args[1]:  # email
                _users[args[1]] = user_dict
            return MockRecord(user_dict)

        # Handle SELECT by email
        email = args[0] if args else None
        if email and "select" in query.lower() and "from users" in query.lower():
            user_data = _users.get(email)
            return MockRecord(user_data) if user_data else None
        return None

    async def fetch(self, query, *args):
        return []

    async def execute(self, query, *args):
        if "insert into users" in query.lower():
            return 1
        return None


class MockAcquire:
    """Wrapper to make pool.acquire() return an async context manager"""

    def __init__(self):
        self.conn = MockConnection()

    async def __aenter__(self):
        return await self.conn.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.conn.__aexit__(exc_type, exc_val, exc_tb)


async def mock_acquire():
    return MockAcquire()


def _make_mock_pool():
    pool = MagicMock()
    pool.acquire = mock_acquire
    return pool


_mock_pool = _make_mock_pool()


async def override_get_db():
    return _mock_pool


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    app.dependency_overrides[get_db] = override_get_db

    with (
        patch("app.db.database.connect_db", new=AsyncMock()),
        patch("app.db.database.close_db", new=AsyncMock()),
        patch("app.db.database.pool", _mock_pool),
        patch("app.ml.models.registry.load_model", return_value=None),
        patch("app.services.scheduler.start_scheduler", return_value=None),
        patch("app.services.scheduler.stop_scheduler", return_value=None),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client: AsyncClient):
    return {}


@pytest.fixture(scope="module")
async def db_client():
    pass
