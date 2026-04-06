import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.services.signal_service import SignalService


class TestSignalService:
    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()

        async def _acquire():
            conn = MagicMock()
            conn.fetchrow = AsyncMock(
                return_value={
                    "id": 1,
                    "ticker": "AAPL",
                    "interval": "1d",
                    "close": 150.0,
                    "rsi_14": 55.0,
                    "timestamp": datetime.now(timezone.utc),
                }
            )
            return conn

        pool.acquire = AsyncMock(side_effect=_acquire)
        return pool

    @pytest.fixture
    def signal_service(self, mock_pool):
        return SignalService(mock_pool)

    @pytest.mark.asyncio
    async def test_generate_signal_returns_valid_structure(
        self, signal_service, mock_pool
    ):
        with patch("app.services.signal_service.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.return_value = (
                MagicMock(argmax=MagicMock(return_value=1)),
                MagicMock(),
            )
            mock_get_model.return_value = mock_model

            result = await signal_service.generate_signal("AAPL", "1d")

        assert "ticker" in result
        assert "action" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_generate_signal_no_model(self, signal_service, mock_pool):
        with patch("app.services.signal_service.get_model", return_value=None):
            result = await signal_service.generate_signal("AAPL", "1d")

        assert result["action"] == "HOLD"
        assert result["source"] == "no_model"
