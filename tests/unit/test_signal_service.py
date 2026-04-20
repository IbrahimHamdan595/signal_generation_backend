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
        predict_result = {
            "action": "BUY",
            "confidence": 0.85,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "net_profit": 10.0,
            "bars_to_entry": 1,
            "entry_time": None,
            "entry_time_label": None,
            "prob_buy": 0.85,
            "prob_sell": 0.10,
            "prob_hold": 0.05,
        }
        with patch("app.services.signal_service.get_model") as mock_get_model, \
             patch.object(signal_service.ml_svc, "predict_ticker", new=AsyncMock(return_value=predict_result)):
            mock_get_model.return_value = MagicMock()

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
