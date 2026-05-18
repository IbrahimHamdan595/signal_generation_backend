"""Unit tests for the Donchian rule-based strategy.

DonchianService is the rule-based FX pipeline — fires BUY when today closes
above the 20-day high (excluding today AND yesterday for whipsaw protection),
SELL when below the 20-day low, HOLD otherwise.

We stub the asyncpg pool so these tests don't need a database — `fetch`
returns synthetic bar series and the INSERT path is a no-op that echoes
back the input row.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest

from app.strategies.donchian import DonchianService, DONCHIAN_WINDOW, HISTORY_NEEDED


# ── Stub pool ────────────────────────────────────────────────────────────────

class _StubConn:
    """Minimal asyncpg connection stub.

    Responses are programmed by the test via `bars`, `atr`, and
    `has_open_position`. INSERT/UPDATE queries echo back a dict that looks
    like the table's RETURNING row.
    """
    def __init__(self, bars: Sequence[float], atr: float | None = 0.005, has_open_position: bool = False):
        self.bars = bars
        self.atr  = atr
        self.has_open_position = has_open_position
        self.inserted_rows: list[dict] = []

    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass

    async def fetch(self, query: str, *args: Any):
        q = query.lower()
        if "from ohlcv_data" in q:
            # Latest-bar-first ordering per the DonchianService SELECT
            now = datetime(2026, 5, 18, tzinfo=timezone.utc)
            return [
                {"timestamp": now - timedelta(days=i), "close": c}
                for i, c in enumerate(reversed(self.bars))
            ]
        return []

    async def fetchrow(self, query: str, *args: Any):
        if "from indicators" in query.lower():
            return {"atr_14": self.atr} if self.atr is not None else None
        return None

    async def fetchval(self, query: str, *args: Any):
        if "from trade_executions" in query.lower():
            return 1 if self.has_open_position else None
        return None

    async def execute(self, query: str, *args: Any):
        return None


class _StubPool:
    def __init__(self, conn: _StubConn):
        self._conn = conn

    def acquire(self):
        return self._conn

    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass


def _make_pool(bars, **kwargs):
    return _StubPool(_StubConn(bars, **kwargs))


# Patch the INSERT path so it doesn't need a real schema. We replace
# DonchianService._persist with a stub that just returns the result dict
# (the actual DB write is exercised by integration tests, not here).
@pytest.fixture
def patched_donchian(monkeypatch):
    async def _fake_persist(self, ticker, interval, result, source):
        return {**result, "ticker": ticker, "interval": interval, "source": source}
    monkeypatch.setattr(DonchianService, "_persist", _fake_persist)
    async def _fake_persist_hold(self, ticker, interval, current_close=None, reject_reasons=None):
        return {
            "action": "HOLD", "ticker": ticker, "interval": interval,
            "source": "rule_donchian",
            "current_close": current_close,
            "reject_reasons": reject_reasons or [],
        }
    monkeypatch.setattr(DonchianService, "_persist_hold", _fake_persist_hold)


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_breakout_above_channel_returns_buy(patched_donchian):
    # 22 bars: first 20 are flat at 1.10, then 1.10 (yesterday inside channel),
    # then 1.20 today (a clear breakout above the prior-20-day max). The
    # `closes[-22:-2]` slice excludes today and yesterday, so the channel
    # max stays at 1.10 even though yesterday's close was also 1.10.
    bars = [1.10] * 20 + [1.10, 1.20]
    pool = _make_pool(bars)
    svc  = DonchianService(pool)
    sig  = await svc.generate_for_ticker("EURUSD")
    assert sig["action"] == "BUY"
    assert sig["channel_upper"] == pytest.approx(1.10)


@pytest.mark.asyncio
async def test_breakdown_below_channel_returns_sell(patched_donchian):
    bars = [1.10] * 20 + [1.10, 1.00]
    pool = _make_pool(bars)
    svc  = DonchianService(pool)
    sig  = await svc.generate_for_ticker("EURUSD")
    assert sig["action"] == "SELL"


@pytest.mark.asyncio
async def test_close_inside_channel_returns_hold(patched_donchian):
    # Today's close at 1.105 is between the channel min and max
    bars = [1.10, 1.12, 1.09, 1.11, 1.10, 1.13, 1.08, 1.10, 1.11, 1.10,
            1.12, 1.09, 1.11, 1.10, 1.13, 1.08, 1.10, 1.11, 1.10, 1.12,
            1.11, 1.105]
    pool = _make_pool(bars)
    svc  = DonchianService(pool)
    sig  = await svc.generate_for_ticker("EURUSD")
    assert sig["action"] == "HOLD"


@pytest.mark.asyncio
async def test_insufficient_history_returns_hold_with_reason(patched_donchian):
    # Fewer than HISTORY_NEEDED (=22) bars → can't apply the rule
    bars = [1.10] * (HISTORY_NEEDED - 5)
    pool = _make_pool(bars)
    svc  = DonchianService(pool)
    sig  = await svc.generate_for_ticker("EURUSD")
    assert sig["action"] == "HOLD"
    assert any("insufficient_history" in r for r in sig.get("reject_reasons", []))


@pytest.mark.asyncio
async def test_open_position_blocks_new_signal(patched_donchian):
    # Even though the rule would have fired (clear breakout), an existing
    # open trade prevents stacking another position from this strategy.
    bars = [1.10] * 20 + [1.10, 1.20]
    pool = _make_pool(bars, has_open_position=True)
    svc  = DonchianService(pool)
    sig  = await svc.generate_for_ticker("EURUSD")
    assert sig["action"] == "HOLD"
    assert any("open_position_exists" in r for r in sig.get("reject_reasons", []))


@pytest.mark.asyncio
async def test_whipsaw_protection_excludes_yesterday(patched_donchian):
    # Yesterday set the high at 1.30. Today's close at 1.25 is BELOW
    # yesterday's high but still ABOVE the prior-20-day max of 1.10.
    # Because the channel slice is `[-22:-2]` (excludes today AND yesterday),
    # the rule correctly compares against 1.10 → BUY fires.
    bars = [1.10] * 20 + [1.30, 1.25]
    pool = _make_pool(bars)
    svc  = DonchianService(pool)
    sig  = await svc.generate_for_ticker("EURUSD")
    assert sig["action"] == "BUY"


@pytest.mark.asyncio
async def test_atr_levels_used_when_atr_available(patched_donchian):
    # Channel min/max ignored — focus on whether the SL/TP get the
    # right ATR-derived distance from entry.
    bars = [1.10] * 20 + [1.10, 1.20]
    pool = _make_pool(bars, atr=0.005)   # 0.5% ATR on EURUSD-ish
    svc  = DonchianService(pool)
    sig  = await svc.generate_for_ticker("EURUSD")
    # ATR levels helper uses k_sl=1.5, k_tp=3.0 by default
    expected_sl_dist = 1.5 * 0.005
    expected_tp_dist = 3.0 * 0.005
    assert sig["stop_loss"]   == pytest.approx(1.20 - expected_sl_dist, abs=1e-4)
    assert sig["take_profit"] == pytest.approx(1.20 + expected_tp_dist, abs=1e-4)


@pytest.mark.asyncio
async def test_window_constant_matches_canonical_value():
    # Guard against drift: the canonical Turtle setup uses a 20-day window.
    assert DONCHIAN_WINDOW == 20
    assert HISTORY_NEEDED == DONCHIAN_WINDOW + 2
