"""Donchian Channel breakout strategy — rule-based, no ML.

The 20-day Donchian channel is one of the most-studied systematic trading
rules in retail/quant trading (Turtle Traders, mid-1980s). The rule:

    BUY  when today's close > max(closes[t-21 : t-1])   (20-day high ex-today)
    SELL when today's close < min(closes[t-21 : t-1])   (20-day low ex-today)
    HOLD otherwise

Whipsaw protection: comparison uses `closes[-22:-2]`, excluding both today
and yesterday — a fresh breakout must persist for at least one bar.

SL/TP: reused from `app.strategies.atr_levels.compute_atr_levels` so this
strategy gets the same volatility-adaptive stops the ML signals use.
2:1 reward:risk by construction.

Signals are tagged `source='rule_donchian'` so the comparison tab can
aggregate them separately from the ML signals.
"""

from __future__ import annotations

import asyncpg
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from app.strategies.atr_levels import compute_atr_levels

logger = logging.getLogger(__name__)


# Strategy constants. Could be moved to `config` if tuning becomes routine,
# but Donchian's whole point is that its parameters are very stable across
# regimes — the canonical Turtle setup uses 20-day exactly.
DONCHIAN_WINDOW    = 20
HISTORY_NEEDED     = DONCHIAN_WINDOW + 2     # +1 for today, +1 for whipsaw-guard yesterday
SOURCE_TAG         = "rule_donchian"

# Confidence/EV defaults baked in. The rule fires deterministically so
# "confidence" is a misnomer, but the executor's filter chain expects a
# value ≥ min_confidence. 0.55 sits above the typical user min (0.50).
_RULE_CONFIDENCE   = 0.55
_RULE_PRED_RR      = 2.0      # matches ATR levels' 1.5×SL / 3.0×TP geometry
_RULE_EXPECTED_VAL = 0.0075   # = 0.55*1.5% - 0.45*1.0%  ≈ +75 bps prior


class DonchianService:
    """Generates BUY/SELL/HOLD signals from a 20-day Donchian breakout rule."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_for_ticker(
        self, ticker: str, interval: str = "1d",
    ) -> Optional[dict]:
        """Compute the Donchian signal for one ticker and persist it.

        Returns the stored signal dict, or None if there was an error
        (e.g. insufficient history). HOLD signals are persisted too so the
        comparison report has a complete record of when the rule was silent.
        """
        ticker = ticker.upper()

        # ── Step 1: pull the last HISTORY_NEEDED daily bars + the ATR snapshot
        async with self.pool.acquire() as conn:
            bars = await conn.fetch(
                """
                SELECT timestamp, close
                FROM ohlcv_data
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC
                LIMIT $3
                """,
                ticker, interval, HISTORY_NEEDED,
            )
            ind = await conn.fetchrow(
                """
                SELECT atr_14
                FROM indicators
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                ticker, interval,
            )

        # ── Step 2: edge-case checks before applying the rule
        if len(bars) < HISTORY_NEEDED:
            return await self._persist_hold(
                ticker, interval, current_close=None,
                reject_reasons=[f"insufficient_history({len(bars)}/{HISTORY_NEEDED})"],
            )

        # Skip if there's already an open position on this symbol from any
        # strategy — Donchian's "always in the market" stance means it would
        # otherwise stack opposing trades.
        async with self.pool.acquire() as conn:
            open_te = await conn.fetchval(
                """
                SELECT 1 FROM trade_executions
                WHERE symbol LIKE $1 || '%'
                  AND status IN ('filled', 'pending')
                LIMIT 1
                """,
                ticker,
            )
        # bars come back in DESC order; reverse for chronological logic
        bars_chrono = list(reversed(bars))
        closes = [float(r["close"]) for r in bars_chrono]
        current_close = closes[-1]

        if open_te:
            return await self._persist_hold(
                ticker, interval, current_close=current_close,
                reject_reasons=["open_position_exists"],
            )

        # ── Step 3: apply the Donchian breakout rule with whipsaw protection
        # Channel = max/min over [t-21 : t-1] (excludes today AND yesterday)
        channel_slice = closes[-HISTORY_NEEDED : -2]   # 20-bar window, neither today nor yesterday
        upper = max(channel_slice)
        lower = min(channel_slice)

        if current_close > upper:
            action = "BUY"
        elif current_close < lower:
            action = "SELL"
        else:
            action = "HOLD"

        # ── Step 4: compute ATR-mechanical SL/TP (reused from ATR helper)
        atr_14 = float(ind["atr_14"]) if ind and ind["atr_14"] is not None else None
        levels = compute_atr_levels(atr_14, current_close, action)

        # ── Step 5: build the result dict (shaped like predict_ticker's output)
        result = {
            "ticker":         ticker,
            "interval":       interval,
            "action":         action,
            "confidence":     _RULE_CONFIDENCE if action != "HOLD" else 0.0,
            "predicted_rr":   _RULE_PRED_RR    if action != "HOLD" else 0.0,
            "expected_value": _RULE_EXPECTED_VAL if action != "HOLD" else 0.0,
            "entry_price":    levels["entry"],
            "stop_loss":      levels["stop_loss"],
            "take_profit":    levels["take_profit"],
            "atr_stop_loss":   levels["stop_loss"],   # rule has no separate ATR path — same value
            "atr_take_profit": levels["take_profit"],
            "channel_upper":  upper,
            "channel_lower":  lower,
            "current_close":  current_close,
        }
        return await self._persist(ticker, interval, result, source=SOURCE_TAG)

    async def generate_batch(
        self, tickers: List[str], interval: str = "1d",
    ) -> List[dict]:
        """Run `generate_for_ticker` over a list. Failures don't abort the
        batch — each ticker's exception is logged and the loop continues."""
        out: list[dict] = []
        for t in tickers:
            try:
                sig = await self.generate_for_ticker(t, interval)
                if sig is not None:
                    out.append(sig)
            except Exception as e:
                logger.error(f"[Donchian] {t} failed: {e}")
        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    async def _persist(self, ticker: str, interval: str, result: dict, source: str) -> dict:
        """Insert the signal row, mirroring SignalService's schema. Donchian
        leaves the ML-specific columns NULL (probabilities, uncertainty,
        kelly fields, event_proximity) so the comparison report can detect
        rule-based rows by either `source` or by those NULLs."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals (
                    ticker, interval, action, confidence,
                    entry_price, stop_loss, take_profit, net_profit,
                    predicted_rr, expected_value,
                    atr_stop_loss, atr_take_profit,
                    reject_reasons, source, created_at
                ) VALUES (
                    $1,  $2,  $3,  $4,  $5,  $6,  $7,  $8,  $9, $10,
                    $11, $12, $13::jsonb, $14, $15
                )
                ON CONFLICT (ticker, interval, DATE(created_at AT TIME ZONE 'UTC'), COALESCE(source, ''))
                DO UPDATE SET
                    action          = EXCLUDED.action,
                    confidence      = EXCLUDED.confidence,
                    entry_price     = EXCLUDED.entry_price,
                    stop_loss       = EXCLUDED.stop_loss,
                    take_profit     = EXCLUDED.take_profit,
                    net_profit      = EXCLUDED.net_profit,
                    predicted_rr    = EXCLUDED.predicted_rr,
                    expected_value  = EXCLUDED.expected_value,
                    atr_stop_loss   = EXCLUDED.atr_stop_loss,
                    atr_take_profit = EXCLUDED.atr_take_profit,
                    reject_reasons  = EXCLUDED.reject_reasons,
                    created_at      = EXCLUDED.created_at
                RETURNING *
                """,
                ticker, interval, result["action"], result.get("confidence"),
                result.get("entry_price"), result.get("stop_loss"),
                result.get("take_profit"), None,   # net_profit unused for rule
                result.get("predicted_rr"), result.get("expected_value"),
                result.get("atr_stop_loss"), result.get("atr_take_profit"),
                json.dumps(result.get("reject_reasons", [])),
                source, datetime.now(timezone.utc),
            )

        doc = dict(row)
        logger.info(
            f"📐 Donchian signal: {ticker} → {result['action']}  "
            f"close={result.get('current_close'):.5f}  "
            f"channel=[{result.get('channel_lower'):.5f}, {result.get('channel_upper'):.5f}]"
            if result["action"] != "HOLD" and result.get("current_close") is not None
            else f"📐 Donchian signal: {ticker} → HOLD"
        )
        return doc

    async def _persist_hold(
        self, ticker: str, interval: str,
        current_close: Optional[float] = None,
        reject_reasons: Optional[List[str]] = None,
    ) -> dict:
        """Persist a HOLD row when an edge case prevents firing the rule."""
        result = {
            "action": "HOLD",
            "confidence": 0.0,
            "predicted_rr": 0.0,
            "expected_value": 0.0,
            "entry_price":  current_close,
            "stop_loss":    current_close,
            "take_profit":  current_close,
            "atr_stop_loss":   current_close,
            "atr_take_profit": current_close,
            "current_close": current_close,
            "channel_upper": current_close,
            "channel_lower": current_close,
            "reject_reasons": reject_reasons or [],
        }
        return await self._persist(ticker, interval, result, source=SOURCE_TAG)
