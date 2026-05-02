"""
Real-Fill Backtest Service
==========================
Honest backtest that walks actual OHLCV bars forward from each signal's
timestamp and resolves WIN/LOSS/EXPIRED at the first SL or TP touch —
the same logic OutcomeService uses for live signals.

Replaces the fictional `_trading_metrics` in evaluator.py which used the
model's own predicted TP/SL as ground-truth returns. This service uses
real future prices, so Sharpe / win-rate reflect what would actually have
happened.

Outputs match the SignalOutcome schema so live and backtested results are
directly comparable.
"""

import asyncpg
import numpy as np
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

MAX_BARS_HELD = 10  # match OutcomeService


class RealFillBacktestService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def backtest(
        self,
        ticker: str,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """
        Walk every stored signal between start_date and end_date and resolve
        each one against actual subsequent bars. Returns aggregate metrics.
        """
        async with self.pool.acquire() as conn:
            signals = await conn.fetch(
                """
                SELECT id, ticker, interval, action, confidence,
                       entry_price, stop_loss, take_profit, created_at
                FROM signals
                WHERE ticker = $1 AND interval = $2
                  AND action IN ('BUY','SELL')
                  AND entry_price IS NOT NULL
                  AND stop_loss   IS NOT NULL
                  AND take_profit IS NOT NULL
                  AND ($3::timestamptz IS NULL OR created_at >= $3)
                  AND ($4::timestamptz IS NULL OR created_at <= $4)
                ORDER BY created_at ASC
                """,
                ticker.upper(), interval, start_date, end_date,
            )

        if not signals:
            return {"ticker": ticker, "total_signals": 0, "trades": []}

        trades = []
        for sig in signals:
            outcome = await self._walk_forward(sig, interval)
            if outcome:
                trades.append(outcome)

        return self._aggregate(ticker, interval, trades)

    async def backtest_portfolio(
        self,
        tickers: list[str],
        interval: str = "1d",
    ) -> dict:
        """Run backtest across many tickers and aggregate."""
        all_trades = []
        per_ticker = {}
        for t in tickers:
            res = await self.backtest(t, interval)
            per_ticker[t] = res
            all_trades.extend(res.get("trades", []))

        agg = self._aggregate("PORTFOLIO", interval, all_trades)
        agg["per_ticker"] = {
            t: {k: v for k, v in r.items() if k != "trades"}
            for t, r in per_ticker.items()
        }
        return agg

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _walk_forward(self, signal, interval: str) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            bars = await conn.fetch(
                """
                SELECT high, low, close, timestamp
                FROM ohlcv_data
                WHERE ticker = $1 AND interval = $2 AND timestamp > $3
                ORDER BY timestamp ASC LIMIT $4
                """,
                signal["ticker"], interval, signal["created_at"], MAX_BARS_HELD,
            )
        if not bars:
            return None

        action = signal["action"]
        entry  = float(signal["entry_price"])
        sl     = float(signal["stop_loss"])
        tp     = float(signal["take_profit"])

        outcome     = "EXPIRED"
        exit_price  = float(bars[-1]["close"])
        bars_held   = len(bars)

        for i, bar in enumerate(bars):
            hi, lo = float(bar["high"]), float(bar["low"])
            if action == "BUY":
                if hi >= tp:
                    outcome, exit_price, bars_held = "WIN",  tp, i + 1; break
                if lo <= sl:
                    outcome, exit_price, bars_held = "LOSS", sl, i + 1; break
            else:  # SELL
                if lo <= tp:
                    outcome, exit_price, bars_held = "WIN",  tp, i + 1; break
                if hi >= sl:
                    outcome, exit_price, bars_held = "LOSS", sl, i + 1; break

        if action == "BUY":
            ret = (exit_price - entry) / entry
        else:
            ret = (entry - exit_price) / entry

        return {
            "signal_id":   signal["id"],
            "ticker":      signal["ticker"],
            "action":      action,
            "confidence":  float(signal["confidence"]),
            "entry":       entry,
            "exit":        exit_price,
            "outcome":     outcome,
            "return_pct":  ret,
            "bars_held":   bars_held,
        }

    def _aggregate(self, label: str, interval: str, trades: list[dict]) -> dict:
        if not trades:
            return {
                "ticker": label, "interval": interval,
                "total_signals": 0, "wins": 0, "losses": 0, "expired": 0,
                "win_rate": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "avg_return": 0.0, "total_return": 0.0, "trades": [],
            }

        rets = np.array([t["return_pct"] for t in trades])
        wins = sum(1 for t in trades if t["outcome"] == "WIN")
        losses = sum(1 for t in trades if t["outcome"] == "LOSS")
        expired = sum(1 for t in trades if t["outcome"] == "EXPIRED")
        total = len(trades)

        # Annualisation by interval (matches evaluator.BARS_PER_YEAR_BY_INTERVAL)
        bars_per_year = {"1d": 252, "1h": 1638, "30m": 3276, "15m": 6552, "5m": 19656}.get(interval, 252)
        ann = float(np.sqrt(bars_per_year))
        std = float(rets.std()) + 1e-8
        sharpe = float(rets.mean() / std * ann)

        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        max_dd = float(((cum - peak) / peak).min())

        return {
            "ticker":        label,
            "interval":      interval,
            "total_signals": total,
            "wins":          wins,
            "losses":        losses,
            "expired":       expired,
            "win_rate":      wins / total,
            "sharpe":        round(sharpe, 4),
            "max_drawdown":  round(max_dd, 4),
            "avg_return":    round(float(rets.mean()), 6),
            "total_return":  round(float(cum[-1] - 1.0), 4),
            "trades":        trades,
        }
