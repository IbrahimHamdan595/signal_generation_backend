"""
Backtesting Engine
==================
Replays the trained ML model over historical OHLCV data bar-by-bar,
simulates trades using the model's entry/SL/TP predictions, and builds
a full equity curve + trade log.

Key design choices:
- Uses stored signals (already generated) rather than re-running inference,
  so results are deterministic and fast.
- Each BUY/SELL signal is tracked through subsequent bars until TP or SL hit.
- HOLD signals are skipped (no position opened).
- Slippage: 0.05% per trade (configurable).
- Commission: $0 (CFD-style, configurable).
"""

import asyncpg
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

SLIPPAGE = 0.0005   # 0.05% per side


class BacktestService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def run(
        self,
        ticker: str,
        interval: str = "1d",
        initial_capital: float = 10_000.0,
        position_size_pct: float = 0.10,   # 10% of equity per trade
        max_bars_held: int = 10,
        slippage: float = SLIPPAGE,
    ) -> dict:
        """
        Run a backtest for one ticker using stored signals.
        Returns equity curve, trade log, and summary statistics.
        """
        ticker = ticker.upper()

        async with self.pool.acquire() as conn:
            signals = await conn.fetch("""
                SELECT s.id, s.action, s.confidence, s.entry_price,
                       s.stop_loss, s.take_profit, s.created_at
                FROM signals s
                WHERE s.ticker   = $1
                  AND s.interval = $2
                  AND s.action   IN ('BUY','SELL')
                  AND s.entry_price IS NOT NULL
                  AND s.stop_loss   IS NOT NULL
                  AND s.take_profit IS NOT NULL
                ORDER BY s.created_at ASC
            """, ticker, interval)

            ohlcv = await conn.fetch("""
                SELECT timestamp, open, high, low, close
                FROM ohlcv_data
                WHERE ticker = $1 AND interval = $2
                ORDER BY timestamp ASC
            """, ticker, interval)

        if not signals or not ohlcv:
            return {"error": f"No signals or OHLCV data for {ticker}"}

        # Index OHLCV by timestamp for fast lookup
        ohlcv_index = {row["timestamp"]: row for row in ohlcv}
        ohlcv_list  = list(ohlcv)

        equity        = initial_capital
        peak_equity   = initial_capital
        max_drawdown  = 0.0
        trades        = []
        equity_curve  = [{"date": ohlcv_list[0]["timestamp"].isoformat(),
                           "equity": equity}]

        for sig in signals:
            if sig["action"] == "HOLD":
                continue

            action      = sig["action"]
            entry_price = float(sig["entry_price"]) * (1 + slippage if action == "BUY" else 1 - slippage)
            stop_loss   = float(sig["stop_loss"])
            take_profit = float(sig["take_profit"])
            position_val = equity * position_size_pct
            shares        = position_val / entry_price

            # Walk forward bars to find exit
            sig_ts  = sig["created_at"]
            outcome = "EXPIRED"
            exit_px = None
            exit_ts = None
            bars_h  = 0

            future_bars = [b for b in ohlcv_list if b["timestamp"] > sig_ts][:max_bars_held]
            for bar in future_bars:
                bars_h += 1
                hi = float(bar["high"])
                lo = float(bar["low"])

                if action == "BUY":
                    if hi >= take_profit:
                        outcome = "WIN";  exit_px = take_profit; exit_ts = bar["timestamp"]; break
                    if lo <= stop_loss:
                        outcome = "LOSS"; exit_px = stop_loss;   exit_ts = bar["timestamp"]; break
                else:
                    if lo <= take_profit:
                        outcome = "WIN";  exit_px = take_profit; exit_ts = bar["timestamp"]; break
                    if hi >= stop_loss:
                        outcome = "LOSS"; exit_px = stop_loss;   exit_ts = bar["timestamp"]; break

            if exit_px is None:
                exit_px = float(future_bars[-1]["close"]) if future_bars else entry_price
                exit_ts = future_bars[-1]["timestamp"]   if future_bars else sig_ts

            # Apply slippage on exit
            exit_px_net = exit_px * (1 - slippage if action == "BUY" else 1 + slippage)

            if action == "BUY":
                pnl = shares * (exit_px_net - entry_price)
                ret = (exit_px_net - entry_price) / entry_price
            else:
                pnl = shares * (entry_price - exit_px_net)
                ret = (entry_price - exit_px_net) / entry_price

            equity += pnl
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, dd)

            trades.append({
                "signal_id":   sig["id"],
                "action":      action,
                "entry_price": round(entry_price, 4),
                "exit_price":  round(exit_px_net, 4),
                "stop_loss":   round(stop_loss, 4),
                "take_profit": round(take_profit, 4),
                "outcome":     outcome,
                "return_pct":  round(ret * 100, 3),
                "pnl":         round(pnl, 2),
                "bars_held":   bars_h,
                "entry_time":  sig["created_at"].isoformat(),
                "exit_time":   exit_ts.isoformat() if exit_ts else None,
                "confidence":  round(float(sig["confidence"]), 4),
            })

            equity_curve.append({
                "date":   (exit_ts or sig["created_at"]).isoformat(),
                "equity": round(equity, 2),
            })

        if not trades:
            return {"error": "No trades executed — generate signals first"}

        wins     = [t for t in trades if t["outcome"] == "WIN"]
        losses   = [t for t in trades if t["outcome"] == "LOSS"]
        rets     = np.array([t["return_pct"] / 100 for t in trades])
        sharpe   = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)) if len(rets) > 1 else 0.0
        total_ret = (equity - initial_capital) / initial_capital

        return {
            "ticker":           ticker,
            "interval":         interval,
            "initial_capital":  initial_capital,
            "final_equity":     round(equity, 2),
            "total_return_pct": round(total_ret * 100, 2),
            "total_trades":     len(trades),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         round(len(wins) / len(trades), 4) if trades else 0.0,
            "avg_return_pct":   round(float(np.mean(rets)) * 100, 3),
            "sharpe_ratio":     round(sharpe, 4),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "equity_curve":     equity_curve,
            "trades":           trades,
        }

    async def run_portfolio(
        self,
        tickers: list[str],
        interval: str = "1d",
        initial_capital: float = 100_000.0,
    ) -> dict:
        """Run backtest across multiple tickers, equal-weight capital allocation."""
        per_ticker_capital = initial_capital / max(len(tickers), 1)
        results = []
        for ticker in tickers:
            r = await self.run(ticker, interval, per_ticker_capital)
            if "error" not in r:
                results.append(r)

        if not results:
            return {"error": "No backtest results"}

        total_final   = sum(r["final_equity"] for r in results)
        total_return  = (total_final - initial_capital) / initial_capital
        avg_sharpe    = float(np.mean([r["sharpe_ratio"] for r in results]))
        avg_win_rate  = float(np.mean([r["win_rate"] for r in results]))
        avg_max_dd    = float(np.mean([r["max_drawdown_pct"] for r in results]))

        return {
            "initial_capital":  initial_capital,
            "final_equity":     round(total_final, 2),
            "total_return_pct": round(total_return * 100, 2),
            "avg_sharpe":       round(avg_sharpe, 4),
            "avg_win_rate":     round(avg_win_rate, 4),
            "avg_max_drawdown": round(avg_max_dd, 2),
            "n_tickers":        len(results),
            "tickers":          results,
        }
