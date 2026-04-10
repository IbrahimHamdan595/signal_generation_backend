"""
Multi-Timeframe Confluence Service
====================================
Fetches the latest 1d and 1h signals for a ticker and computes a
confluence score (0–1).  When both timeframes agree, confidence is high.

Score logic:
  - Both BUY  : score = avg(conf_1d, conf_1h),  label = BUY
  - Both SELL : score = avg(conf_1d, conf_1h),  label = SELL
  - Disagree  : score = 0.0,                     label = HOLD
  - One HOLD  : score = conf_active * 0.5,       label = active direction
"""

import asyncpg
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConfluenceService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_confluence(self, ticker: str) -> dict:
        ticker = ticker.upper()

        async with self.pool.acquire() as conn:
            daily = await conn.fetchrow("""
                SELECT action, confidence, entry_price, stop_loss, take_profit,
                       prob_buy, prob_sell, prob_hold, created_at
                FROM signals
                WHERE ticker = $1 AND interval = '1d'
                ORDER BY created_at DESC LIMIT 1
            """, ticker)

            hourly = await conn.fetchrow("""
                SELECT action, confidence, entry_price, stop_loss, take_profit,
                       prob_buy, prob_sell, prob_hold, created_at
                FROM signals
                WHERE ticker = $1 AND interval = '1h'
                ORDER BY created_at DESC LIMIT 1
            """, ticker)

        daily_action  = daily["action"]  if daily  else "HOLD"
        hourly_action = hourly["action"] if hourly else "HOLD"
        daily_conf    = float(daily["confidence"])  if daily  else 0.0
        hourly_conf   = float(hourly["confidence"]) if hourly else 0.0

        # ── Compute confluence ────────────────────────────────────────────────
        if daily_action == hourly_action and daily_action != "HOLD":
            # Full agreement — strongest signal
            label = daily_action
            score = round((daily_conf + hourly_conf) / 2, 4)
            strength = "strong"
        elif daily_action != "HOLD" and hourly_action == "HOLD":
            label = daily_action
            score = round(daily_conf * 0.5, 4)
            strength = "weak"
        elif hourly_action != "HOLD" and daily_action == "HOLD":
            label = hourly_action
            score = round(hourly_conf * 0.5, 4)
            strength = "weak"
        else:
            # Conflict or both HOLD
            label = "HOLD"
            score = 0.0
            strength = "conflicting" if (daily_action != hourly_action and
                                          daily_action != "HOLD" and
                                          hourly_action != "HOLD") else "neutral"

        return {
            "ticker":    ticker,
            "label":     label,
            "score":     score,
            "strength":  strength,
            "daily": {
                "action":     daily_action,
                "confidence": daily_conf,
                "entry_price":  float(daily["entry_price"])  if daily and daily["entry_price"]  else None,
                "stop_loss":    float(daily["stop_loss"])    if daily and daily["stop_loss"]    else None,
                "take_profit":  float(daily["take_profit"])  if daily and daily["take_profit"]  else None,
                "updated_at":   daily["created_at"].isoformat() if daily else None,
            },
            "hourly": {
                "action":     hourly_action,
                "confidence": hourly_conf,
                "entry_price":  float(hourly["entry_price"])  if hourly and hourly["entry_price"]  else None,
                "stop_loss":    float(hourly["stop_loss"])    if hourly and hourly["stop_loss"]    else None,
                "take_profit":  float(hourly["take_profit"])  if hourly and hourly["take_profit"]  else None,
                "updated_at":   hourly["created_at"].isoformat() if hourly else None,
            },
        }

    async def get_confluence_batch(self, tickers: list[str]) -> list[dict]:
        results = []
        for ticker in tickers:
            try:
                results.append(await self.get_confluence(ticker))
            except Exception as e:
                logger.warning(f"Confluence error for {ticker}: {e}")
        return results
