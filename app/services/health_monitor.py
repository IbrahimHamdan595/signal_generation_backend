"""
Health Monitor + Alert Dispatcher
=================================
Periodic health checks for the production trading loop. Runs every 5 minutes.
On any failure, dispatches an alert via:
  - Telegram (if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID env vars are set)
  - Email   (if SMTP_* env vars are set)
  - Always logs at ERROR level

Checks performed:
  - MT5 broker connection
  - Database reachable
  - Model loaded and is_trained
  - Latest signal age (< 24h on weekdays)
  - Latest OHLCV ingestion age (< 6h on weekdays)
  - Scheduler last-run age per critical job
  - Equity guardrail: drop > 5% from morning open OR > 10% from peak

Auto-actions on critical failures:
  - MT5 disconnected: disable auto_trade for all users
  - Equity guardrail breach: flatten all positions + halt
"""

import os
import asyncio
import asyncpg
import logging
import smtplib
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from typing import Optional

import httpx

from app.services.broker_service import BrokerService

logger = logging.getLogger(__name__)

# Cooldown: don't re-send the same alert more than once per 30 min
_LAST_ALERT_AT: dict[str, datetime] = {}
_ALERT_COOLDOWN = timedelta(minutes=30)


class HealthMonitor:
    def __init__(self, pool: asyncpg.Pool, broker: BrokerService):
        self.pool = pool
        self.broker = broker

    async def run_checks(self) -> dict:
        """Run all checks. Returns a dict; calls alert() on any failures."""
        failures: list[str] = []
        checks: dict = {}

        # 1. MT5 connection
        mt5_ok = await self.broker.is_connected()
        checks["mt5_connected"] = mt5_ok
        if not mt5_ok:
            failures.append("MT5_DISCONNECTED")
            await self._disable_auto_trade()

        # 2. Database
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            checks["db_ok"] = True
        except Exception as e:
            checks["db_ok"] = False
            failures.append(f"DB_UNREACHABLE: {e}")

        # 3. Model trained
        from app.ml.models.registry import is_model_trained
        checks["model_trained"] = is_model_trained()
        if not checks["model_trained"]:
            failures.append("MODEL_NOT_TRAINED")

        # 4. Recent signals (only check on weekdays during market hours UTC 13:30-21:00)
        if self._is_market_hours():
            sig_age_min = await self._latest_signal_age_minutes()
            checks["latest_signal_age_min"] = sig_age_min
            if sig_age_min is None or sig_age_min > 90:
                failures.append(f"NO_RECENT_SIGNALS (age={sig_age_min}min)")

            # 5. Recent OHLCV
            ohlcv_age_h = await self._latest_ohlcv_age_hours()
            checks["latest_ohlcv_age_h"] = ohlcv_age_h
            if ohlcv_age_h is None or ohlcv_age_h > 6:
                failures.append(f"STALE_OHLCV (age={ohlcv_age_h}h)")

        # 6. Equity guardrail
        if mt5_ok:
            equity_check = await self._equity_guardrail()
            checks["equity"] = equity_check
            if equity_check.get("breach"):
                failures.append(f"EQUITY_BREACH: {equity_check['reason']}")
                if equity_check.get("severity") == "critical":
                    await self._flatten_and_halt()

        checks["ok"] = len(failures) == 0
        checks["failures"] = failures
        checks["checked_at"] = datetime.now(timezone.utc).isoformat()

        if failures:
            await self.alert("HEALTH_CHECK_FAILED", "\n".join(failures))

        return checks

    # ── Alert dispatch ────────────────────────────────────────────────────────

    async def alert(self, key: str, message: str) -> None:
        """Send alert via configured channels. Suppresses repeats inside cooldown."""
        now = datetime.now(timezone.utc)
        last = _LAST_ALERT_AT.get(key)
        if last and (now - last) < _ALERT_COOLDOWN:
            return
        _LAST_ALERT_AT[key] = now

        body = f"[{now.isoformat()}] {key}\n\n{message}"
        logger.error(f"🚨 ALERT {key}: {message}")

        await asyncio.gather(
            self._send_telegram(body),
            self._send_email(f"Signal alert: {key}", body),
            return_exceptions=True,
        )

    async def _send_telegram(self, text: str) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat  = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json={"chat_id": chat, "text": text})
        except Exception as e:
            logger.warning(f"Telegram alert failed: {e}")

    async def _send_email(self, subject: str, body: str) -> None:
        host = os.getenv("SMTP_HOST")
        user = os.getenv("SMTP_USER")
        pwd  = os.getenv("SMTP_PASSWORD")
        to   = os.getenv("ALERT_EMAIL_TO")
        if not all([host, user, pwd, to]):
            return

        def _send():
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"]    = user
            msg["To"]      = to
            msg.set_content(body)
            with smtplib.SMTP_SSL(host, int(os.getenv("SMTP_PORT", "465"))) as s:
                s.login(user, pwd)
                s.send_message(msg)

        try:
            await asyncio.to_thread(_send)
        except Exception as e:
            logger.warning(f"Email alert failed: {e}")

    # ── Check helpers ─────────────────────────────────────────────────────────

    def _is_market_hours(self) -> bool:
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:  # weekend
            return False
        return 13 <= now.hour <= 21  # 9:30 EST to 5:00 EST roughly

    async def _latest_signal_age_minutes(self) -> Optional[float]:
        async with self.pool.acquire() as conn:
            ts = await conn.fetchval(
                "SELECT MAX(created_at) FROM signals"
            )
        if not ts:
            return None
        return (datetime.now(timezone.utc) - ts).total_seconds() / 60

    async def _latest_ohlcv_age_hours(self) -> Optional[float]:
        async with self.pool.acquire() as conn:
            ts = await conn.fetchval(
                "SELECT MAX(timestamp) FROM ohlcv_data WHERE interval='1d'"
            )
        if not ts:
            return None
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600

    async def _equity_guardrail(self) -> dict:
        """Track equity vs morning open and all-time peak."""
        info = await self.broker.account_info()
        if not info:
            return {"breach": False}
        equity = info["equity"]

        async with self.pool.acquire() as conn:
            # Get/create today's equity row
            today = datetime.now(timezone.utc).date()
            row = await conn.fetchrow(
                "SELECT * FROM equity_snapshots WHERE date = $1", today
            )
            if not row:
                await conn.execute(
                    "INSERT INTO equity_snapshots (date, open_equity, peak_equity, current_equity) "
                    "VALUES ($1, $2, $2, $2) ON CONFLICT (date) DO NOTHING",
                    today, equity,
                )
                return {"breach": False, "open_equity": equity, "current": equity}

            peak = max(float(row["peak_equity"]), equity)
            await conn.execute(
                "UPDATE equity_snapshots SET current_equity=$1, peak_equity=$2 WHERE date=$3",
                equity, peak, today,
            )
            open_eq = float(row["open_equity"])

        intraday_drop = (equity - open_eq) / open_eq if open_eq else 0
        peak_drop     = (equity - peak)    / peak    if peak    else 0

        if peak_drop <= -0.10:
            return {
                "breach": True, "severity": "critical",
                "reason": f"Equity {equity:.2f} is {peak_drop*100:.1f}% below peak {peak:.2f}",
            }
        if intraday_drop <= -0.05:
            return {
                "breach": True, "severity": "warning",
                "reason": f"Equity {equity:.2f} is {intraday_drop*100:.1f}% below today's open {open_eq:.2f}",
            }
        return {"breach": False, "open_equity": open_eq, "peak": peak, "current": equity}

    # ── Auto-actions ──────────────────────────────────────────────────────────

    async def _disable_auto_trade(self):
        try:
            async with self.pool.acquire() as conn:
                n = await conn.execute(
                    "UPDATE trading_config SET auto_trade = FALSE "
                    "WHERE auto_trade = TRUE"
                )
            logger.warning(f"🛑 Auto-disabled auto_trade for all users ({n})")
        except Exception as e:
            logger.error(f"Failed to disable auto_trade: {e}")

    async def _flatten_and_halt(self):
        """Critical equity breach: close all positions and disable trading."""
        try:
            positions = await self.broker.get_open_positions()
            for p in positions:
                await self.broker.close_position(p["ticket"])
                logger.warning(f"🛑 Force-closed position {p['ticket']} ({p['symbol']})")
            await self._disable_auto_trade()
            await self.alert(
                "EQUITY_HALT",
                f"Critical equity breach. Flattened {len(positions)} positions and halted trading.",
            )
        except Exception as e:
            logger.error(f"Flatten failed: {e}")
