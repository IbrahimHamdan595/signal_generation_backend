"""
Execution Service
=================

Bridges generated signals with the MT5 broker. Handles:

  • Position sizing  — fixed-fraction risk (risk_per_trade_pct % of balance
                       divided by the distance from entry to stop-loss)
  • Risk controls    — max open positions, daily loss circuit breaker,
                       minimum confidence threshold
  • Order placement  — sends a market order via BrokerService, stores the
                       result in trade_executions, and syncs the positions
                       table
  • Auto-execution   — called by the scheduler after each signal regen;
                       only fires for users who have auto_trade = TRUE

All writes go to two tables added in migration 0002:
  trade_executions — immutable audit log of every order sent to MT5
  trading_config   — per-user settings (risk %, limits, MT5 credentials)
"""

import asyncpg
import logging
from datetime import datetime, timezone, date
from typing import Optional

from app.services.broker_service import BrokerService

logger = logging.getLogger(__name__)


class ExecutionService:
    def __init__(self, pool: asyncpg.Pool, broker: BrokerService):
        self.pool = pool
        self.broker = broker

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute_signal(
        self, signal_id: int, user_id: int, volume_override: Optional[float] = None
    ) -> dict:
        """
        Manually execute a specific signal for a user.
        Performs all risk checks then places the order.
        volume_override skips auto position-sizing and uses the supplied lot size.
        """
        config = await self._get_config(user_id)
        if not config:
            return {"ok": False, "error": "Trading not configured for this user"}
        if not config["enabled"]:
            return {"ok": False, "error": "Trading is disabled — enable it in /trading/config"}

        signal = await self._get_signal(signal_id)
        if not signal:
            return {"ok": False, "error": f"Signal {signal_id} not found"}

        return await self._place_for_signal(signal, config, user_id, volume_override=volume_override)

    async def auto_execute(self, user_id: int) -> list[dict]:
        """
        Execute recent BUY/SELL signals that have not been executed yet for
        users with auto_trade = TRUE.

        **Per-pipeline quota allocation with redistribute.** Slots are divided
        evenly across enabled pipelines that have actionable signals:

            quota = (max_open_positions - currently_open) // n_active_pipelines

        Each pipeline first takes its top-`quota` signals ranked by confidence
        DESC (EV DESC tiebreak). Any leftover capacity — integer-division
        remainder plus unused quota from pipelines that ran short — is then
        redistributed round-robin to pipelines that still have signals. So
        with max=12 and three pipelines producing 8/2/9 actionable signals,
        the split is 5/2/5 rather than 4/2/4.

        Trades are placed in interleaved order (1 from pipe-A, 1 from pipe-B,
        …) so a partial-fill abort still leaves a balanced mix.

        Called by `/trading/run-now` (Generate & Trade) and the hourly :15
        scheduler tick — both paths share this policy.
        """
        config = await self._get_config(user_id)
        if not config or not config["enabled"] or not config["auto_trade"]:
            return []

        # ── How many slots can we still open? ─────────────────────────────────
        # Computed up-front so the quota divides the *remaining* capacity,
        # not the total. `_place_for_signal` still re-checks per trade for
        # late-arriving positions.
        try:
            mt5_positions = (
                await self.broker.get_open_positions()
                if await self.broker.is_connected()
                else []
            )
        except Exception as e:
            logger.warning(f"Auto-execute: could not read MT5 positions ({e}), assuming 0 open")
            mt5_positions = []
        slots_remaining = max(0, int(config["max_open_positions"]) - len(mt5_positions))
        if slots_remaining == 0:
            logger.info(
                f"Auto-execute: max_open_positions ({config['max_open_positions']}) "
                f"already reached, nothing to do"
            )
            return []

        # Signals from the last 24 hours that haven't been successfully filled.
        # Ordered confidence-first so each pipeline's top-conf signals win its
        # quota; EV is the tiebreak so Donchian's fixed-confidence rows still
        # order sensibly.
        async with self.pool.acquire() as conn:
            signals = await conn.fetch(
                """
                SELECT s.* FROM signals s
                WHERE s.action IN ('BUY', 'SELL')
                  AND s.confidence >= $1
                  AND COALESCE(s.expected_value, 0) > 0
                  AND s.created_at >= NOW() - INTERVAL '24 hours'
                  AND NOT EXISTS (
                      SELECT 1 FROM trade_executions te
                      WHERE te.signal_id = s.id
                        AND te.user_id = $2
                        AND te.status IN ('filled', 'pending')
                  )
                ORDER BY COALESCE(s.source, 'ml_equities'),
                         s.confidence DESC,
                         COALESCE(s.expected_value, 0) DESC
                """,
                config["min_confidence"],
                user_id,
            )

        # Bucket by source. The SQL above already orders each bucket
        # confidence-DESC then EV-DESC, so simple list slicing picks top-N.
        buckets: dict[str, list[dict]] = {}
        for row in signals:
            src = row.get("source") or "ml_equities"
            buckets.setdefault(src, []).append(dict(row))

        # Stable source order keeps placements deterministic across runs.
        SOURCE_ORDER = ["ml_equities", "ml_fx", "rule_donchian"]
        active = [s for s in SOURCE_ORDER if buckets.get(s)] + [
            s for s in buckets if s not in SOURCE_ORDER   # any unexpected source
        ]
        n_active = len(active)
        if n_active == 0:
            return []

        quota = slots_remaining // n_active
        selected_by_src: dict[str, list[dict]] = {s: [] for s in active}

        # Phase 1 — each pipeline takes its quota (or fewer if it ran short).
        for src in active:
            take = min(quota, len(buckets[src]))
            selected_by_src[src] = buckets[src][:take]
            buckets[src] = buckets[src][take:]

        # Phase 2 — distribute the remainder + any unused-quota slots in
        # round-robin from pipelines that still have signals. A pipeline that
        # produced fewer signals than its quota gives its leftover slots to
        # others, so e.g. 12 slots / 3 pipelines with 2/8/9 actionable becomes
        # 2/5/5 rather than 2/4/4.
        spare = slots_remaining - sum(len(v) for v in selected_by_src.values())
        while spare > 0:
            progressed = False
            for src in active:
                if spare <= 0:
                    break
                if buckets[src]:
                    selected_by_src[src].append(buckets[src].pop(0))
                    spare -= 1
                    progressed = True
            if not progressed:
                break  # no pipeline has any more signals

        # Interleave for placement: trade #1 from pipe-A, #2 from pipe-B, etc.
        # If the loop aborts early (late-arriving position fills max_open
        # mid-flight) the executed trades still keep a balanced pipeline mix.
        ordered: list[dict] = []
        queues = {s: list(selected_by_src[s]) for s in active}
        while any(queues.values()):
            for src in active:
                if queues[src]:
                    ordered.append(queues[src].pop(0))

        if ordered:
            tally = {s: len(selected_by_src[s]) for s in active}
            logger.info(
                f"Auto-execute (per-pipeline quota): slots={slots_remaining} "
                f"n_pipelines={n_active} quota={quota} -> "
                f"{', '.join(f'{k}={v}' for k, v in tally.items())}"
            )

        results = []
        for signal in ordered:
            result = await self._place_for_signal(signal, config, user_id)
            results.append(result)
            if not result["ok"]:
                logger.warning(
                    f"Auto-execute skipped signal {signal['id']} ({signal.get('source')}) "
                    f"for user {user_id}: {result.get('error')}"
                )

        return results

    async def sync_positions(self, user_id: int) -> int:
        """
        Pull open positions from MT5 and reconcile with the local positions
        table. Returns the number of positions synced.
        """
        if not await self.broker.is_connected():
            return 0

        mt5_positions = await self.broker.get_open_positions()

        synced = 0
        async with self.pool.acquire() as conn:
            for p in mt5_positions:
                # Upsert: update price/profit if already tracked, insert if new
                await conn.execute(
                    """
                    INSERT INTO positions (user_id, ticker, quantity, avg_cost, is_open)
                    VALUES ($1, $2, $3, $4, TRUE)
                    ON CONFLICT DO NOTHING
                    """,
                    user_id,
                    p["symbol"],
                    p["volume"],
                    p["open_price"],
                )
                synced += 1

        return synced

    async def close_position_by_ticket(
        self, ticket: int, user_id: int
    ) -> dict:
        """Close a position by its MT5 ticket number."""
        result = await self.broker.close_position(ticket)
        if not result.get("success"):
            return {"ok": False, "error": result.get("comment", "Unknown error")}

        # Mark the trade_execution row as closed
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE trade_executions
                   SET status = 'closed',
                       pnl    = $1,
                       closed_at = NOW()
                 WHERE mt5_ticket = $2 AND user_id = $3
                """,
                result.get("profit", 0.0),
                ticket,
                user_id,
            )

        return {"ok": True, "ticket": ticket, "close_price": result.get("close_price")}

    # ── Config CRUD ───────────────────────────────────────────────────────────

    async def get_config(self, user_id: int) -> Optional[dict]:
        return await self._get_config(user_id)

    async def upsert_config(self, user_id: int, updates: dict) -> dict:
        """
        Create or update the trading_config row for a user.
        Only the keys present in ``updates`` are written.
        """
        allowed = {
            "enabled", "auto_trade", "risk_per_trade_pct",
            "max_open_positions", "max_daily_loss_pct",
            "min_confidence", "allowed_actions",
            "mt5_account", "mt5_server", "mt5_path", "symbol_suffix",
        }
        safe = {k: v for k, v in updates.items() if k in allowed}

        async with self.pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT * FROM trading_config WHERE user_id = $1", user_id
            )
            if not existing:
                await conn.execute(
                    "INSERT INTO trading_config (user_id) VALUES ($1)", user_id
                )

            for col, val in safe.items():
                await conn.execute(
                    f"UPDATE trading_config SET {col} = $1, updated_at = NOW() "
                    f"WHERE user_id = $2",
                    val, user_id,
                )

        return await self._get_config(user_id)

    # ── Read helpers ──────────────────────────────────────────────────────────

    async def get_executions(
        self, user_id: int, limit: int = 20, offset: int = 0
    ) -> dict:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT te.*, s.ticker, s.action, s.confidence
                FROM trade_executions te
                LEFT JOIN signals s ON s.id = te.signal_id
                WHERE te.user_id = $1
                ORDER BY te.created_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id, limit, offset,
            )
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM trade_executions WHERE user_id = $1",
                user_id,
            )
        return {"items": [dict(r) for r in rows], "total": total}

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get_config(self, user_id: int) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM trading_config WHERE user_id = $1", user_id
            )
        return dict(row) if row else None

    async def _get_signal(self, signal_id: int) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM signals WHERE id = $1", signal_id
            )
        return dict(row) if row else None

    async def _place_for_signal(
        self, signal: dict, config: dict, user_id: int,
        volume_override: Optional[float] = None,
    ) -> dict:
        """
        Run all pre-trade checks and place the order if everything passes.
        Returns a result dict with keys: ok, signal_id, ticket, error, ...
        """
        signal_id = signal["id"]
        ticker = signal["ticker"]
        action = signal["action"]   # BUY | SELL
        entry = signal.get("entry_price")
        sl = signal.get("stop_loss")
        tp = signal.get("take_profit")

        # ── Guard: action must be tradeable ──────────────────────────────────
        allowed = config.get("allowed_actions") or ["BUY", "SELL"]
        if action not in allowed:
            return {"ok": False, "signal_id": signal_id,
                    "error": f"Action {action} not in allowed_actions"}

        # ── Guard: stop-loss required for position sizing ────────────────────
        if not entry or not sl:
            return {"ok": False, "signal_id": signal_id,
                    "error": "Signal missing entry_price or stop_loss — cannot size position"}

        # ── Guard: daily loss circuit breaker ─────────────────────────────────
        if not await self._daily_loss_ok(user_id, config):
            return {"ok": False, "signal_id": signal_id,
                    "error": "Daily loss limit reached — trading suspended for today"}

        # ── Guard: max open positions ─────────────────────────────────────────
        mt5_positions = await self.broker.get_open_positions()
        if len(mt5_positions) >= config["max_open_positions"]:
            return {"ok": False, "signal_id": signal_id,
                    "error": f"Max open positions ({config['max_open_positions']}) reached"}

        # ── Get account balance for sizing ────────────────────────────────────
        account = await self.broker.account_info()
        if not account:
            return {"ok": False, "signal_id": signal_id,
                    "error": "Could not read account info from MT5"}

        balance = account["balance"]

        # ── Resolve MT5 symbol ────────────────────────────────────────────────
        suffix = config.get("symbol_suffix") or ""
        symbol = ticker + suffix

        sym_info = await self.broker.get_symbol_info(symbol)
        if not sym_info:
            return {"ok": False, "signal_id": signal_id,
                    "error": f"Symbol '{symbol}' not found in MT5 — check symbol_suffix"}

        # Use the broker's actual symbol name (may differ from ticker, e.g. Apple vs AAPL)
        symbol = sym_info["symbol"]

        # ── Position sizing ────────────────────────────────────────────────────
        if volume_override is not None:
            # Clamp manual lot size to broker limits
            step = sym_info["volume_step"]
            steps = int(volume_override / step)
            volume = round(steps * step, 8)
            volume = max(sym_info["volume_min"], min(volume, sym_info["volume_max"]))
        else:
            # Hybrid sizing: take the SMALLER of fixed-fraction risk and
            # Kelly-scaled risk. Kelly bumps risk UP on high-edge signals
            # (within the user's risk_per_trade_pct ceiling) and DOWN on
            # marginal ones. Falls back to fixed-fraction if kelly_fraction
            # is missing (legacy signal rows).
            kelly_fraction = signal.get("kelly_fraction")
            risk_pct = config["risk_per_trade_pct"]
            if kelly_fraction is not None and kelly_fraction > 0:
                # kelly_fraction is in [0, 1] — interpret as a multiplier on the
                # user's risk budget. A kelly_fraction of 0.05 → use 5% of the
                # configured risk_per_trade_pct on this trade.
                effective_risk_pct = min(risk_pct, risk_pct * kelly_fraction * 4)
                logger.info(
                    f"📐 Kelly sizing for sig#{signal_id}: kelly_fraction={kelly_fraction:.3f}, "
                    f"risk_pct={risk_pct}% → effective={effective_risk_pct:.3f}%"
                )
            else:
                effective_risk_pct = risk_pct

            volume = self._calculate_volume(
                balance=balance,
                risk_pct=effective_risk_pct,
                entry=entry,
                stop_loss=sl,
                contract_size=sym_info["contract_size"],
                volume_min=sym_info["volume_min"],
                volume_max=sym_info["volume_max"],
                volume_step=sym_info["volume_step"],
            )
        if volume <= 0:
            return {"ok": False, "signal_id": signal_id,
                    "error": "Volume is 0 — stop distance too small or requested size below broker minimum"}

        # ── Use current market price for SL/TP if signal prices are stale ────
        current = await self.broker.current_price(symbol, action)
        if current is None:
            return {"ok": False, "signal_id": signal_id,
                    "error": f"Market data unavailable for {symbol}"}

        # Prefer ATR-scaled SL/TP when the signal carries them — the model's
        # regression head learned noisy targets (MAE ~ size of the labels) so
        # the model-predicted SL distance is often inside intraday noise.
        # ATR stops scale with realised volatility and survive normal wiggle.
        atr_sl = signal.get("atr_stop_loss")
        atr_tp = signal.get("atr_take_profit")
        if atr_sl and atr_tp:
            sl_source = "atr"
            entry_distance = abs(entry - atr_sl)
            tp_distance    = abs(atr_tp - entry)
        else:
            sl_source = "model"
            entry_distance = abs(entry - sl)
            tp_distance    = abs(tp - entry) if tp else entry_distance * 2

        if action == "BUY":
            live_sl = round(current - entry_distance, sym_info["digits"])
            live_tp = round(current + tp_distance, sym_info["digits"])
        else:
            live_sl = round(current + entry_distance, sym_info["digits"])
            live_tp = round(current - tp_distance, sym_info["digits"])

        # ── Place order ───────────────────────────────────────────────────────
        logger.info(
            f"📤 Placing {action} {volume} lots of {symbol} | "
            f"current={current:.5f} SL={live_sl:.5f} TP={live_tp:.5f} "
            f"(sl_source={sl_source}) | "
            f"risk={config['risk_per_trade_pct']}% of {balance:.2f}"
        )

        order_result = await self.broker.place_order(
            symbol=symbol,
            side=action,
            volume=volume,
            stop_loss=live_sl,
            take_profit=live_tp,
            comment=f"sig#{signal_id}",
        )

        status = "filled" if order_result["success"] else "error"
        ticket = order_result.get("ticket")

        # ── Persist execution record ──────────────────────────────────────────
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trade_executions (
                    signal_id, user_id, symbol, mt5_ticket, order_type,
                    volume, requested_price, fill_price,
                    stop_loss, take_profit, status,
                    mt5_retcode, mt5_comment
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                """,
                signal_id, user_id, symbol, ticket, action,
                volume, current, order_result.get("fill_price"),
                live_sl, live_tp, status,
                order_result.get("retcode"), order_result.get("comment"),
            )

        if order_result["success"]:
            logger.info(
                f"✅ Order filled: {action} {volume} lots of {symbol} "
                f"@ {order_result.get('fill_price')} | ticket={ticket}"
            )
            return {
                "ok": True,
                "signal_id": signal_id,
                "ticket": ticket,
                "symbol": symbol,
                "side": action,
                "volume": volume,
                "fill_price": order_result.get("fill_price"),
            }
        else:
            logger.error(
                f"❌ Order failed for {symbol}: {order_result.get('comment')} "
                f"(retcode={order_result.get('retcode')})"
            )
            return {
                "ok": False,
                "signal_id": signal_id,
                "error": order_result.get("comment", "Order rejected by broker"),
                "retcode": order_result.get("retcode"),
            }

    # ── Risk helpers ──────────────────────────────────────────────────────────

    def _calculate_volume(
        self,
        balance: float,
        risk_pct: float,
        entry: float,
        stop_loss: float,
        contract_size: float,
        volume_min: float,
        volume_max: float,
        volume_step: float,
        max_position_pct: float = 10.0,
    ) -> float:
        """
        Fixed-fraction position sizing with affordability cap.

          risk_amount   = balance × risk_pct / 100
          stop_distance = |entry - stop_loss|
          risk_volume   = risk_amount / (stop_distance × contract_size)

          affordability cap: position cost ≤ balance × max_position_pct / 100
          affordable_volume = (balance × max_position_pct / 100) / (entry × contract_size)

          volume = min(risk_volume, affordable_volume)

        Prevents tiny stop distances from producing absurdly large lot sizes.
        """
        stop_distance = abs(entry - stop_loss)
        if stop_distance == 0 or contract_size == 0 or entry == 0:
            return 0.0

        risk_amount  = balance * risk_pct / 100.0
        risk_volume  = risk_amount / (stop_distance * contract_size)

        # Cap: total position cost must not exceed max_position_pct% of balance
        max_cost             = balance * max_position_pct / 100.0
        affordable_volume    = max_cost / (entry * contract_size)

        raw_volume = min(risk_volume, affordable_volume)

        # Round down to nearest volume_step
        steps  = int(raw_volume / volume_step)
        volume = round(steps * volume_step, 8)

        return max(volume_min, min(volume, volume_max)) if volume >= volume_min else 0.0

    async def _daily_loss_ok(self, user_id: int, config: dict) -> bool:
        """
        Check whether today's realised losses exceed max_daily_loss_pct of
        the account balance. Returns True if trading is allowed.
        """
        max_loss_pct = config.get("max_daily_loss_pct", 3.0)
        today = date.today()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(pnl), 0) AS daily_pnl
                FROM trade_executions
                WHERE user_id = $1
                  AND status = 'closed'
                  AND DATE(closed_at) = $2
                  AND pnl < 0
                """,
                user_id, today,
            )

        daily_loss = abs(float(row["daily_pnl"])) if row else 0.0

        account = await self.broker.account_info()
        if not account:
            return True  # Can't check — allow trading

        max_loss_amount = account["balance"] * max_loss_pct / 100.0
        if daily_loss >= max_loss_amount:
            logger.warning(
                f"🛑 Circuit breaker: daily loss {daily_loss:.2f} ≥ "
                f"limit {max_loss_amount:.2f} ({max_loss_pct}%) for user {user_id}"
            )
            return False
        return True
