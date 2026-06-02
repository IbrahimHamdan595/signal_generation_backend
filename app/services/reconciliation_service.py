"""
Reconciliation Service
======================
Verifies that the local DB and the MT5 terminal agree about open positions
and recent fills. Mismatches mean either:
  - We placed an order that didn't reach MT5
  - MT5 has a position we never recorded (manual trade or other system)
  - A position was closed by SL/TP but we missed the close event

Run hourly via scheduler. Mismatches log a WARNING and trigger a health
alert via HealthMonitor.alert().
"""

import asyncpg
import logging
from datetime import datetime, timezone

from app.services.broker_service import BrokerService

logger = logging.getLogger(__name__)


class ReconciliationService:
    def __init__(self, pool: asyncpg.Pool, broker: BrokerService):
        self.pool = pool
        self.broker = broker

    async def reconcile(self) -> dict:
        """
        Compare DB trade_executions (status='filled') with MT5 open positions.
        Returns a dict describing any mismatches.
        """
        if not await self.broker.is_connected():
            return {"ok": False, "reason": "MT5 not connected"}

        mt5_positions = await self.broker.get_open_positions()
        mt5_tickets = {int(p["ticket"]) for p in mt5_positions}

        async with self.pool.acquire() as conn:
            db_rows = await conn.fetch(
                """
                SELECT mt5_ticket, symbol, volume, fill_price, user_id
                FROM trade_executions
                WHERE status = 'filled' AND mt5_ticket IS NOT NULL
                """
            )
        db_tickets = {int(r["mt5_ticket"]) for r in db_rows}

        # Mismatches in either direction
        only_in_mt5 = mt5_tickets - db_tickets   # positions DB doesn't know about
        only_in_db  = db_tickets - mt5_tickets   # DB thinks open but MT5 closed

        # For only_in_db: likely SL/TP hit, MT5 closed, we missed it. Mark closed.
        closed_now = []
        if only_in_db:
            async with self.pool.acquire() as conn:
                for ticket in only_in_db:
                    # Look up MT5 deal history to recover close price + pnl
                    pnl = await self._lookup_closed_pnl(ticket)
                    await conn.execute(
                        """
                        UPDATE trade_executions
                        SET status = 'closed',
                            pnl    = COALESCE($1, pnl),
                            closed_at = COALESCE(closed_at, NOW())
                        WHERE mt5_ticket = $2
                        """,
                        pnl, ticket,
                    )
                    closed_now.append({"ticket": ticket, "recovered_pnl": pnl})

        result = {
            "ok": len(only_in_mt5) == 0 and len(only_in_db) == 0,
            "checked_at":  datetime.now(timezone.utc).isoformat(),
            "mt5_open":    len(mt5_tickets),
            "db_open":     len(db_tickets),
            "only_in_mt5": list(only_in_mt5),
            "only_in_db":  list(only_in_db),
            "closed_now":  closed_now,
        }

        if not result["ok"]:
            logger.warning(
                f"⚠️  Reconciliation mismatch: "
                f"{len(only_in_mt5)} unknown MT5 positions, "
                f"{len(only_in_db)} stale DB positions (auto-closed {len(closed_now)})"
            )
        else:
            logger.info(
                f"✅ Reconciliation OK — {len(mt5_tickets)} positions in sync"
            )

        return result

    async def _lookup_closed_pnl(self, ticket: int) -> float | None:
        """Pull the closed-deal PnL from MT5 history for a given position ticket.

        Sums profit + swap + commission across ALL deals for the position so
        partial fills and multi-leg closes resolve correctly. 90-day lookback
        is generous — MT5 keeps history far longer; tighter windows were
        missing positions closed >1 week ago and leaving pnl=NULL.
        """
        from app.services.broker_service import _import_mt5, _run

        def _hist():
            mt5 = _import_mt5()
            from datetime import datetime as _dt, timedelta
            deals = mt5.history_deals_get(_dt.now() - timedelta(days=90), _dt.now())
            if not deals:
                logger.warning(f"⚠️  mt5.history_deals_get returned empty for ticket {ticket}")
                return None
            total = 0.0
            found = False
            t = int(ticket)
            # We store result.order as mt5_ticket, but the deal record may match
            # via position_id, order, or ticket depending on broker conventions.
            # Try all three so we don't miss legitimate matches.
            for d in deals:
                pid = int(getattr(d, "position_id", 0) or 0)
                oid = int(getattr(d, "order",       0) or 0)
                did = int(getattr(d, "ticket",      0) or 0)
                if pid == t or oid == t or did == t:
                    found = True
                    total += float(d.profit) \
                           + float(getattr(d, "swap",       0) or 0) \
                           + float(getattr(d, "commission", 0) or 0)
            if not found:
                logger.warning(
                    f"⚠️  Ticket {ticket} not found in {len(deals)} MT5 deals "
                    f"(checked position_id, order, ticket)"
                )
            return total if found else None

        try:
            pnl = await _run(_hist)
            if pnl is None:
                logger.warning(f"⚠️  No MT5 deal history found for ticket {ticket} — pnl will stay NULL")
            return pnl
        except Exception as e:
            logger.warning(f"Could not look up history for ticket {ticket}: {e}")
            return None
