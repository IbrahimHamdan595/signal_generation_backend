import asyncio
import asyncpg
import json
import os
from datetime import datetime, timezone
from typing import List, Optional
import logging

_SIGNAL_PROBS_PATH = "checkpoints/recent_signal_probs.json"
_MAX_RECENT = 200   # keep last 200 predictions for drift checks

def _record_signal_probs(probs: dict) -> None:
    """Append prediction probabilities to the drift monitoring log."""
    try:
        log = []
        if os.path.exists(_SIGNAL_PROBS_PATH):
            with open(_SIGNAL_PROBS_PATH) as f:
                log = json.load(f)
        log.append({"hold": probs.get("hold", 0), "buy": probs.get("buy", 0), "sell": probs.get("sell", 0)})
        log = log[-_MAX_RECENT:]
        os.makedirs(os.path.dirname(_SIGNAL_PROBS_PATH), exist_ok=True)
        with open(_SIGNAL_PROBS_PATH, "w") as f:
            json.dump(log, f)
    except Exception:
        pass  # drift logging is non-critical

from app.services.ml_service import MLService
from app.services.alert_service import AlertService
from app.ml.models.registry import is_model_trained, get_model
from app.core.asset_class import is_fx as _is_fx_ticker

logger = logging.getLogger(__name__)


class SignalService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        asset_class_override: Optional[str] = None,
        source_override: Optional[str] = None,
    ):
        """
        `asset_class_override` (e.g. "equities_1h", "fx_1h") forces the
        ML inference to use a specific checkpoint folder regardless of the
        ticker's natural asset class. Used by the 1h pipelines to route to
        their own trained checkpoints.

        `source_override` (e.g. "ml_equities_1h") overrides the `source` tag
        written to the signals table. When None, the natural per-ticker
        source ("ml_equities" / "ml_fx") is used.
        """
        self.pool = pool
        self.ml_svc = MLService(pool)
        self._ac_override     = asset_class_override
        self._source_override = source_override

    def _effective_asset_class(self, ticker: str) -> str:
        """The checkpoint folder this service should hit for `ticker`."""
        if self._ac_override:
            return self._ac_override
        return "fx" if _is_fx_ticker(ticker) else "equities"

    def _effective_source_tag(self, ticker: str) -> str:
        """The `source` column value to write to signals."""
        if self._source_override:
            return self._source_override
        return "ml_fx" if _is_fx_ticker(ticker) else "ml_equities"

    async def generate_and_store(
        self, ticker: str, interval: str = "1d"
    ) -> Optional[dict]:
        ticker = ticker.upper()

        # Class-aware trained check — ml_fx_1h shouldn't be blocked when
        # equities isn't trained (and vice versa).
        ac = self._effective_asset_class(ticker)
        if not is_model_trained(ac):
            logger.warning(f"⚠️  Signal requested but {ac} model not trained yet.")
            return None

        result = await self.ml_svc.predict_ticker(
            ticker, interval, asset_class_override=self._ac_override
        )

        if "error" in result:
            logger.error(f"Signal error for {ticker}: {result['error']}")
            return None

        entry_time = None
        if result.get("entry_time"):
            try:
                entry_time = datetime.fromisoformat(result["entry_time"])
            except ValueError:
                logger.warning(f"[{ticker}] Could not parse entry_time: {result['entry_time']!r}")

        predicted  = result.get("predicted", {})
        timing     = result.get("timing", {})
        bucket     = timing.get("bucket")
        bucket_label = timing.get("bucket_label")

        # Profit-engineering fields populated by predict_ticker (filter gates,
        # EV, Kelly sizing, event proximity, ATR-based levels). The execution
        # layer reads kelly_fraction / expected_value from these columns.
        atr_levels   = result.get("atr_levels", {}) or {}
        event_prox   = result.get("event_proximity", {}) or {}
        reject_json  = json.dumps(result.get("reject_reasons", []))

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals (
                    ticker, interval, action, confidence,
                    entry_price, stop_loss, take_profit, net_profit,
                    bars_to_entry, entry_time, entry_time_label,
                    prob_buy, prob_sell, prob_hold,
                    entry_offset_pct, timing_bucket, timing_bucket_label,
                    uncertainty, predicted_rr, expected_value,
                    kelly_full, kelly_fraction, reject_reasons,
                    atr_stop_loss, atr_take_profit,
                    fomc_days, cpi_days, nfp_days, earnings_days,
                    source, created_at
                ) VALUES (
                    $1,  $2,  $3,  $4,  $5,  $6,  $7,  $8,  $9,  $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23::jsonb, $24, $25, $26, $27, $28, $29, $30, $31
                )
                ON CONFLICT (ticker, interval, DATE(created_at AT TIME ZONE 'UTC'), COALESCE(source, ''))
                DO UPDATE SET
                    action              = EXCLUDED.action,
                    confidence          = EXCLUDED.confidence,
                    entry_price         = EXCLUDED.entry_price,
                    stop_loss           = EXCLUDED.stop_loss,
                    take_profit         = EXCLUDED.take_profit,
                    net_profit          = EXCLUDED.net_profit,
                    bars_to_entry       = EXCLUDED.bars_to_entry,
                    entry_time          = EXCLUDED.entry_time,
                    entry_time_label    = EXCLUDED.entry_time_label,
                    prob_buy            = EXCLUDED.prob_buy,
                    prob_sell           = EXCLUDED.prob_sell,
                    prob_hold           = EXCLUDED.prob_hold,
                    entry_offset_pct    = EXCLUDED.entry_offset_pct,
                    timing_bucket       = EXCLUDED.timing_bucket,
                    timing_bucket_label = EXCLUDED.timing_bucket_label,
                    uncertainty         = EXCLUDED.uncertainty,
                    predicted_rr        = EXCLUDED.predicted_rr,
                    expected_value      = EXCLUDED.expected_value,
                    kelly_full          = EXCLUDED.kelly_full,
                    kelly_fraction      = EXCLUDED.kelly_fraction,
                    reject_reasons      = EXCLUDED.reject_reasons,
                    atr_stop_loss       = EXCLUDED.atr_stop_loss,
                    atr_take_profit     = EXCLUDED.atr_take_profit,
                    fomc_days           = EXCLUDED.fomc_days,
                    cpi_days            = EXCLUDED.cpi_days,
                    nfp_days            = EXCLUDED.nfp_days,
                    earnings_days       = EXCLUDED.earnings_days,
                    created_at          = EXCLUDED.created_at
                RETURNING *
                """,
                ticker,
                interval,
                result["action"],
                result["confidence"],
                result.get("entry_price"),
                result.get("stop_loss"),
                result.get("take_profit"),
                result.get("net_profit"),
                result.get("bars_to_entry"),
                entry_time,
                result.get("entry_time_label"),
                result.get("probabilities", {}).get("buy"),
                result.get("probabilities", {}).get("sell"),
                result.get("probabilities", {}).get("hold"),
                predicted.get("entry_offset_pct"),
                bucket,
                bucket_label,
                result.get("uncertainty"),
                result.get("predicted_rr"),
                result.get("expected_value"),
                result.get("kelly_full"),
                result.get("kelly_fraction"),
                reject_json,
                atr_levels.get("stop_loss"),
                atr_levels.get("take_profit"),
                event_prox.get("fomc_days"),
                event_prox.get("cpi_days"),
                event_prox.get("nfp_days"),
                event_prox.get("earnings_days"),
                self._effective_source_tag(ticker),
                datetime.now(timezone.utc),
            )

            doc = dict(row)
            logger.info(
                f"✅ Signal stored: {ticker} → {result['action']} @ {result.get('entry_time')}"
            )
            _record_signal_probs(result.get("probabilities", {}))

        # Fire alert for high-confidence directional signals
        try:
            alert_svc = AlertService(self.pool)
            await alert_svc.maybe_create(
                ticker=ticker,
                action=result["action"],
                confidence=result["confidence"],
                signal_id=doc["id"],
            )
        except Exception as e:
            logger.warning(f"⚠️  Alert creation failed for {ticker}: {e}")

        return doc

    async def generate_signal(self, ticker: str, interval: str = "1d") -> dict:
        ticker = ticker.upper()
        model = get_model()

        if model is None:
            return {
                "ticker": ticker,
                "interval": interval,
                "action": "HOLD",
                "confidence": 0.0,
                "source": "no_model",
            }

        result = await self.ml_svc.predict_ticker(ticker, interval)

        if "error" in result:
            return {
                "ticker": ticker,
                "interval": interval,
                "action": "HOLD",
                "confidence": 0.0,
                "source": "error",
                "error": result["error"],
            }

        return {
            "ticker": ticker,
            "interval": interval,
            "action": result.get("action", "HOLD"),
            "confidence": result.get("confidence", 0.0),
            "source": result.get("source") or ("ml_fx" if _is_fx_ticker(ticker) else "ml_equities"),
            **{
                k: result[k]
                for k in (
                    "entry_price",
                    "stop_loss",
                    "take_profit",
                    "net_profit",
                    "bars_to_entry",
                    "entry_time",
                )
                if k in result
            },
        }

    async def generate_batch(
        self, tickers: List[str], interval: str = "1d"
    ) -> List[dict]:
        tasks = [self.generate_and_store(ticker, interval) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def get_latest(self, ticker: str, interval: str = "1d") -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM signals
                WHERE ticker = $1 AND interval = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                ticker.upper(),
                interval,
            )
            return dict(row) if row else None

    async def get_history(
        self, ticker: str, interval: str = "1d", limit: int = 50
    ) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM signals
                WHERE ticker = $1 AND interval = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                ticker.upper(),
                interval,
                limit,
            )
            return [dict(r) for r in rows]

    async def get_all_latest(self) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (ticker) * FROM signals
                ORDER BY ticker, created_at DESC
                """
            )
            return [dict(r) for r in rows]

    async def get_by_action(
        self, action: str, interval: str = "1d", limit: int = 50
    ) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM signals
                WHERE action = $1 AND interval = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                action.upper(),
                interval,
                limit,
            )
            return [dict(r) for r in rows]

    async def get_high_confidence(
        self, min_confidence: float = 0.75, interval: str = "1d", limit: int = 20
    ) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM signals
                WHERE confidence >= $1 AND interval = $2
                AND action IN ('BUY', 'SELL')
                ORDER BY created_at DESC
                LIMIT $3
                """,
                min_confidence,
                interval,
                limit,
            )
            return [dict(r) for r in rows]
