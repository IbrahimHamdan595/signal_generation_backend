"""
WebSocket — Live Price Streaming
==================================
Clients connect to /ws/prices?tickers=AAPL,MSFT and receive a JSON
message every PUSH_INTERVAL seconds with the latest close price for
each requested ticker.

Message format:
  { "type": "prices", "data": { "AAPL": 183.45, "MSFT": 412.10 }, "ts": "..." }

The server reads from the ohlcv_data table (latest bar per ticker).
No external API calls on each push — all data is already in the DB.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.db.database import get_db

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)

PUSH_INTERVAL = 10          # seconds between price updates
MAX_TICKERS   = 20          # prevent abuse


@router.websocket("/ws/prices")
async def price_stream(
    websocket: WebSocket,
    tickers: str = Query(..., description="Comma-separated tickers e.g. AAPL,MSFT"),
):
    await websocket.accept()

    requested = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    requested = requested[:MAX_TICKERS]

    if not requested:
        await websocket.close(code=1003, reason="No valid tickers provided")
        return

    logger.info(f"🔌 WS connected — tickers: {requested}")

    try:
        while True:
            pool = await get_db()
            if pool is None:
                await asyncio.sleep(PUSH_INTERVAL)
                continue

            prices: dict[str, float | None] = {}
            changes: dict[str, float | None] = {}

            async with pool.acquire() as conn:
                for ticker in requested:
                    rows = await conn.fetch("""
                        SELECT close, timestamp
                        FROM ohlcv_data
                        WHERE ticker = $1 AND interval = '1d'
                        ORDER BY timestamp DESC
                        LIMIT 2
                    """, ticker)

                    if rows:
                        prices[ticker]  = float(rows[0]["close"])
                        prev = float(rows[1]["close"]) if len(rows) > 1 else None
                        if prev:
                            changes[ticker] = round((prices[ticker] - prev) / prev, 6)
                        else:
                            changes[ticker] = None
                    else:
                        prices[ticker]  = None
                        changes[ticker] = None

            payload = {
                "type":    "prices",
                "data":    prices,
                "changes": changes,
                "ts":      datetime.now(timezone.utc).isoformat(),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(PUSH_INTERVAL)

    except WebSocketDisconnect:
        logger.info(f"🔌 WS disconnected — {requested}")
    except Exception as e:
        logger.error(f"❌ WS error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
