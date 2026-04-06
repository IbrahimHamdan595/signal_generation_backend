import logging
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

US_MARKET_HOLIDAYS = [
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-04",
    "2026-07-03",
    "2026-09-07",
    "2026-10-12",
    "2026-11-26",
    "2026-12-25",
]


def _is_trading_day(dt: datetime) -> bool:
    if dt.weekday() >= 5:
        return False
    date_str = dt.strftime("%Y-%m-%d")
    return date_str not in US_MARKET_HOLIDAYS


def _next_trading_day(dt: datetime) -> datetime:
    next_day = dt + timedelta(days=1)
    while not _is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day


def entry_time_from_bars(
    current_ts: datetime, bars_to_entry: float, interval: str = "1d"
) -> datetime:
    bars_to_entry = max(0.0, min(bars_to_entry, settings.MAX_BARS_TO_ENTRY))

    if bars_to_entry < 0.5:
        logger.debug(f"bars_to_entry={bars_to_entry} < 0.5, using immediate entry")
        return current_ts

    bars_rounded = round(bars_to_entry)

    if bars_rounded == 0:
        return current_ts

    result = current_ts

    if interval == "1d":
        for _ in range(bars_rounded):
            result = _next_trading_day(result)
        result = result.replace(hour=14, minute=30, second=0, microsecond=0)
    elif interval == "1h":
        result = current_ts + timedelta(hours=bars_rounded)
        result = result.replace(minute=0, second=0, microsecond=0)
    else:
        result = current_ts + timedelta(hours=bars_rounded)

    if bars_to_entry != bars_rounded:
        logger.debug(
            f"Fractional bars_to_entry={bars_to_entry} rounded to {bars_rounded}"
        )

    return result


def compute_entry_time(current_ts: datetime, interval: str) -> datetime:
    return entry_time_from_bars(current_ts, 1.0, interval)
