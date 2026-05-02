import logging
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

# Prefer pandas_market_calendars (or exchange_calendars) for an authoritative
# NYSE schedule that doesn't go stale. Fall back to a multi-year static list
# if neither library is installed — covers 2025–2030 so this code keeps
# working until the next maintenance window.
_NYSE_CALENDAR = None
try:
    import pandas_market_calendars as mcal  # type: ignore
    _NYSE_CALENDAR = mcal.get_calendar("NYSE")
    logger.info("✅ entry_time: using pandas_market_calendars for NYSE holidays")
except Exception:
    try:
        import exchange_calendars as xcals  # type: ignore
        _NYSE_CALENDAR = xcals.get_calendar("XNYS")
        logger.info("✅ entry_time: using exchange_calendars for NYSE holidays")
    except Exception:
        _NYSE_CALENDAR = None
        logger.info("ℹ️  entry_time: market-calendar libs unavailable — using static list")

# Static fallback — extend as needed. Includes observed-day shifts for fixed
# holidays that fall on weekends.
US_MARKET_HOLIDAYS = {
    # 2025
    "2025-01-01", "2025-01-09",  # Carter day of mourning
    "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    # 2028
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19",
    "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25",
    # 2029
    "2029-01-01", "2029-01-15", "2029-02-19", "2029-03-30", "2029-05-28",
    "2029-06-19", "2029-07-04", "2029-09-03", "2029-11-22", "2029-12-25",
    # 2030
    "2030-01-01", "2030-01-21", "2030-02-18", "2030-04-19", "2030-05-27",
    "2030-06-19", "2030-07-04", "2030-09-02", "2030-11-28", "2030-12-25",
}


def _is_trading_day(dt: datetime) -> bool:
    if dt.weekday() >= 5:
        return False
    if _NYSE_CALENDAR is not None:
        try:
            ts = dt.replace(tzinfo=None)
            schedule = _NYSE_CALENDAR.valid_days(start_date=ts, end_date=ts)
            return len(schedule) > 0
        except Exception:
            pass  # fall through to static list
    return dt.strftime("%Y-%m-%d") not in US_MARKET_HOLIDAYS


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
