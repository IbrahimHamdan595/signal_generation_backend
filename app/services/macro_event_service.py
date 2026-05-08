"""MacroEventService — fetches and stores FOMC / CPI / NFP event dates.

Sources (in priority order):
  FOMC  — federalreserve.gov calendar HTML (no key needed)
  CPI   — FRED API release dates (FRED_API_KEY) → BLS website scrape fallback
  NFP   — FRED API release dates (FRED_API_KEY) → algorithmic fallback

Call `ingest_macro_events(start_year, end_year)` once after running migrations.
Re-running is safe (ON CONFLICT DO NOTHING).
"""

import asyncpg
import httpx
import logging
import re
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_nfp_dates_algorithmic(start_year: int, end_year: int) -> List[date]:
    """BLS Employment Situation always releases on the first Friday ≥ 3rd of month."""
    results: List[date] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = date(year, month, 1)
            days_to_friday = (4 - d.weekday()) % 7
            first_friday = d + timedelta(days=days_to_friday)
            if first_friday.day < 3:
                first_friday += timedelta(weeks=1)
            results.append(first_friday)
    return results


async def _fetch_fomc_from_fed(start_year: int, end_year: int) -> List[date]:
    """Parse FOMC decision dates from the Federal Reserve calendar HTML page.

    The page lists each meeting as "Month D1[-D2], YEAR" where the last day is
    the decision/announcement day. Emergency single-day meetings also appear.
    """
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "signal-bot/1.0"})
            resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning(f"⚠️  FOMC calendar fetch failed: {exc}")
        return []

    results: set[date] = set()

    # Two-day meeting: "Month DD-DD, YYYY" — decision on second day
    rx_range = re.compile(
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+(\d{1,2})[–—\-](\d{1,2})[*†]?,?\s+(\d{4})",
        re.IGNORECASE,
    )
    # Single-day meeting: "Month DD, YYYY" (emergency cuts etc.)
    rx_single = re.compile(
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+(\d{1,2})[*†]?,\s+(\d{4})",
        re.IGNORECASE,
    )

    range_spans: List[tuple] = []
    for m in rx_range.finditer(html):
        month_num = _MONTH_MAP[m.group(1).lower()]
        day2 = int(m.group(3))
        year = int(m.group(4))
        if start_year <= year <= end_year:
            try:
                results.add(date(year, month_num, day2))
                range_spans.append((m.start(), m.end()))
            except ValueError:
                pass

    for m in rx_single.finditer(html):
        # Skip positions already consumed by range pattern
        if any(s <= m.start() < e for s, e in range_spans):
            continue
        month_num = _MONTH_MAP[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3))
        if start_year <= year <= end_year:
            try:
                results.add(date(year, month_num, day))
            except ValueError:
                pass

    dates = sorted(results)
    logger.info(f"✅ FOMC: scraped {len(dates)} meeting dates from federalreserve.gov")
    return dates


async def _fetch_fred_release_dates(release_id: int, start_year: int, end_year: int) -> List[date]:
    """Fetch BLS/Fed release dates from FRED API.

    release_id=10  → Consumer Price Index (CPI)
    release_id=50  → Employment Situation (NFP)
    """
    api_key = settings.FRED_API_KEY
    if not api_key:
        return []

    url = "https://api.stlouisfed.org/fred/release/dates"
    params = {
        "release_id": release_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": f"{start_year}-01-01",
        "realtime_end": f"{end_year}-12-31",
        "include_release_dates_with_no_data": "true",
        "limit": 1000,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"⚠️  FRED release/{release_id} fetch failed: {exc}")
        return []

    dates: List[date] = []
    for item in data.get("release_dates", []):
        try:
            dates.append(date.fromisoformat(item["date"]))
        except (KeyError, ValueError):
            pass

    logger.info(f"✅ FRED release {release_id}: {len(dates)} dates")
    return sorted(dates)


async def _fetch_bls_schedule(url: str, start_year: int, end_year: int) -> List[date]:
    """Scrape a BLS news-release schedule page for dates in "Month DD, YYYY" format."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "signal-bot/1.0"})
            resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning(f"⚠️  BLS schedule scrape failed ({url}): {exc}")
        return []

    rx = re.compile(
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{1,2}),\s+(\d{4})",
        re.IGNORECASE,
    )
    results: set[date] = set()
    for m in rx.finditer(html):
        month_num = _MONTH_MAP[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3))
        if start_year <= year <= end_year:
            try:
                results.add(date(year, month_num, day))
            except ValueError:
                pass

    dates = sorted(results)
    logger.info(f"✅ BLS scrape {url}: {len(dates)} dates")
    return dates


# ── Public helpers (used by ohlcv_service) ───────────────────────────────────

def days_to_next_event(bar_dates: pd.Series, event_dates: List[str], neutral: float = 30.0) -> np.ndarray:
    """Vectorized: for each bar date return days until the next event.

    Uses binary search — O(n log m). Returns `neutral` when no future event exists.
    event_dates: list of ISO date strings "YYYY-MM-DD" (sorted or unsorted).
    """
    if not event_dates:
        return np.full(len(bar_dates), neutral)

    event_arr = np.array(
        [np.datetime64(d, "D") for d in sorted(event_dates)], dtype="datetime64[D]"
    )
    bar_arr = bar_dates.values.astype("datetime64[D]")
    result = np.full(len(bar_arr), neutral)

    for i, bd in enumerate(bar_arr):
        idx = int(np.searchsorted(event_arr, bd, side="left"))
        if idx < len(event_arr):
            result[i] = float((event_arr[idx] - bd).astype(int))

    return result


# ── Service class ─────────────────────────────────────────────────────────────

class MacroEventService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def ingest_macro_events(
        self,
        start_year: int = 2015,
        end_year: Optional[int] = None,
    ) -> dict:
        """Fetch FOMC/CPI/NFP dates and upsert into macro_events table.

        Safe to re-run — duplicates are silently ignored.
        """
        if end_year is None:
            end_year = date.today().year + 1

        # ── FOMC ──────────────────────────────────────────────────────────────
        fomc_dates = await _fetch_fomc_from_fed(start_year, end_year)
        if not fomc_dates:
            logger.warning("⚠️  FOMC scrape returned 0 dates — check Fed website structure")

        # ── CPI ───────────────────────────────────────────────────────────────
        cpi_dates = await _fetch_fred_release_dates(10, start_year, end_year)
        if not cpi_dates:
            logger.info("ℹ️  FRED key missing or failed — scraping BLS for CPI dates")
            cpi_dates = await _fetch_bls_schedule(
                "https://www.bls.gov/schedule/news_release/cpi.htm",
                start_year, end_year,
            )

        # ── NFP ───────────────────────────────────────────────────────────────
        nfp_dates = await _fetch_fred_release_dates(50, start_year, end_year)
        if not nfp_dates:
            logger.info("ℹ️  FRED key missing or failed — computing NFP dates algorithmically")
            nfp_dates = _compute_nfp_dates_algorithmic(start_year, end_year)

        records = (
            [("fomc", d) for d in fomc_dates] +
            [("cpi",  d) for d in cpi_dates]  +
            [("nfp",  d) for d in nfp_dates]
        )

        if records:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO macro_events (event_type, event_date)
                    VALUES ($1, $2)
                    ON CONFLICT (event_type, event_date) DO NOTHING
                    """,
                    records,
                )

        counts = {
            "fomc": len(fomc_dates),
            "cpi":  len(cpi_dates),
            "nfp":  len(nfp_dates),
            "total": len(records),
        }
        logger.info(f"✅ macro_events stored: {counts}")
        return counts

    async def get_all_dates(self) -> Dict[str, List[str]]:
        """Return {event_type: [iso_date_str, ...]} for all stored events."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT event_type, event_date FROM macro_events ORDER BY event_date ASC"
            )
        result: Dict[str, List[str]] = {"fomc": [], "cpi": [], "nfp": []}
        for r in rows:
            et = r["event_type"]
            if et in result:
                result[et].append(r["event_date"].isoformat())
        return result
