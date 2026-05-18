"""Asset-class detection and FX ticker registry.

Loads the FX/metal pair definitions from `backend/data/fx_majors.json` at import
time and exposes lookup helpers used across the ingest, training, signal-
generation and execution layers to route tickers down the right pipeline.

Equity tickers are anything not in this registry — the system never enumerates
them up front (the S&P list lives in `data/sp500.json` and changes over time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# Resolve `backend/data/fx_majors.json` from this file's location so the import
# works regardless of the current working directory the server was launched
# from (uvicorn from project root vs. backend/ vs. an absolute path).
_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "fx_majors.json"


def _load_registry() -> dict:
    """Parse fx_majors.json and build the in-memory lookup tables."""
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = raw.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(
            f"fx_majors.json at {_DATA_PATH} has no `pairs` array — file is malformed"
        )

    fx_tickers: set[str] = set()
    yfinance_map: dict[str, Optional[str]] = {}
    class_map: dict[str, str] = {}
    name_map: dict[str, str] = {}

    for i, entry in enumerate(pairs):
        mt5 = entry.get("mt5_symbol")
        if not mt5:
            raise ValueError(f"fx_majors.json entry {i} missing `mt5_symbol`")
        cls = entry.get("asset_class")
        if cls not in ("fx_major", "fx_metal"):
            raise ValueError(
                f"fx_majors.json entry {mt5} has invalid asset_class={cls!r} "
                f"(must be 'fx_major' or 'fx_metal')"
            )
        sym = mt5.upper()
        fx_tickers.add(sym)
        yfinance_map[sym] = entry.get("yfinance_symbol")  # may be None
        class_map[sym]    = cls
        name_map[sym]     = entry.get("name", sym)

    return {
        "fx_tickers":   fx_tickers,
        "yfinance_map": yfinance_map,
        "class_map":    class_map,
        "name_map":     name_map,
    }


_REGISTRY = _load_registry()

# Public read-only views — capitalised constants so callers can `from
# asset_class import FX_TICKERS` for set membership checks without going
# through a function call hot-path.
FX_TICKERS:   frozenset[str]            = frozenset(_REGISTRY["fx_tickers"])
YFINANCE_MAP: dict[str, Optional[str]]  = dict(_REGISTRY["yfinance_map"])


def is_fx(ticker: str) -> bool:
    """True iff `ticker` is one of the configured FX/metal pairs."""
    return ticker.upper() in FX_TICKERS


def asset_class_for(ticker: str) -> str:
    """Returns 'equity', 'fx_major', or 'fx_metal' for routing decisions."""
    sym = ticker.upper()
    return _REGISTRY["class_map"].get(sym, "equity")


def yfinance_symbol_for(ticker: str) -> Optional[str]:
    """Mapped yfinance ticker for an FX/metal pair, or None when the registry
    says yfinance has no coverage (caller should fall back to MT5 ingest).
    Returns None for unknown / equity tickers — callers shouldn't be asking
    about those here."""
    return _REGISTRY["yfinance_map"].get(ticker.upper())


def fx_pair_count() -> int:
    """Diagnostic: number of FX pairs configured. Useful for startup logs."""
    return len(FX_TICKERS)
