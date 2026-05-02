"""
Broker Service — MetaTrader 5
==============================

Thin async wrapper around the synchronous MetaTrader5 Python library.

All MT5 calls are dispatched through a single-threaded ThreadPoolExecutor
(_MT5_EXECUTOR) because the MT5 library maintains process-level state and
is not safe to call from multiple threads concurrently.

Usage
-----
    broker = BrokerService()
    ok = await broker.connect(account=12345678, password="...", server="Demo-Server")
    if ok:
        info = await broker.account_info()
        ticket = await broker.place_order(...)
    await broker.disconnect()

MT5 terminal must be installed and running on the same Windows machine.
Download: https://www.metatrader5.com/en/download
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Single-threaded executor — every MT5 call serialised through this
_MT5_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")

# MT5 magic number — identifies orders placed by this application
_MAGIC = 20250001

# Filled lazily on first import attempt so missing MetaTrader5 package
# doesn't crash the whole server if trading is not configured.
_mt5 = None


def _import_mt5():
    global _mt5
    if _mt5 is None:
        try:
            import MetaTrader5 as mt5  # noqa: N812
            _mt5 = mt5
        except ImportError:
            raise RuntimeError(
                "MetaTrader5 package not installed. "
                "Run: pip install MetaTrader5"
            )
    return _mt5


def _run_sync(fn, *args):
    """Run a plain callable in the MT5 executor (called from sync context)."""
    mt5 = _import_mt5()
    return fn(mt5, *args)


async def _run(fn, *args):
    """Dispatch a lambda into the MT5 executor from async context."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_MT5_EXECUTOR, fn, *args)


class BrokerService:
    """
    Async-friendly MetaTrader 5 broker interface.

    MT5 state is process-level (one terminal connection per Python process),
    so multiple BrokerService instances all share the same underlying
    connection — instantiating more than one is harmless.
    """

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(
        self,
        account: int,
        password: str,
        server: str,
        path: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Connect to the running MT5 terminal and log in to the given account.

        Step 1: mt5.initialize() — attaches to the already-running terminal
                process (no credentials needed, terminal must be open).
        Step 2: mt5.login()     — switches to / authenticates the account.

        Returns (success, error_message).
        path is only needed if you want the library to launch the terminal
        from scratch rather than attach to a running instance.
        """
        def _connect():
            mt5 = _import_mt5()

            # Step 1: attach to the running terminal (no credentials)
            init_kwargs = {}
            if path:
                init_kwargs["path"] = path

            if not mt5.initialize(**init_kwargs):
                err = mt5.last_error()
                logger.error(f"MT5 initialize failed: {err}")
                return False, (
                    f"Cannot attach to MT5 terminal (code {err[0]}): {err[1]}. "
                    "Make sure MetaTrader 5 is open and fully loaded."
                )

            # Step 2: log in to the account
            if not mt5.login(account, password=password, server=server):
                err = mt5.last_error()
                logger.error(f"MT5 login failed: {err}")
                mt5.shutdown()
                return False, (
                    f"Login rejected (code {err[0]}): {err[1]}. "
                    "Check account number, password, and server name."
                )

            info = mt5.account_info()
            if info is None:
                err = mt5.last_error()
                mt5.shutdown()
                return False, f"Connected but could not read account info: {err}"

            logger.info(
                f"✅ MT5 connected — account #{info.login} | "
                f"server={info.server} | "
                f"balance={info.balance:.2f} {info.currency} | "
                f"{'DEMO' if info.trade_mode == 0 else 'LIVE'}"
            )
            return True, None

        return await _run(_connect)

    async def disconnect(self):
        def _shutdown():
            mt5 = _import_mt5()
            mt5.shutdown()
            logger.info("MT5 connection closed")

        await _run(_shutdown)

    async def is_connected(self) -> bool:
        def _check():
            try:
                mt5 = _import_mt5()
                return mt5.terminal_info() is not None
            except Exception:
                return False

        return await _run(_check)

    # ── Account ───────────────────────────────────────────────────────────────

    async def account_info(self) -> Optional[dict]:
        """Return key account metrics as a plain dict."""
        def _info():
            mt5 = _import_mt5()
            info = mt5.account_info()
            if info is None:
                return None
            return {
                "login":        info.login,
                "server":       info.server,
                "currency":     info.currency,
                "balance":      info.balance,
                "equity":       info.equity,
                "margin":       info.margin,
                "free_margin":  info.margin_free,
                "margin_level": info.margin_level,
                "profit":       info.profit,
                "leverage":     info.leverage,
                "is_demo":      info.trade_mode == 0,
            }

        return await _run(_info)

    # ── Symbol helpers ────────────────────────────────────────────────────────

    async def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Return contract spec needed for position sizing.
        Tries exact match first, then searches broker catalog by ticker pattern
        so brokers that use full names (e.g. 'Apple' for 'AAPL') still work
        for any ticker without a hardcoded mapping.
        """
        def _sym():
            mt5 = _import_mt5()

            def _build(info):
                mt5.symbol_select(info.name, True)
                return {
                    "symbol":          info.name,
                    "contract_size":   info.trade_contract_size,
                    "volume_min":      info.volume_min,
                    "volume_max":      info.volume_max,
                    "volume_step":     info.volume_step,
                    "point":           info.point,
                    "digits":          info.digits,
                    "currency_base":   info.currency_base,
                    "currency_profit": info.currency_profit,
                }

            # 1. Exact match
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
            if info is not None:
                return _build(info)

            # 2. Search broker catalog — ticker appears in category field
            #    e.g. broker stores 'Apple' with category 'AAPL.OQ'
            all_symbols = mt5.symbols_get()
            if all_symbols:
                ticker_upper = symbol.upper()
                for s in all_symbols:
                    category = (s.category or "").upper()
                    name     = (s.name or "").upper()
                    if category.startswith(ticker_upper) or name == ticker_upper:
                        return _build(s)

            return None

        return await _run(_sym)

    async def validate_symbols(
        self, tickers: list[str], suffix: str = ""
    ) -> tuple[list[str], list[str]]:
        """
        Check which tickers exist in MT5.
        Returns (valid_tickers, invalid_tickers).
        valid_tickers are the original ticker strings that resolved successfully.
        """
        def _validate():
            mt5 = _import_mt5()
            all_symbols = mt5.symbols_get()
            symbol_names  = set()
            symbol_cats   = {}
            if all_symbols:
                for s in all_symbols:
                    symbol_names.add((s.name or "").upper())
                    cat = (s.category or "").upper()
                    if cat:
                        symbol_cats[cat] = s.name

            valid, invalid = [], []
            for ticker in tickers:
                candidate = (ticker + suffix).upper()
                # 1. Exact match (with suffix)
                if candidate in symbol_names:
                    valid.append(ticker)
                    continue
                # 2. Exact match (without suffix)
                if ticker.upper() in symbol_names:
                    valid.append(ticker)
                    continue
                # 3. Category match (e.g. AAPL.OQ → Apple)
                found = False
                for cat in symbol_cats:
                    if cat.startswith(ticker.upper()):
                        valid.append(ticker)
                        found = True
                        break
                if not found:
                    invalid.append(ticker)
            return valid, invalid

        return await _run(_validate)

    async def current_price(self, symbol: str, side: str) -> Optional[float]:
        """
        Return the current ask (for BUY) or bid (for SELL).
        side: 'BUY' | 'SELL'
        """
        def _price():
            mt5 = _import_mt5()
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            price = tick.ask if side == "BUY" else tick.bid
            return price if price > 0 else None

        return await _run(_price)

    # ── Order placement ───────────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,          # 'BUY' | 'SELL'
        volume: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "",
    ) -> dict:
        """
        Place a market order. Returns a result dict with keys:
          ticket, retcode, comment, fill_price, volume, success
        """
        def _place():
            mt5 = _import_mt5()

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {
                    "success": False,
                    "comment": f"Symbol {symbol} not found or market closed",
                    "retcode": -1,
                }

            order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if side == "BUY" else tick.bid

            request = {
                "action":      mt5.TRADE_ACTION_DEAL,
                "symbol":      symbol,
                "volume":      float(volume),
                "type":        order_type,
                "price":       price,
                "sl":          float(stop_loss),
                "tp":          float(take_profit),
                "deviation":   20,          # max slippage in points
                "magic":       _MAGIC,
                "comment":     comment[:31] if comment else "",
                "type_time":   mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                return {"success": False, "comment": str(err), "retcode": -1}

            success = result.retcode == mt5.TRADE_RETCODE_DONE
            if not success:
                logger.warning(
                    f"Order rejected for {symbol}: retcode={result.retcode} "
                    f"comment={result.comment}"
                )
            return {
                "success":    success,
                "ticket":     result.order,
                "retcode":    result.retcode,
                "comment":    result.comment,
                "fill_price": result.price,
                "volume":     result.volume,
            }

        return await _run(_place)

    # ── Position management ───────────────────────────────────────────────────

    async def get_open_positions(self) -> list[dict]:
        """Return all open positions placed by this application (by magic number)."""
        def _positions():
            mt5 = _import_mt5()
            positions = mt5.positions_get()
            if positions is None:
                return []
            return [
                {
                    "ticket":      p.ticket,
                    "symbol":      p.symbol,
                    "side":        "BUY" if p.type == 0 else "SELL",
                    "volume":      p.volume,
                    "open_price":  p.price_open,
                    "current_price": p.price_current,
                    "stop_loss":   p.sl,
                    "take_profit": p.tp,
                    "profit":      p.profit,
                    "swap":        p.swap,
                    "open_time":   p.time,
                    "comment":     p.comment,
                    "magic":       p.magic,
                }
                for p in positions
                if p.magic == _MAGIC
            ]

        return await _run(_positions)

    async def close_position(self, ticket: int) -> dict:
        """Close an open position by ticket number at market price."""
        def _close():
            mt5 = _import_mt5()

            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return {"success": False, "comment": f"Position {ticket} not found"}

            pos = positions[0]
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return {"success": False, "comment": f"Symbol {pos.symbol} tick unavailable"}

            price = tick.bid if pos.type == 0 else tick.ask

            request = {
                "action":      mt5.TRADE_ACTION_DEAL,
                "symbol":      pos.symbol,
                "volume":      pos.volume,
                "type":        close_type,
                "position":    ticket,
                "price":       price,
                "deviation":   20,
                "magic":       _MAGIC,
                "comment":     "close",
                "type_time":   mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                return {"success": False, "comment": str(err), "retcode": -1}

            return {
                "success":    result.retcode == mt5.TRADE_RETCODE_DONE,
                "ticket":     ticket,
                "retcode":    result.retcode,
                "comment":    result.comment,
                "close_price": result.price,
                "profit":     pos.profit,
            }

        return await _run(_close)
