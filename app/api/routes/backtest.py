from fastapi import APIRouter, Depends, Query
from typing import List
from app.db.database import get_db
from app.services.backtest_service import BacktestService
from app.core.security import get_current_user

router = APIRouter(prefix="/backtest", tags=["backtest"])


def _svc(pool=Depends(get_db)) -> BacktestService:
    return BacktestService(pool)


@router.get("/{ticker}")
async def backtest_ticker(
    ticker: str,
    interval: str = Query("1d", regex="^(1d|1h)$"),
    initial_capital: float = Query(10_000.0, ge=100),
    position_size_pct: float = Query(0.10, ge=0.01, le=1.0),
    max_bars_held: int = Query(10, ge=1, le=50),
    svc: BacktestService = Depends(_svc),
    _=Depends(get_current_user),
):
    return await svc.run(
        ticker=ticker,
        interval=interval,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        max_bars_held=max_bars_held,
    )


@router.post("/portfolio")
async def backtest_portfolio(
    tickers: List[str],
    interval: str = Query("1d"),
    initial_capital: float = Query(100_000.0, ge=1000),
    svc: BacktestService = Depends(_svc),
    _=Depends(get_current_user),
):
    return await svc.run_portfolio(tickers, interval, initial_capital)
