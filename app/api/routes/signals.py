from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List
from pydantic import BaseModel

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.signal_service import SignalService
from app.services.ticker_list_service import ticker_list_service
from app.ml.models.registry import is_model_trained
from app.schemas.signal import (
    SignalResponse,
    SignalGenerateRequest,
    SignalBatchRequest,
)

router = APIRouter(prefix="/signals", tags=["Trading Signals"])


def _to_response(doc: dict) -> SignalResponse:
    return SignalResponse(
        id=str(doc.get("id", "")),
        ticker=doc["ticker"],
        interval=doc["interval"],
        action=doc["action"],
        confidence=doc["confidence"],
        entry_price=doc.get("entry_price"),
        stop_loss=doc.get("stop_loss"),
        take_profit=doc.get("take_profit"),
        net_profit=doc.get("net_profit"),
        bars_to_entry=doc.get("bars_to_entry"),
        entry_time=doc.get("entry_time"),
        probabilities={
            "buy": doc.get("prob_buy"),
            "sell": doc.get("prob_sell"),
            "hold": doc.get("prob_hold"),
        },
        prob_buy=doc.get("prob_buy"),
        prob_sell=doc.get("prob_sell"),
        prob_hold=doc.get("prob_hold"),
        source=doc.get("source", "ml_model"),
        created_at=doc.get("created_at"),
    )


def _check_model():
    if not is_model_trained():
        raise HTTPException(
            400, "No trained model found. Run POST /api/v1/ml/train first, then retry."
        )


@router.get("", response_model=List[SignalResponse])
async def list_signals(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """List latest signals from all tickers"""
    svc = SignalService(pool)
    docs = await svc.get_all_latest()
    return [_to_response(d) for d in docs]


@router.post("/generate", response_model=SignalResponse)
async def generate_signal(
    body: SignalGenerateRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    _check_model()
    svc = SignalService(pool)
    doc = await svc.generate_and_store(body.ticker, body.interval)
    if not doc:
        raise HTTPException(
            404, f"Could not generate signal for {body.ticker}. Check data."
        )
    return _to_response(doc)


@router.post("/generate/batch", response_model=List[SignalResponse])
async def generate_batch(
    body: SignalBatchRequest,
    background_tasks: BackgroundTasks,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    _check_model()
    if len(body.tickers) > 50:
        raise HTTPException(400, "Max 50 tickers per batch request")

    async def _run():
        svc = SignalService(pool)
        await svc.generate_batch(body.tickers, body.interval)

    background_tasks.add_task(_run)
    return []


@router.get("/latest/{ticker}", response_model=SignalResponse)
async def get_latest(
    ticker: str,
    interval: str = Query(default="1d"),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = SignalService(pool)
    doc = await svc.get_latest(ticker.upper(), interval)
    if not doc:
        raise HTTPException(404, f"No signal found for {ticker.upper()}")
    return _to_response(doc)


@router.get("/history/{ticker}", response_model=List[SignalResponse])
async def get_history(
    ticker: str,
    interval: str = Query(default="1d"),
    limit: int = Query(default=50, le=200),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = SignalService(pool)
    docs = await svc.get_history(ticker.upper(), interval, limit)
    if not docs:
        raise HTTPException(404, f"No signal history for {ticker.upper()}")
    return [_to_response(d) for d in docs]


@router.get("/all/latest", response_model=List[SignalResponse])
async def get_all_latest(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = SignalService(pool)
    docs = await svc.get_all_latest()
    return [_to_response(d) for d in docs]


@router.get("/filter/action/{action}", response_model=List[SignalResponse])
async def get_by_action(
    action: str,
    interval: str = Query(default="1d"),
    limit: int = Query(default=50, le=200),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if action.upper() not in {"BUY", "SELL", "HOLD"}:
        raise HTTPException(400, "action must be BUY, SELL, or HOLD")
    svc = SignalService(pool)
    docs = await svc.get_by_action(action, interval, limit)
    return [_to_response(d) for d in docs]


@router.get("/filter/high-confidence", response_model=List[SignalResponse])
async def get_high_confidence(
    min_confidence: float = Query(default=0.75, ge=0.5, le=1.0),
    interval: str = Query(default="1d"),
    limit: int = Query(default=20, le=100),
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = SignalService(pool)
    docs = await svc.get_high_confidence(min_confidence, interval, limit)
    return [_to_response(d) for d in docs]


class TickerListOverrideRequest(BaseModel):
    tickers: List[str]


@router.post("/tickers/override")
async def override_ticker_list(
    body: TickerListOverrideRequest,
    current_user=Depends(get_current_active_user),
):
    ticker_list_service.set_custom_tickers(body.tickers)
    return {"message": f"Custom ticker list set: {len(body.tickers)} tickers"}


@router.delete("/tickers/override")
async def clear_ticker_override(
    current_user=Depends(get_current_active_user),
):
    ticker_list_service.clear_custom_tickers()
    return {"message": "Custom ticker list cleared"}


@router.get("/tickers")
async def get_ticker_list(
    current_user=Depends(get_current_active_user),
):
    tickers = await ticker_list_service.get_tickers()
    return {"tickers": tickers, "count": len(tickers)}
