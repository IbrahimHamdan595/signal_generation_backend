from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List
from pydantic import BaseModel

from app.core.security import get_current_active_user

limiter = Limiter(key_func=get_remote_address)
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
@limiter.limit("30/minute")
async def generate_signal(
    request: Request,
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


@router.get("/{signal_id}/explain")
async def explain_signal(
    signal_id: int,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Return gradient-based feature importance for a stored signal.
    Shows which price/indicator features most influenced the model's decision.
    """
    from app.ml.models.registry import get_model, load_scaler_params
    from app.ml.data.dataset import DatasetBuilder, FEATURE_COLS, SEQUENCE_LEN
    import torch, numpy as np

    # Load the signal from DB
    async with pool.acquire() as conn:
        sig = await conn.fetchrow("SELECT * FROM signals WHERE id = $1", signal_id)
    if not sig:
        raise HTTPException(404, "Signal not found")

    model = get_model()
    if model is None:
        raise HTTPException(400, "Model not trained")

    ticker, interval = sig["ticker"], sig["interval"]
    builder = DatasetBuilder(pool)
    X_price, X_sent, _, _, _ = await builder.build([ticker], interval, sequence_len=SEQUENCE_LEN)
    if X_price is None or len(X_price) == 0:
        raise HTTPException(404, "No feature data available for this signal")

    scaler = load_scaler_params()
    if scaler:
        ts = scaler.get(ticker.upper()) or scaler.get(ticker)
        if ts:
            X_price = (X_price - np.array(ts["mean"])) / (np.array(ts["std"]) + 1e-8)

    device = next(model.parameters()).device
    x_p = torch.tensor(X_price[-1:], dtype=torch.float32, requires_grad=True).to(device)
    x_s = torch.tensor(X_sent[-1:], dtype=torch.float32).to(device)

    model.eval()
    out = model(x_p, x_s)
    class_logits = out[0]
    pred_class = int(class_logits.argmax(dim=1).item())
    class_logits[0, pred_class].backward()

    # Average absolute gradient across time dimension → per-feature importance
    grads = x_p.grad.abs().mean(dim=1).squeeze().cpu().numpy()
    total = float(grads.sum()) or 1.0
    importances = {
        col: round(float(grads[i]) / total, 4)
        for i, col in enumerate(FEATURE_COLS)
        if i < len(grads)
    }
    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    top = dict(list(importances.items())[:15])

    return {
        "signal_id": signal_id,
        "ticker": ticker,
        "action": sig["action"],
        "top_features": top,
        "all_features": importances,
    }
