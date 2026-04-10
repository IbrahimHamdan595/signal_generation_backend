from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.portfolio_service import PortfolioService

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


class OpenPositionRequest(BaseModel):
    ticker: str
    quantity: float
    price: float
    signal_id: Optional[int] = None


class ClosePositionRequest(BaseModel):
    price: float


@router.get("/positions")
async def get_positions(
    open_only: bool = True,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = PortfolioService(pool)
    return await svc.get_positions(current_user["id"], open_only)


@router.get("/summary")
async def get_summary(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = PortfolioService(pool)
    return await svc.get_summary(current_user["id"])


@router.post("/positions")
async def open_position(
    body: OpenPositionRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = PortfolioService(pool)
    return await svc.open_position(
        user_id=current_user["id"],
        ticker=body.ticker,
        quantity=body.quantity,
        price=body.price,
        signal_id=body.signal_id,
    )


@router.post("/positions/{position_id}/close")
async def close_position(
    position_id: int,
    body: ClosePositionRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = PortfolioService(pool)
    try:
        return await svc.close_position(current_user["id"], position_id, body.price)
    except ValueError as e:
        raise HTTPException(404, str(e))
