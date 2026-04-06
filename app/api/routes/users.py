from fastapi import APIRouter, Depends
from typing import List

from app.core.security import get_current_active_user
from app.db.database import get_db
from app.services.user_service import UserService
from app.schemas.schemas import UserResponse, WatchlistUpdate, MessageResponse

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me/watchlist", response_model=List[str])
async def get_watchlist(current_user=Depends(get_current_active_user)):
    return current_user.get("watchlist", [])


@router.put("/me/watchlist", response_model=UserResponse)
async def set_watchlist(
    body: WatchlistUpdate,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = UserService(pool)
    user = await svc.update_watchlist(current_user["id"], body.tickers)
    return UserResponse(
        id=str(user["id"]),
        full_name=user.get("full_name"),
        email=user["email"],
        is_active=user.get("is_active", True),
        is_admin=user.get("is_admin", False),
        watchlist=list(user.get("watchlist", [])),
        created_at=user.get("created_at"),
    )


@router.post("/me/watchlist/{ticker}", response_model=MessageResponse)
async def add_ticker_to_watchlist(
    ticker: str,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = UserService(pool)
    await svc.add_to_watchlist(current_user["id"], ticker.upper())
    return MessageResponse(message=f"{ticker.upper()} added to watchlist")


@router.delete("/me/watchlist/{ticker}", response_model=MessageResponse)
async def remove_ticker_from_watchlist(
    ticker: str,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    svc = UserService(pool)
    await svc.remove_from_watchlist(current_user["id"], ticker.upper())
    return MessageResponse(message=f"{ticker.upper()} removed from watchlist")
