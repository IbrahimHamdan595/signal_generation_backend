from fastapi import APIRouter, Depends, HTTPException
from app.db.database import get_db
from app.services.alert_service import AlertService
from app.core.security import get_current_user

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _svc(pool=Depends(get_db)) -> AlertService:
    return AlertService(pool)


@router.get("")
async def get_alerts(
    unread_only: bool = False,
    limit: int = 50,
    svc: AlertService = Depends(_svc),
    _=Depends(get_current_user),
):
    return await svc.get_alerts(unread_only=unread_only, limit=limit)


@router.get("/count")
async def unread_count(
    svc: AlertService = Depends(_svc),
    _=Depends(get_current_user),
):
    return {"unread": await svc.unread_count()}


@router.post("/{alert_id}/read")
async def mark_read(
    alert_id: int,
    svc: AlertService = Depends(_svc),
    _=Depends(get_current_user),
):
    ok = await svc.mark_read(alert_id)
    if not ok:
        raise HTTPException(404, "Alert not found")
    return {"ok": True}


@router.post("/read-all")
async def mark_all_read(
    svc: AlertService = Depends(_svc),
    _=Depends(get_current_user),
):
    count = await svc.mark_all_read()
    return {"marked": count}
