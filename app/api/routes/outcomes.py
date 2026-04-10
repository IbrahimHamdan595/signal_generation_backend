from fastapi import APIRouter, Depends
from app.db.database import get_db
from app.services.outcome_service import OutcomeService
from app.core.security import get_current_user

router = APIRouter(prefix="/outcomes", tags=["outcomes"])


def _svc(pool=Depends(get_db)) -> OutcomeService:
    return OutcomeService(pool)


@router.get("")
async def get_outcomes(
    ticker: str | None = None,
    limit: int = 100,
    svc: OutcomeService = Depends(_svc),
    _=Depends(get_current_user),
):
    return await svc.get_outcomes(ticker=ticker, limit=limit)


@router.get("/summary")
async def accuracy_summary(
    svc: OutcomeService = Depends(_svc),
    _=Depends(get_current_user),
):
    return await svc.get_accuracy_summary()


@router.post("/check")
async def trigger_check(
    svc: OutcomeService = Depends(_svc),
    _=Depends(get_current_user),
):
    """Manually trigger outcome resolution (normally runs on scheduler)."""
    return await svc.check_pending_outcomes()
