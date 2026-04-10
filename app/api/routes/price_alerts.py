from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Literal

from app.core.security import get_current_active_user
from app.db.database import get_db

router = APIRouter(prefix="/price-alerts", tags=["Price Alert Rules"])


class CreateRuleRequest(BaseModel):
    ticker: str
    condition: Literal["above", "below"]
    target_price: float


@router.get("")
async def get_rules(
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM price_alert_rules
            WHERE user_id = $1
            ORDER BY created_at DESC
        """, current_user["id"])
    return [dict(r) for r in rows]


@router.post("")
async def create_rule(
    body: CreateRuleRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO price_alert_rules (user_id, ticker, condition, target_price)
            VALUES ($1, $2, $3, $4) RETURNING *
        """, current_user["id"], body.ticker.upper(), body.condition, body.target_price)
    return dict(row)


@router.delete("/{rule_id}")
async def delete_rule(
    rule_id: int,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM price_alert_rules
            WHERE id = $1 AND user_id = $2
        """, rule_id, current_user["id"])
    if result == "DELETE 0":
        raise HTTPException(404, "Rule not found")
    return {"message": "Rule deleted"}
