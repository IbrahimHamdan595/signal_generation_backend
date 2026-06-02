from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel

from app.core.security import get_current_active_user, hash_password, verify_password
from app.db.database import get_db
from app.schemas.schemas import UserResponse, MessageResponse

router = APIRouter(prefix="/users", tags=["Users"])


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.put("/me", response_model=UserResponse)
async def update_profile(
    body: UpdateProfileRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return UserResponse(
            id=str(current_user["id"]),
            full_name=current_user.get("full_name"),
            email=current_user["email"],
            is_active=current_user.get("is_active", True),
            is_admin=current_user.get("is_admin", False),
            created_at=current_user.get("created_at"),
        )
    set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates))
    vals = list(updates.values())
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"UPDATE users SET {set_clause}, updated_at = NOW() WHERE id = $1 RETURNING *",
            current_user["id"], *vals,
        )
    return UserResponse(
        id=str(row["id"]),
        full_name=row.get("full_name"),
        email=row["email"],
        is_active=row.get("is_active", True),
        is_admin=row.get("is_admin", False),
        created_at=row.get("created_at"),
    )


@router.post("/me/password", response_model=MessageResponse)
async def change_password(
    body: ChangePasswordRequest,
    pool=Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    if not verify_password(body.current_password, current_user["password_hash"]):
        raise HTTPException(400, "Current password is incorrect")
    new_hash = hash_password(body.new_password)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2",
            new_hash, current_user["id"],
        )
    return MessageResponse(message="Password updated successfully")
