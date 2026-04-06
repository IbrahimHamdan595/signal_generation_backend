from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_active_user,
)
from app.db.database import get_db
from app.services.user_service import UserService
from app.schemas.schemas import (
    RegisterRequest,
    TokenResponse,
    RefreshRequest,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(data: RegisterRequest, pool=Depends(get_db)):
    svc = UserService(pool)

    existing = await svc.get_user_by_email(data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = await svc.create_user(data)
    return UserResponse(
        id=str(user["id"]),
        full_name=user.get("full_name"),
        email=user["email"],
        is_active=user.get("is_active", True),
        is_admin=user.get("is_admin", False),
        watchlist=list(user.get("watchlist", [])),
        created_at=user.get("created_at"),
    )


@router.post("/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends(), pool=Depends(get_db)):
    svc = UserService(pool)
    user = await svc.get_user_by_email(form.username)

    # Debug: Check if user exists
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # Debug: Check if password_hash exists
    password_hash = user.get("password_hash")
    if not password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account issue: password_hash missing",
        )

    # Verify password
    if not verify_password(form.password, password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    if not user.get("is_active", True):
        raise HTTPException(status_code=400, detail="Account is inactive")

    access_token = create_access_token(data={"sub": user["email"]})
    refresh_token = create_refresh_token(data={"sub": user["email"]})

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest, pool=Depends(get_db)):
    payload = decode_token(body.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    email = payload.get("sub")
    svc = UserService(pool)
    user = await svc.get_user_by_email(email)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_token = create_access_token(data={"sub": email})
    refresh_token = create_refresh_token(data={"sub": email})

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_active_user)):
    return UserResponse(
        id=str(current_user.get("id", "")),
        full_name=current_user.get("full_name"),
        email=current_user["email"],
        is_active=current_user.get("is_active", True),
        is_admin=current_user.get("is_admin", False),
        watchlist=list(current_user.get("watchlist", [])),
        created_at=current_user.get("created_at"),
    )
