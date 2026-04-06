import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Test User",
            "email": "test@example.com",
            "password": "securepassword123",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data


@pytest.mark.asyncio
async def test_register_duplicate(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Test User",
            "email": "duplicate@example.com",
            "password": "securepassword123",
        },
    )
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Test User 2",
            "email": "duplicate@example.com",
            "password": "securepassword456",
        },
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Login User",
            "email": "login@example.com",
            "password": "securepassword123",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "login@example.com", "password": "securepassword123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "nobody@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Me User",
            "email": "me@example.com",
            "password": "securepassword123",
        },
    )
    login_resp = await client.post(
        "/api/v1/auth/login",
        data={"username": "me@example.com", "password": "securepassword123"},
    )
    token = login_resp.json()["access_token"]
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "me@example.com"
    assert data["full_name"] == "Me User"


@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient):
    await client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Refresh User",
            "email": "refresh@example.com",
            "password": "securepassword123",
        },
    )
    login_resp = await client.post(
        "/api/v1/auth/login",
        data={"username": "refresh@example.com", "password": "securepassword123"},
    )
    refresh_token = login_resp.json()["refresh_token"]
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
