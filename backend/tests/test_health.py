"""Tests for the health-check endpoint."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    """Provide an async HTTP client wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check_returns_healthy(client: AsyncClient) -> None:
    """GET /api/v1/health should return status 'healthy'."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DSSPrototype"
    assert data["version"] == "0.1.0"
