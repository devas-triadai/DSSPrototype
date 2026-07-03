"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Force anyio to use asyncio for async tests."""
    return "asyncio"
