"""Health-check endpoint for readiness probes.

This is the **only** API endpoint implemented at this stage.
All future endpoints will be added in separate router modules.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return basic service health information.

    Returns
    -------
    dict
        A JSON payload with status, service name, and version.
    """
    return {
        "status": "healthy",
        "service": "DSSPrototype",
        "version": "0.1.0",
    }
