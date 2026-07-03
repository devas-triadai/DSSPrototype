"""Exception handlers for the DSS API layer.

Maps internal exceptions to standardised HTTP error responses.
No business logic.
"""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from backend.api.responses import ApiErrorResponse

logger = logging.getLogger("dss.api.exceptions")


async def validation_exception_handler(
    request: Request,  # noqa: ARG001
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic / FastAPI request-validation errors."""
    errors = [str(e) for e in exc.errors()]
    logger.warning("Validation error: %s", errors)
    return JSONResponse(
        status_code=422,
        content=ApiErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": errors},
        ).model_dump(),
    )


async def runtime_error_handler(
    request: Request,  # noqa: ARG001
    exc: RuntimeError,
) -> JSONResponse:
    """Handle ``RuntimeError`` (e.g. module not registered)."""
    logger.error("Runtime error: %s", exc)
    return JSONResponse(
        status_code=503,
        content=ApiErrorResponse(
            error_code="RUNTIME_ERROR",
            message=str(exc),
            details={"error": str(exc)},
        ).model_dump(),
    )


async def general_exception_handler(
    request: Request,  # noqa: ARG001
    exc: Exception,
) -> JSONResponse:
    """Catch-all handler for unhandled exceptions."""
    logger.exception("Unhandled exception: %s", exc)
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content=ApiErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An internal server error occurred",
            details={"request_id": request_id},
        ).model_dump(),
    )
