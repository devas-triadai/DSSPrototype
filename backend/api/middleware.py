"""FastAPI middleware for the DSS API layer.

Provides request-level logging, unique request identifiers,
execution timing, and centralised exception handling.
No business logic.
"""

import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from backend.api.responses import ApiErrorResponse

logger = logging.getLogger("dss.api.middleware")

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status, and duration."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Any:
        request_id = str(uuid4())
        request.state.request_id = request_id

        start = time.monotonic()
        method = request.method
        path = request.url.path

        logger.debug("→ %s %s [%s]", method, path, request_id)

        try:
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.info(
                "%s %s → %d (%.1f ms) [%s]",
                method,
                path,
                response.status_code,
                elapsed_ms,
                request_id,
            )
            response.headers[_REQUEST_ID_HEADER] = request_id
            return response

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.exception(
                "%s %s → 500 (%.1f ms) [%s]: %s",
                method,
                path,
                elapsed_ms,
                request_id,
                exc,
            )
            return JSONResponse(
                status_code=500,
                content=ApiErrorResponse(
                    error_code="INTERNAL_ERROR",
                    message="An internal server error occurred",
                    details={"request_id": request_id},
                ).model_dump(),
            )
