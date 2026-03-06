"""
LUMINARK API — Request Tracing Middleware
Assigns a UUID request_id to every request, injects it into request.state,
adds X-Request-ID and X-Response-Time headers, and binds structlog context.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("luminark.middleware")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.monotonic()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)

        duration_ms = (time.monotonic() - request.state.start_time) * 1000
        response.headers["X-Request-ID"]    = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"

        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 1),
        )
        return response
