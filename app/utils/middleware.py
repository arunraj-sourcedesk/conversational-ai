"""
HTTP middleware stack.

1. RequestLoggingMiddleware  — logs every request/response with timing
2. ExceptionMiddleware       — catches unhandled exceptions and returns
                               structured JSON errors
"""

import time
import uuid
import json

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each HTTP request with a unique trace ID and duration."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        trace_id = str(uuid.uuid4())[:8]
        request.state.trace_id = trace_id
        start = time.perf_counter()

        # Read and log request body (body() caches the body for downstream)
        try:
            raw_req_body = await request.body()
        except Exception:
            raw_req_body = b""

        def _format_body(raw: bytes) -> str:
            if not raw:
                return ""
            max_len = 2000
            try:
                text = raw.decode("utf-8")
            except Exception:
                return "<binary>"

            # If JSON, pretty-print it for readability
            try:
                parsed = json.loads(text)
                pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                pretty = text

            if len(pretty) > max_len:
                return pretty[:max_len] + "...(<truncated>)"
            return pretty

        req_body_text = _format_body(raw_req_body)

        logger.info(
            "→ %s %s trace_id=%s body=%s",
            request.method,
            request.url.path,
            trace_id,
            (req_body_text if req_body_text else "-"),
        )


        try:
            response: Response = await call_next(request)
        except Exception:
            logger.exception("Unhandled exception trace_id=%s", trace_id)
            raise


        # Capture response body. Streaming responses expose .body_iterator.
        resp_body = b""
        try:
            if hasattr(response, "body_iterator") and response.body_iterator is not None:
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    resp_body += chunk
            elif hasattr(response, "body"):
                resp_body = response.body or b""
        except Exception:
            # If we fail to read the body, fall back and continue
            resp_body = b""

        resp_text = _format_body(resp_body)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "← %s %s status=%d %.1fms trace_id=%s body=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            trace_id,
            (resp_text if resp_text else "-"),
        )

        # Rebuild response because we consumed the body iterator
        new_headers = dict(response.headers) if hasattr(response, "headers") else {}
        new_headers["X-Trace-Id"] = trace_id

        media_type = getattr(response, "media_type", None)
        new_response = Response(content=resp_body, status_code=response.status_code, headers=new_headers, media_type=media_type)
        return new_response


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Convert domain exceptions and unexpected errors into JSON responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        try:
            return await call_next(request)
        except AppError as exc:
            logger.warning(
                "AppError %s: %s", type(exc).__name__, exc.message
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.message, "type": type(exc).__name__},
            )
        except Exception as exc:
            logger.exception("Unexpected error: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "type": "InternalError"},
            )
