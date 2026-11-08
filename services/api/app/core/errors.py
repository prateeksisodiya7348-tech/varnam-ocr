"""Typed application exceptions and their HTTP mapping.

Why a hierarchy instead of raising ``HTTPException`` inline: route handlers
must stay readable, and every failure needs a stable machine-readable ``code``
so the frontend can branch without string-matching English prose.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class VarnamError(Exception):
    """Base class for all expected, client-visible failures.

    Attributes:
        code: Stable snake_case identifier consumed by the frontend.
        message: Human-readable explanation, safe to display.
        http_status: HTTP status code the API should return.
    """

    code: str = "internal_error"
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class InvalidUploadError(VarnamError):
    """The uploaded file failed a hard validation rule."""

    code = "invalid_upload"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY


class PayloadTooLargeError(VarnamError):
    """The file exceeds the configured byte or page ceiling."""

    code = "payload_too_large"
    http_status = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class UnsupportedMediaError(VarnamError):
    """The content is not a supported image or PDF."""

    code = "unsupported_media"
    http_status = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class EngineUnavailableError(VarnamError):
    """The model is not loaded, still warming, or the runtime failed."""

    code = "engine_unavailable"
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE


class EngineBusyError(VarnamError):
    """All inference workers are occupied; the client should retry."""

    code = "engine_busy"
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE


class RateLimitedError(VarnamError):
    """The caller exceeded the token bucket."""

    code = "rate_limited"
    http_status = status.HTTP_429_TOO_MANY_REQUESTS


class NotFoundError(VarnamError):
    """The requested job or resource does not exist."""

    code = "not_found"
    http_status = status.HTTP_404_NOT_FOUND


def install_exception_handlers(app: FastAPI) -> None:
    """Attach JSON error handling to ``app``.

    Args:
        app: The FastAPI application being configured.

    Why this shape: the response envelope is identical for every error
    (``{error: {code, message, request_id}}``), so the frontend has exactly one
    parser and never has to guess. Stack traces are logged, never returned.
    """

    @app.exception_handler(VarnamError)
    async def _handle_varnam_error(request: Request, exc: VarnamError) -> JSONResponse:
        headers: dict[str, str] = {}
        # 503/429 must tell well-behaved clients (and CDNs) when to come back.
        retryable_statuses = (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            status.HTTP_429_TOO_MANY_REQUESTS,
        )
        if exc.http_status in retryable_statuses:
            headers["Retry-After"] = "5"
        return JSONResponse(
            status_code=exc.http_status,
            headers=headers,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Deliberately opaque: leaking a traceback from an ML service can expose
        # filesystem layout and model paths. The request id is the bridge to logs.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Unexpected server error. Quote the request id when reporting.",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )
