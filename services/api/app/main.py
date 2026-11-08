"""Application factory.

``create_app`` takes optional overrides so tests can build an isolated instance
with a fake engine and a temp storage directory. A module-level ``app`` is still
exported because ``uvicorn app.main:app`` must work - that is the deployment
contract Render expects.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from app import __version__
from app.config import Settings, get_settings
from app.core.errors import install_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.metrics import Metrics
from app.core.ratelimit import TokenBucketLimiter
from app.db import init_db
from app.dependencies import AppServices
from app.routes import feedback, health, ocr
from app.services.engine import OcrEngine, build_engine
from app.services.storage import StorageBackend, build_storage

logger = get_logger(__name__)

API_DESCRIPTION = """
Devanagari OCR for printed and handwritten text.

* **POST /v1/ocr** - synchronously recognize one PNG/JPEG/WEBP page.
* **POST /v1/ocr/pdf** - queue a PDF; poll `GET /v1/jobs/{id}`.
* **POST /v1/feedback** - submit a corrected line; it becomes a training sample.

Every response carries pixel-accurate boxes in the coordinate space of the
original upload, so highlights line up without client-side math.
""".strip()


def create_app(  # noqa: PLR0915 - a factory is a flat, readable wiring list by design
    settings: Settings | None = None,
    *,
    engine: OcrEngine | None = None,
    storage: StorageBackend | None = None,
) -> FastAPI:
    """Build a fully wired FastAPI application.

    Args:
        settings: Override configuration; defaults to the environment.
        engine: Override the OCR backend (tests inject a fake).
        storage: Override upload persistence (tests inject a temp directory).

    Returns:
        A ready-to-serve application.
    """
    resolved_settings = settings or get_settings()
    configure_logging(
        level=resolved_settings.log_level,
        json_output=resolved_settings.is_production,
    )

    # Built outside the lifespan so tests can inspect them before startup and
    # so a misconfigured model fails at import time, not on the first request.
    ocr_engine = engine or build_engine(resolved_settings)
    upload_storage = storage or build_storage(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize state once per process and release it on shutdown."""
        app.state.started_at = time.monotonic()
        app.state.metrics = Metrics()
        app.state.services = AppServices(
            engine=ocr_engine,
            storage=upload_storage,
            limiter=TokenBucketLimiter.per_minute(resolved_settings.rate_limit_per_minute),
            settings=resolved_settings,
        )

        init_db()
        logger.info(
            "startup",
            version=__version__,
            env=resolved_settings.app_env,
            engine=ocr_engine.name,
            storage=resolved_settings.storage_backend,
        )
        try:
            yield
        finally:
            # Nothing holds OS resources here except the onnxruntime session,
            # which is reclaimed with the process. Logging shutdown explicitly
            # makes "why did my instance die?" answerable from logs alone.
            logger.info("shutdown", version=__version__)

    app = FastAPI(
        title="Varnam OCR",
        version=__version__,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        # Docs are useful in every environment for a portfolio project; the
        # write endpoints are rate limited regardless.
        # Docs live under /api so they cannot collide with the frontend's own
        # /docs page when the SPA is mounted at "/".
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    # Order matters: CORS must wrap everything, and gzip must sit outside the
    # routes so it compresses JSON responses (a 40-line result is ~8KB raw).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        max_age=600,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def _observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request id, time the request, and record metrics.

        Args:
            request: The incoming request.
            call_next: The downstream handler.

        Returns:
            The downstream response with tracing headers added.

        Why a header-derived id: a client that already has a correlation id
        (from its own logs) keeps it, so one click in the UI ties to one backend
        trace without a separate tracing backend.
        """
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = str(duration_ms)

        metrics: Metrics | None = getattr(request.app.state, "metrics", None)
        if metrics is not None:
            metrics.inc("ocr_requests_total")
            metrics.observe_latency(duration_ms)
            if response.status_code >= 400:
                metrics.inc("ocr_failures_total")

        logger.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    @app.middleware("http")
    async def _security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Add baseline security headers to every response.

        Args:
            request: The incoming request.
            call_next: The downstream handler.

        Returns:
            The response with headers applied.
        """
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        # API responses are never documents; a restrictive CSP costs nothing and
        # prevents an XSS from being escalated through a reflected text/plain body.
        response.headers.setdefault(
            "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
        )
        if resolved_settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response

    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(ocr.router)
    app.include_router(feedback.router)

    # The built frontend ships inside the image (see services/api/Dockerfile) and
    # is served from the same origin, so the browser never makes a cross-origin
    # request and CORS is irrelevant for normal use.
    static_dir = Path(os.environ.get("STATIC_DIR", "/app/static"))
    if static_dir.is_dir():
        # Mounted LAST so every exact API route above takes precedence over it.
        # html=True serves index.html for directories, which is what the static
        # export produces for /app and /docs.
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
        logger.info("serving_frontend", directory=str(static_dir))
    else:
        # Local development (or an API-only deployment): keep a JSON landing page
        # so hitting the root still tells you what this service is.
        @app.get("/", include_in_schema=False)
        def root() -> dict[str, str]:
            """Human-friendly landing payload for anyone hitting the API root."""
            return {
                "name": "Varnam OCR",
                "version": __version__,
                "docs": "/api/docs",
                "health": "/health/ready",
            }

    return app


# Module-level instance for `uvicorn app.main:app`.
app = create_app()
