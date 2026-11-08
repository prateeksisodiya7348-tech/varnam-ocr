"""Operational endpoints.

Three distinct questions, three distinct endpoints - conflating them is the
classic cause of a platform killing a container that was merely slow to start:

* ``/health/live``  - is the process alive? Never touches the model or the DB.
* ``/health/ready`` - should traffic be routed here? Depends on the model.
* ``/metrics``      - Prometheus text for latency and failure rates.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response

from app.dependencies import ServicesDep
from app.models.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
def liveness(services: ServicesDep) -> HealthResponse:
    """Report that the process is up, without checking dependencies.

    Args:
        services: Service container.

    Returns:
        Always ``status="ok"`` when the event loop can respond at all.

    Why no DB check: a transient database blip must not cause a restart loop on a
    free host that only allows one instance.
    """
    started = getattr(services, "_started_at", None)
    return HealthResponse(
        status="ok",
        engine=services.engine.name,
        model_loaded=services.engine.is_ready(),
        version=_version(),
        uptime_seconds=round(time.monotonic() - (started or time.monotonic()), 1),
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
def readiness(request: Request, services: ServicesDep) -> Response:
    """Report whether this instance can serve OCR traffic right now.

    Args:
        request: Used to read the app start time.
        services: Service container.

    Returns:
        200 when the model is loaded, 503 otherwise, so the platform does not
        route traffic to an instance that would return ``engine_unavailable``.
    """
    started_at = getattr(request.app.state, "started_at", time.monotonic())
    ready = services.engine.is_ready()
    payload = HealthResponse(
        status="ok" if ready else "degraded",
        engine=services.engine.name,
        model_loaded=ready,
        version=_version(),
        uptime_seconds=round(time.monotonic() - started_at, 1),
    ).model_dump_json()

    return Response(
        content=payload,
        status_code=200 if ready else 503,
        media_type="application/json",
    )


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    """Expose counters and the latency histogram.

    Args:
        request: Used to reach the process-wide ``Metrics`` instance.

    Returns:
        Prometheus text exposition format.
    """
    collector = getattr(request.app.state, "metrics", None)
    body = collector.render() if collector is not None else "# metrics unavailable\n"
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


def _version() -> str:
    """Return the package version without importing the app package lazily twice."""
    from app import __version__

    return __version__
