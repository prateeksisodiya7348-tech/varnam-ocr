"""OCR endpoints.

Latency budgets (documented here so reviewers do not have to guess):

* ``POST /v1/ocr``      single image, synchronous, target p95 <= 3s on 1 vCPU.
* ``POST /v1/ocr/pdf``  returns 202 in < 200ms; a 10-page scan takes ~20s in the
  background and is polled through ``GET /v1/jobs/{id}``.
* ``GET  /v1/jobs/{id}`` single row read, p95 < 30ms.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, Request, UploadFile, status
from fastapi.responses import PlainTextResponse, Response
from sqlmodel import Session

from app.core.errors import InvalidUploadError, NotFoundError, RateLimitedError
from app.db import get_session
from app.dependencies import AppServices, ServicesDep
from app.models.db import JobStatus, OcrJob
from app.models.schemas import (
    BBox,
    ErrorResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    LineOut,
    OcrResponse,
    PageOut,
)
from app.services import runner
from app.services.export import to_hocr, to_plain_text

router = APIRouter(prefix="/v1", tags=["ocr"])

SessionDep = Annotated[Session, Depends(get_session)]

# Multi-page work costs more of the bucket than a single page; the numbers are
# proportional to CPU seconds, not to requests, which is what actually breaks.
_COST_SINGLE = 1.0
_COST_PDF = 4.0


def _lines_from_payload(payload: dict[str, Any]) -> list[LineOut]:
    """Rebuild typed line objects from a stored job result.

    Args:
        payload: The ``result`` JSON column of a job.

    Returns:
        Lines with boxes and confidence. Word boxes are not persisted, so they
        come back empty rather than being invented.

    Why this exists: the stored form is plain JSON. Leaving raw dicts in place
    type-checks as ``list[dict]`` and silently bypasses the response schema, so a
    change to ``LineOut`` would not be caught by mypy.
    """
    lines: list[LineOut] = []
    for raw in payload.get("lines", []):
        x0, y0, x1, y1 = raw["bbox"]
        lines.append(
            LineOut(
                text=raw["text"],
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                confidence=raw["confidence"],
                words=[],
            )
        )
    return lines


def _enforce_rate_limit(services: AppServices, request: Request, *, cost: float) -> None:
    """Consume tokens for this caller or raise ``RateLimitedError``.

    Args:
        services: Service container holding the limiter.
        request: Used to derive the client key.
        cost: Tokens this request consumes.

    Raises:
        RateLimitedError: When the bucket is exhausted.
    """
    key = services.client_key(request)
    allowed, retry_after = services.limiter.allow(key, cost=cost)
    if not allowed:
        raise RateLimitedError(
            f"Rate limit reached. Try again in {max(1, round(retry_after))}s.",
        )


@router.post(
    "/ocr",
    response_model=OcrResponse,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Recognize a single image",
)
def create_ocr(
    request: Request,
    services: ServicesDep,
    session: SessionDep,
    file: Annotated[UploadFile, File(description="PNG, JPEG or WEBP image")],
    export: Annotated[
        Literal["json", "txt", "hocr"] | None,
        Query(description="When set, return the raw export body instead of JSON."),
    ] = None,
) -> OcrResponse | Response:
    """Run OCR on one image and persist the result.

    Args:
        request: Incoming request (client key, request id).
        services: Engine, storage, limiter, settings.
        session: Database session.
        file: The uploaded image.
        export: Optional export format shortcut for curl users.

    Returns:
        The JSON result, or a text/hOCR body when ``export`` is set.

    Raises:
        RateLimitedError: Bucket exhausted.
        InvalidUploadError: Undecodable or empty file.
    """
    _enforce_rate_limit(services, request, cost=_COST_SINGLE)

    # `file.file.read()` is bounded by the same cap we validate against; reading
    # the whole body into memory is safe because the cap is 10MB.
    data = file.file.read(services.settings.max_upload_mb * 1_048_576 + 1)

    result = runner.process_image_upload(
        data,
        engine=services.engine,
        settings=services.settings,
        storage=services.storage,
        session=session,
        client_hash=services.client_key(request),
    )

    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.inc("ocr_pages_total")

    # Rebuild the response through the same mapper used for stored jobs so the
    # export formats and the JSON can never disagree.
    job = session.get(OcrJob, result.job_id)
    if job is None:  # pragma: no cover - impossible immediately after commit
        raise InvalidUploadError("Job vanished after processing.")

    payload = runner.to_response(
        job,
        result.page,
        original_size=tuple((job.result or {}).get("original_size", [0, 0])),
    )

    if export == "txt":
        return PlainTextResponse(to_plain_text(result.page), media_type="text/plain; charset=utf-8")
    if export == "hocr":
        return Response(
            content=to_hocr(result.page),
            media_type="application/xhtml+xml; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="varnam.hocr"'},
        )
    return payload


@router.post(
    "/ocr/pdf",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
    summary="Queue a PDF for asynchronous OCR",
)
def create_pdf_job(
    request: Request,
    services: ServicesDep,
    session: SessionDep,
    background: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF document")],
) -> JobAcceptedResponse:
    """Accept a PDF, return 202, and process pages in the background.

    Args:
        request: Incoming request.
        services: Service container.
        session: Database session.
        background: FastAPI background task registry.
        file: The uploaded PDF.

    Returns:
        The accepted-job envelope with a poll URL.

    Why 202 instead of blocking: a multi-page scan exceeds the platform request
    timeout, and a timeout at the proxy produces an HTML error page the client
    cannot parse. A job id keeps the contract JSON-only.
    """
    _enforce_rate_limit(services, request, cost=_COST_PDF)
    data = file.file.read(services.settings.max_upload_mb * 1_048_576 + 1)

    job_id = runner.process_pdf_upload(
        data,
        engine=services.engine,
        settings=services.settings,
        storage=services.storage,
        session=session,
        client_hash=services.client_key(request),
    )
    job = session.get(OcrJob, job_id)
    page_count = job.page_count if job else 1

    # Bound by the job's page count so a 20-page document cannot occupy the
    # single inference slot for a minute without the limiter noticing.
    background.add_task(
        runner.process_pdf_job,
        job_id,
        settings=services.settings,
        storage=services.storage,
        engine=services.engine,
    )

    return JobAcceptedResponse(
        job_id=job_id,
        page_count=page_count,
        poll_url=f"/v1/jobs/{job_id}",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Poll an asynchronous job",
)
def get_job(job_id: str, session: SessionDep) -> JobStatusResponse:
    """Return the current state of a job.

    Args:
        job_id: Job identifier returned by ``POST /v1/ocr/pdf``.
        session: Database session.

    Returns:
        Status payload; ``result`` is populated only when the job succeeded.

    Raises:
        NotFoundError: When the id is unknown.
    """
    job = session.get(OcrJob, job_id)
    if job is None:
        raise NotFoundError(f"No job with id {job_id}.")

    result: OcrResponse | None = None
    if job.status == JobStatus.SUCCEEDED and job.result:
        payload = job.result
        original = payload.get("original_size", [0, 0])
        result = OcrResponse(
            job_id=job.id,
            engine=job.engine,
            text=payload.get("text", ""),
            lines=_lines_from_payload(payload),
            page=PageOut(
                width=original[0] if original else 0,
                height=original[1] if original else 0,
                scale=1.0,
                line_count=len(payload.get("lines", [])),
            ),
            mean_confidence=payload.get("mean_confidence", 0.0),
            devanagari_ratio=payload.get("devanagari_ratio", 0.0),
            duration_ms=job.duration_ms,
            created_at=job.created_at,
        )

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        page_count=job.page_count,
        engine=job.engine,
        error=job.error_message,
        result=result,
    )
