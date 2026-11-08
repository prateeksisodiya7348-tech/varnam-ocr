"""The single place where a request becomes a stored result.

Routes stay thin; this module owns the sequence and the failure semantics. That
separation is what lets ``tests/test_runner.py`` exercise the whole business
path without HTTP, and lets the PDF background task reuse the exact same code
as the synchronous image path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from sqlmodel import Session

from app.config import Settings
from app.core.errors import EngineUnavailableError
from app.core.logging import Timer, get_logger
from app.models.db import JobStatus, OcrJob
from app.models.schemas import BBox, LineOut, OcrResponse, PageOut, WordOut
from app.services.engine import OcrEngine
from app.services.export import PageResult, run_page
from app.services.preprocess import prepare_image
from app.services.segmentation import Box
from app.services.storage import StorageBackend
from app.services.validation import ValidatedUpload, validate_upload

logger = get_logger(__name__)

# One inference at a time. Render's free instance has a single vCPU, so parallel
# inference produces no extra throughput - it only multiplies peak RSS. A
# semaphore converts overload into a clean 503 instead of an OOM kill.
_INFERENCE_LOCK = threading.BoundedSemaphore(value=1)

_EXTENSION_BY_KIND = {"png": "png", "jpeg": "jpg", "webp": "webp", "pdf": "pdf"}


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    """Outcome of processing a single image upload.

    Attributes:
        job_id: Persisted job identifier.
        page: The canonical recognition result.
    """

    job_id: str
    page: PageResult


def _scaled_bbox(box: Box, scale: float) -> BBox:
    """Convert a box from preprocessed coordinates back to original pixels.

    Args:
        box: Box in preprocessed-image space.
        scale: Factor preprocessing applied (<= 1.0).

    Returns:
        A ``BBox`` in the coordinate space of the user's original upload, which
        is what the browser canvas draws against.

    Why this matters: without the inverse transform, every highlight box on a
    downscaled photo is offset toward the top-left, which looks like a model bug
    to a reviewer when it is really a coordinate-space bug.
    """
    inverse = 1.0 / scale if scale > 0 else 1.0
    return BBox(
        x0=round(box.x0 * inverse, 2),
        y0=round(box.y0 * inverse, 2),
        x1=round(box.x1 * inverse, 2),
        y1=round(box.y1 * inverse, 2),
    )


def to_response(job: OcrJob, page: PageResult, *, original_size: tuple[int, int]) -> OcrResponse:
    """Map a ``PageResult`` onto the public response schema.

    Args:
        job: The persisted job row.
        page: Recognition result.
        original_size: ``(width, height)`` of the uploaded image before resizing.

    Returns:
        A fully populated ``OcrResponse``.

    Note: ``page.width``/``height`` are the preprocessing dimensions, while the
    schema reports the ORIGINAL size because the frontend renders the original
    bytes. Boxes are already inverse-scaled, so the two agree.
    """
    return OcrResponse(
        job_id=job.id,
        engine=page.engine,
        text=page.text,
        lines=[
            LineOut(
                text=line.text,
                bbox=_scaled_bbox(line.box, page.scale),
                confidence=line.confidence,
                words=[
                    WordOut(
                        text=word.text,
                        bbox=_scaled_bbox(word.box, page.scale),
                        confidence=word.confidence,
                    )
                    for word in line.words
                ],
            )
            for line in page.lines
        ],
        page=PageOut(
            width=original_size[0],
            height=original_size[1],
            scale=page.scale,
            line_count=len(page.lines),
        ),
        mean_confidence=page.mean_confidence,
        devanagari_ratio=page.devanagari_ratio,
        duration_ms=page.duration_ms,
        created_at=job.created_at,
    )


def store_upload(upload: ValidatedUpload, storage: StorageBackend) -> str:
    """Persist an upload under a content-addressed key.

    Args:
        upload: Validated bytes.
        storage: Target backend.

    Returns:
        The storage key.

    Why content-addressed: identical files dedupe for free, and the key cannot
    be influenced by the client's filename (no traversal, no collisions).
    """
    extension = _EXTENSION_BY_KIND.get(upload.kind, "bin")
    key = f"uploads/{upload.sha256[:2]}/{upload.sha256}.{extension}"
    storage.put(key, upload.data, content_type=f"image/{upload.kind}")
    return key


def process_image_upload(
    data: bytes,
    *,
    engine: OcrEngine,
    settings: Settings,
    storage: StorageBackend,
    session: Session,
    client_hash: str = "",
    retain_image: bool = True,
) -> ProcessedImage:
    """Validate, store, recognize and persist one image.

    Args:
        data: Raw upload bytes.
        engine: OCR backend.
        settings: Runtime configuration.
        storage: Upload backend.
        session: Database session for the job row.
        client_hash: Hashed client identifier for abuse analysis.
        retain_image: Keep the upload so the viewer can display it.

    Returns:
        A ``ProcessedImage`` with the persisted job id and result.

    Raises:
        VarnamError: Any typed validation or engine failure. The route maps it
            to a status code; nothing is swallowed here.
    """
    if not engine.is_ready():
        raise EngineUnavailableError("The OCR model is still warming up. Retry in a moment.")

    upload = validate_upload(data, max_bytes=settings.max_upload_mb * 1_048_576)
    storage_key = store_upload(upload, storage) if retain_image else None

    prepared = prepare_image(upload.data, max_long_edge=settings.max_image_long_edge)

    # Blocking acquire with a deadline: better a fast 503 than a request that
    # hangs until the platform times out at the edge with an opaque error.
    acquired = _INFERENCE_LOCK.acquire(timeout=settings.request_timeout_seconds)
    if not acquired:
        from app.core.errors import EngineBusyError  # noqa: PLC0415 - avoid cycle at import time

        raise EngineBusyError("The OCR engine is busy with another page. Retry shortly.")

    try:
        with Timer() as timer:
            page = run_page(prepared.gray, engine, scale=prepared.scale)
    finally:
        _INFERENCE_LOCK.release()

    job = OcrJob(
        status=JobStatus.SUCCEEDED,
        engine=page.engine,
        content_kind=upload.kind,
        page_count=1,
        storage_key=storage_key,
        result={
            "text": page.text,
            "lines": [
                {
                    "text": line.text,
                    "bbox": [line.box.x0, line.box.y0, line.box.x1, line.box.y1],
                    "confidence": line.confidence,
                }
                for line in page.lines
            ],
            "page": {"width": prepared.original_size[0], "height": prepared.original_size[1]},
            # Stored so a result can be re-rendered later without the image: the
            # frontend needs the original dimensions to place boxes correctly.
            "original_size": list(prepared.original_size),
            "mean_confidence": page.mean_confidence,
            "devanagari_ratio": page.devanagari_ratio,
        },
        duration_ms=timer.elapsed_ms,
        client_hash=client_hash,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    logger.info(
        "ocr_succeeded",
        job_id=job.id,
        engine=page.engine,
        lines=len(page.lines),
        confidence=page.mean_confidence,
        devanagari_ratio=page.devanagari_ratio,
        duration_ms=timer.elapsed_ms,
        engine_ms=page.duration_ms,
    )

    # Keep the original size on the job so the response mapper is stateless.
    return ProcessedImage(job_id=job.id, page=page)


def process_pdf_upload(
    data: bytes,
    *,
    engine: OcrEngine,
    settings: Settings,
    storage: StorageBackend,
    session: Session,
    client_hash: str = "",
) -> str:
    """Create a QUEUED job for a PDF and return its id.

    Args:
        data: Raw PDF bytes.
        engine: OCR backend.
        settings: Runtime configuration.
        storage: Upload backend.
        session: Database session.
        client_hash: Hashed client identifier.

    Returns:
        The new job id. Rendering and inference happen in a background task so
        the HTTP response returns immediately - a required property because a
        10-page scan takes far longer than any free-tier request timeout.
    """
    from app.services.validation import count_pdf_pages  # noqa: PLC0415 - optional backend

    upload = validate_upload(data, max_bytes=settings.max_upload_mb * 1_048_576)
    page_count = count_pdf_pages(upload.data, max_pages=settings.max_pages_per_job)
    storage_key = store_upload(upload, storage)

    job = OcrJob(
        status=JobStatus.QUEUED,
        engine=engine.name,
        content_kind="pdf",
        page_count=page_count,
        storage_key=storage_key,
        client_hash=client_hash,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    logger.info("pdf_job_queued", job_id=job.id, pages=page_count)
    return job.id


def process_pdf_job(job_id: str, *, settings: Settings, storage: StorageBackend, engine: OcrEngine) -> None:
    """Render and recognize every page of a queued PDF job.

    Args:
        job_id: Job to process.
        settings: Runtime configuration.
        storage: Backend holding the PDF.
        engine: OCR backend.

    Why this opens its own session: it runs after the response has been sent, so
    the request-scoped session is already closed. Reusing it would raise
    ``ResourceClosedError`` in a background thread.
    """
    from app.db import get_engine
    from app.services.pdf import render_pages
    from sqlmodel import Session as ModelSession

    with ModelSession(get_engine()) as session:
        job = session.get(OcrJob, job_id)
        if job is None:
            logger.warning("pdf_job_missing", job_id=job_id)
            return

        job.status = JobStatus.PROCESSING
        session.add(job)
        session.commit()

        try:
            if job.storage_key is None:
                raise ValueError("Job has no stored file.")
            key = job.storage_key.split("://", 1)[-1]
            pdf_bytes = storage.get(key)
            pages = render_pages(pdf_bytes, max_pages=settings.max_pages_per_job)

            texts: list[str] = []
            page_payloads: list[dict] = []
            total_ms = 0.0
            confidences: list[float] = []

            for index, png in enumerate(pages, start=1):
                prepared = prepare_image(png, max_long_edge=settings.max_image_long_edge)
                page = run_page(prepared.gray, engine, scale=prepared.scale)
                total_ms += page.duration_ms
                texts.append(page.text)
                confidences.append(page.mean_confidence)
                page_payloads.append(
                    {
                        "index": index,
                        "text": page.text,
                        "lines": [
                            {
                                "text": line.text,
                                "bbox": [line.box.x0, line.box.y0, line.box.x1, line.box.y1],
                                "confidence": line.confidence,
                            }
                            for line in page.lines
                        ],
                    }
                )
                logger.info("pdf_page_done", job_id=job_id, page=index, of=len(pages))

            job.status = JobStatus.SUCCEEDED
            job.result = {
                "text": "\n\n".join(texts),
                "pages": page_payloads,
                "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            }
            job.duration_ms = total_ms
        except Exception as exc:  # noqa: BLE001 - must persist a terminal state
            # Any failure inside a background task is invisible to the client
            # unless it is written to the row, so we always record it.
            job.status = JobStatus.FAILED
            job.error_message = str(exc)[:1000]
            logger.exception("pdf_job_failed", job_id=job_id, error=str(exc))
        finally:
            from datetime import UTC, datetime

            job.finished_at = datetime.now(UTC)
            session.add(job)
            session.commit()
