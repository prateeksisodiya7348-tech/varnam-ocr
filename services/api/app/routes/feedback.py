"""Feedback endpoint - the active-learning capture point.

Every correction a user makes is a labelled training example from the exact
distribution the model fails on. That is more valuable than any public dataset,
because it is the real deployment distribution. This endpoint is therefore a
product feature, not a support form.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlmodel import Session

from app.db import get_session
from app.dependencies import ServicesDep
from app.models.db import FeedbackSample, JobStatus, OcrJob
from app.models.schemas import ErrorResponse, FeedbackRequest, FeedbackResponse
from app.core.errors import NotFoundError, InvalidUploadError
from devanagari_text.normalize import normalize_text

router = APIRouter(prefix="/v1", tags=["feedback"])

SessionDep = Annotated[Session, Depends(get_session)]


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Store a human correction for a recognized line",
)
def create_feedback(
    request: Request,
    body: FeedbackRequest,
    services: ServicesDep,
    session: SessionDep,
) -> FeedbackResponse:
    """Persist a corrected line, pairing it with the model's prediction.

    Args:
        request: Incoming request (for metrics only).
        body: Job id, line index and the corrected text.
        services: Service container.
        session: Database session.

    Returns:
        The id of the stored sample.

    Raises:
        NotFoundError: Unknown job.
        InvalidUploadError: Line index out of range, or the job has no result.

    Why validate the index: a stale client (the user edited a result that was
    re-run) would otherwise attach a correction to the wrong line, silently
    poisoning the training set. Rejecting is cheaper than reconciling later.
    """
    job = session.get(OcrJob, body.job_id)
    if job is None:
        raise NotFoundError(f"No job with id {body.job_id}.")
    if job.status != JobStatus.SUCCEEDED or not job.result:
        raise InvalidUploadError("This job has no successful result to correct.")

    lines = job.result.get("lines", [])
    if body.line_index >= len(lines):
        raise InvalidUploadError(
            f"line_index {body.line_index} is out of range for a {len(lines)}-line result."
        )

    predicted = lines[body.line_index].get("text", "")
    # Normalize the correction exactly like model output so the training corpus
    # has one canonical form. This is the shared module doing its job.
    corrected = normalize_text(body.corrected_text, keep_line_breaks=False)

    sample = FeedbackSample(
        job_id=body.job_id,
        line_index=body.line_index,
        predicted=predicted,
        corrected=corrected,
    )
    session.add(sample)
    session.commit()
    session.refresh(sample)

    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.inc("feedback_samples_total")

    return FeedbackResponse(stored=True, sample_id=sample.id or 0)
