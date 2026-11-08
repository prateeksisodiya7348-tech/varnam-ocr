"""Persistent tables.

Only two entities are stored, deliberately:

* ``OcrJob``      - the durable record of every request, including the full
                   result JSON so a result can be re-opened without re-running
                   inference (and without keeping the image forever).
* ``FeedbackSample`` - human corrections. This is the active-learning corpus:
                   the highest-value asset the product generates, because it is
                   exactly the distribution of real-world mistakes.

Images are stored only long enough to render the viewer; the text is the product.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    """Timezone-aware UTC now.

    Why not ``datetime.utcnow()``: it returns a naive datetime, which silently
    compares wrong against the aware timestamps Postgres returns.
    """
    return datetime.now(UTC)


class JobStatus(StrEnum):
    """Lifecycle of an OCR job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OcrJob(SQLModel, table=True):
    """One OCR request and its result.

    Attributes:
        id: UUID primary key, used as the public job id.
        status: Current lifecycle state.
        engine: Engine identifier that produced the result.
        content_kind: ``png``/``jpeg``/``webp``/``pdf``.
        page_count: Pages processed.
        storage_key: Where the upload lives, or None when not retained.
        result: Full serialized result (text, boxes, confidences).
        error_message: Populated only when ``status`` is FAILED.
        created_at: Insert time.
        finished_at: Terminal-state time; lets us compute queue latency.
        duration_ms: End-to-end processing time for this job.
        client_hash: Salted hash of the client IP, for abuse analysis without
            storing personal data.
    """

    __tablename__ = "ocr_jobs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status: JobStatus = Field(default=JobStatus.QUEUED, index=True)
    engine: str = Field(default="unknown")
    content_kind: str = Field(default="image")
    page_count: int = Field(default=1)
    storage_key: str | None = Field(default=None)

    # JSON column: the result is read as a whole by the viewer and never queried
    # field-wise, so normalizing it into tables would add joins for no benefit.
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    error_message: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=_utcnow, index=True)
    finished_at: datetime | None = Field(default=None)
    duration_ms: float = Field(default=0.0)
    client_hash: str = Field(default="")


class FeedbackSample(SQLModel, table=True):
    """A human correction of one recognized line.

    Attributes:
        id: Auto-incrementing primary key.
        job_id: The job this correction belongs to.
        line_index: Zero-based line index within the job result.
        predicted: What the model produced.
        corrected: What the user says it should be.
        image_storage_key: Optional crop key, needed to train on the correction.
        created_at: Insert time.
        used_in_training: Set once the sample is folded into a training set, so
            the same correction is never trained on twice.
    """

    __tablename__ = "feedback_samples"

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, foreign_key="ocr_jobs.id")
    line_index: int = Field(default=0)
    predicted: str = Field(sa_column=Column(Text))
    corrected: str = Field(sa_column=Column(Text))
    image_storage_key: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
    used_in_training: bool = Field(default=False, index=True)
