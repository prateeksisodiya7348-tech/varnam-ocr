"""Pydantic request/response schemas.

These are the public contract. They are separated from the SQLModel tables so
that internal columns (``client_hash``, ``storage_key``) can never leak into an
API response by accident - which is a real risk when reusing ORM classes as
response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """Pixel rectangle in *original* upload coordinates."""

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)


class WordOut(BaseModel):
    """A word with geometry and confidence."""

    text: str
    bbox: BBox
    confidence: float = Field(ge=0, le=1)


class LineOut(BaseModel):
    """A recognized line."""

    text: str
    bbox: BBox
    confidence: float = Field(ge=0, le=1)
    words: list[WordOut] = Field(default_factory=list)


class PageOut(BaseModel):
    """Metadata about the processed page."""

    width: int
    height: int
    scale: float
    line_count: int


class OcrResponse(BaseModel):
    """Result of a successful OCR request."""

    job_id: str
    status: Literal["succeeded"] = "succeeded"
    engine: str
    text: str
    lines: list[LineOut]
    page: PageOut
    mean_confidence: float = Field(ge=0, le=1)
    devanagari_ratio: float = Field(ge=0, le=1)
    duration_ms: float
    created_at: datetime


class JobAcceptedResponse(BaseModel):
    """Returned when a PDF was accepted for asynchronous processing."""

    job_id: str
    status: Literal["queued"] = "queued"
    page_count: int
    poll_url: str


class JobStatusResponse(BaseModel):
    """Polling payload for an asynchronous job."""

    job_id: str
    status: Literal["queued", "processing", "succeeded", "failed"]
    page_count: int
    engine: str
    error: str | None = None
    result: OcrResponse | None = None


class FeedbackRequest(BaseModel):
    """A user's correction of one recognized line."""

    model_config = ConfigDict(str_strip_whitespace=True)

    job_id: str
    line_index: int = Field(ge=0)
    corrected_text: str = Field(min_length=1, max_length=4000)


class FeedbackResponse(BaseModel):
    """Acknowledgement that a correction was stored."""

    stored: bool
    sample_id: int


class HealthResponse(BaseModel):
    """Liveness / readiness payload."""

    status: Literal["ok", "degraded"]
    engine: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class ErrorBody(BaseModel):
    """Machine-readable error payload."""

    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Envelope used for every non-2xx response."""

    error: ErrorBody
