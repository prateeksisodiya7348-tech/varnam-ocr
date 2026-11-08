"""Tests for upload validation - the security boundary."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.core.errors import InvalidUploadError, PayloadTooLargeError, UnsupportedMediaError
from app.services.validation import sniff_kind, validate_upload


def _png_bytes(size: tuple[int, int] = (64, 32)) -> bytes:
    """Build a minimal valid PNG for tests."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_sniff_detects_png_and_jpeg() -> None:
    """Magic bytes, not extensions, decide the type."""
    assert sniff_kind(_png_bytes()) == "png"
    assert sniff_kind(b"\xff\xd8\xff\xe0fake jpeg body") == "jpeg"
    assert sniff_kind(b"%PDF-1.7\n") == "pdf"


def test_sniff_rejects_text() -> None:
    """A text file masquerading as an image is refused."""
    with pytest.raises(UnsupportedMediaError):
        sniff_kind(b"<html>definitely not an image</html>")


def test_riff_must_be_webp() -> None:
    """A RIFF container that is not WEBP (e.g. WAV) is rejected."""
    with pytest.raises(UnsupportedMediaError):
        sniff_kind(b"RIFF0000WAVEfmt ")


def test_validate_accepts_valid_png() -> None:
    """Valid uploads pass through with a stable content hash."""
    result = validate_upload(_png_bytes(), max_bytes=1024 * 1024)
    assert result.kind == "png"
    assert result.size_bytes == len(result.data)
    assert len(result.sha256) == 64


def test_validate_rejects_empty() -> None:
    """An empty body is an invalid upload, not a 500."""
    with pytest.raises(InvalidUploadError):
        validate_upload(b"", max_bytes=1024)


def test_validate_enforces_byte_cap() -> None:
    """Oversized payloads fail before any decoding work happens."""
    with pytest.raises(PayloadTooLargeError):
        validate_upload(_png_bytes() + b"\x00" * 2048, max_bytes=1024)


def test_validate_rejects_truncated_image() -> None:
    """A truncated PNG (valid header, no pixel data) is caught up front."""
    data = _png_bytes()
    with pytest.raises(InvalidUploadError):
        validate_upload(data[: len(data) // 2], max_bytes=1024 * 1024)


def test_validate_rejects_tiny_image() -> None:
    """Sub-8px images cannot contain text and are rejected explicitly."""
    with pytest.raises(InvalidUploadError):
        validate_upload(_png_bytes((4, 4)), max_bytes=1024 * 1024)
