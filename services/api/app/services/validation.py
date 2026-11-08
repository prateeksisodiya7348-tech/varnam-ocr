"""Upload validation.

Security posture: the browser-supplied filename and ``Content-Type`` are both
attacker-controlled, so neither is trusted. We decide the real type from magic
bytes and then bound the decoded pixel count, which protects against image
decompression bombs (a 2KB PNG that expands to 50000x50000 pixels and OOMs a
512MB container).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Literal

from PIL import Image, UnidentifiedImageError

from app.core.errors import InvalidUploadError, PayloadTooLargeError, UnsupportedMediaError

ImageKind = Literal["png", "jpeg", "webp", "pdf"]

# Signatures are checked on the first bytes of the upload. Deliberately short
# tuples: we only need enough to disambiguate, and reading more wastes memory.
_MAGIC_SIGNATURES: tuple[tuple[bytes, ImageKind], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"RIFF", "webp"),  # refined by checking bytes 8:12 below
    (b"%PDF-", "pdf"),
)

# Pillow must reject absurd dimensions while decoding, not after.
Image.MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    """A file that passed every validation rule.

    Attributes:
        kind: Detected content type, from magic bytes.
        data: The original bytes, unmodified.
        size_bytes: ``len(data)``, cached because it is logged repeatedly.
        sha256: Content hash used for dedupe and as the storage key.
    """

    kind: ImageKind
    data: bytes
    size_bytes: int
    sha256: str


def sniff_kind(data: bytes) -> ImageKind:
    """Identify the content type from magic bytes.

    Args:
        data: Raw uploaded bytes.

    Returns:
        The detected kind.

    Raises:
        UnsupportedMediaError: When no signature matches.

    Why: extension and ``Content-Type`` are trivially spoofed; a PNG renamed to
    ``.pdf`` must not reach the PDF branch.
    """
    for signature, kind in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            # RIFF is a container format (also used by WAV); confirm "WEBP".
            if kind == "webp" and data[8:12] != b"WEBP":
                continue
            return kind
    raise UnsupportedMediaError(
        "Unsupported file type. Upload a PNG, JPEG, WEBP image or a PDF.",
    )


def validate_upload(data: bytes, *, max_bytes: int) -> ValidatedUpload:
    """Validate size, type and decodability of an upload.

    Args:
        data: Raw bytes received from the client.
        max_bytes: Hard byte ceiling from settings.

    Returns:
        A ``ValidatedUpload`` ready to persist.

    Raises:
        PayloadTooLargeError: When the payload exceeds ``max_bytes``.
        UnsupportedMediaError: When the type is not supported.
        InvalidUploadError: When an image cannot be decoded or is degenerate.

    Why validate up front: every later stage (storage write, model inference)
    is expensive; rejecting a bad file at the edge keeps the 512MB instance
    from spending its budget on garbage.
    """
    if not data:
        raise InvalidUploadError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"File is {len(data) / 1_048_576:.1f}MB; the limit is {max_bytes / 1_048_576:.0f}MB."
        )

    kind = sniff_kind(data)

    if kind != "pdf":
        _assert_decodable_image(data)

    return ValidatedUpload(
        kind=kind,
        data=data,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _assert_decodable_image(data: bytes) -> tuple[int, int]:
    """Fully decode an image to prove it is not truncated or malformed.

    Args:
        data: Encoded image bytes.

    Returns:
        The ``(width, height)`` of the decoded image.

    Raises:
        InvalidUploadError: When Pillow cannot decode the bytes.

    Note: ``Image.open`` is lazy, so a corrupt file would otherwise only fail
    deep inside the pipeline after we had already stored it. ``verify()`` plus a
    full reload is the only reliable check.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            probe.verify()
        # verify() leaves the object unusable, so reopen to confirm the pixel
        # data actually decodes (catches truncated JPEGs that pass verify()).
        with Image.open(io.BytesIO(data)) as check:
            check.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidUploadError("The image is corrupted or truncated.") from exc

    if width < 8 or height < 8:
        raise InvalidUploadError("The image is too small to contain readable text.")
    return width, height


def count_pdf_pages(data: bytes, *, max_pages: int) -> int:
    """Count pages in a PDF without rendering it.

    Args:
        data: Raw PDF bytes.
        max_pages: Ceiling from settings.

    Returns:
        The number of pages.

    Raises:
        PayloadTooLargeError: When the document exceeds ``max_pages``.
        UnsupportedMediaError: When no PDF backend is installed.

    Why lazy-import pypdfium2: the PDF backend is an optional extra so the
    default image is small. If it is not installed we fail with a precise,
    actionable message instead of a 500.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise UnsupportedMediaError(
            "PDF support is not enabled on this deployment. "
            "Upload page images instead, or install the 'pdf' extra."
        ) from exc

    document = pdfium.PdfDocument(io.BytesIO(data))
    page_count = len(document)
    document.close()

    if page_count > max_pages:
        raise PayloadTooLargeError(f"PDF has {page_count} pages; the limit is {max_pages}.")
    if page_count == 0:
        raise InvalidUploadError("The PDF contains no pages.")
    return page_count
