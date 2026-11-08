"""PDF page rendering.

Isolated in its own module so the optional ``pypdfium2`` dependency is confined
to one import. PDF is genuinely harder than images here: a page can be A0 at
300 DPI, which is far beyond the memory budget of a free instance, so the
render scale is derived from the page size rather than hardcoded.
"""

from __future__ import annotations

import io

from app.core.errors import InvalidUploadError, UnsupportedMediaError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Target long edge for a rendered page. Matches MAX_IMAGE_LONG_EDGE so the rest
# of the pipeline sees a consistent pixel budget regardless of input format.
_TARGET_LONG_EDGE = 2000


def render_pages(data: bytes, *, max_pages: int) -> list[bytes]:
    """Render every PDF page to PNG bytes.

    Args:
        data: Raw PDF bytes.
        max_pages: Hard page ceiling.

    Returns:
        PNG-encoded pages in order.

    Raises:
        UnsupportedMediaError: When ``pypdfium2`` is not installed.
        InvalidUploadError: When the PDF cannot be parsed.
        PayloadTooLargeError: When the document exceeds ``max_pages``.

    Why PNG and not JPEG: PNG is lossless, and JPEG ringing artifacts around
    Devanagari strokes measurably hurt recognition. The pages are short-lived,
    so the size cost is acceptable.
    """
    try:
        import pypdfium2 as pdfium  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise UnsupportedMediaError(
            "PDF support is not enabled on this deployment. Upload page images instead."
        ) from exc

    from app.core.errors import PayloadTooLargeError  # noqa: PLC0415 - narrow scope

    try:
        document = pdfium.PdfDocument(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - the backend raises its own types
        raise InvalidUploadError("The PDF could not be parsed.") from exc

    try:
        page_count = len(document)
        if page_count == 0:
            raise InvalidUploadError("The PDF contains no pages.")
        if page_count > max_pages:
            raise PayloadTooLargeError(f"PDF has {page_count} pages; the limit is {max_pages}.")

        pages: list[bytes] = []
        for index in range(page_count):
            page = document[index]
            width, height = page.get_size()
            # pdfium renders at 72 DPI at scale 1.0, so this ratio lands the long
            # edge on our target regardless of the physical page size.
            scale = _TARGET_LONG_EDGE / max(1.0, max(width, height))
            # Clamp: a tiny page must not be upscaled into blur, and a huge one
            # must not be rendered at a scale that allocates gigabytes.
            scale = max(1.0, min(scale, 6.0))
            bitmap = page.render(scale=scale)
            pages.append(bitmap.to_pil().convert("RGB").save_to_bytes if False else _to_png(bitmap))
            bitmap.close()
            page.close()
        return pages
    finally:
        document.close()


def _to_png(bitmap) -> bytes:
    """Encode a pdfium bitmap as PNG.

    Args:
        bitmap: A ``PdfBitmap`` from pypdfium2.

    Returns:
        PNG bytes.
    """
    import io as _io  # noqa: PLC0415 - local alias keeps the module import surface tiny

    buffer = _io.BytesIO()
    bitmap.to_pil().convert("RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()
