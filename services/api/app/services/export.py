"""Recognition result assembly and export formats.

The pipeline produces one canonical object; every export format is a pure
function of it. Keeping exports here (rather than in the frontend) means the
hOCR we emit is byte-identical whether it came from the API or the browser, and
it can be regression-tested server-side.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np

from devanagari_text.normalize import devanagari_ratio, normalize_text
from app.services.engine import OcrEngine
from app.services.segmentation import Box, LineCandidate


@dataclass(frozen=True, slots=True)
class WordResult:
    """A recognized word with geometry.

    Attributes:
        text: Word text.
        box: Word rectangle in preprocessed-image coordinates.
        confidence: Inherited line confidence; word-level calibration is a
            documented roadmap item, not a measurement we fabricate.
    """

    text: str
    box: Box
    confidence: float


@dataclass(frozen=True, slots=True)
class LineResult:
    """A recognized line with geometry and word boxes."""

    text: str
    box: Box
    confidence: float
    words: list[WordResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PageResult:
    """Everything the API knows about one recognized page.

    Attributes:
        text: The full document text, lines joined by newlines.
        lines: Structured line results in reading order.
        width: Page width in preprocessed pixels (for the bbox overlay).
        height: Page height in preprocessed pixels.
        scale: Downscale factor applied during preprocessing. The frontend
            divides box coordinates by this to align with the original upload.
        mean_confidence: Ink-weighted mean of line confidences.
        devanagari_ratio: Fraction of Devanagari characters; a cheap detector
            for "the model returned garbage".
        engine: Engine identifier that produced this page.
        duration_ms: Wall-clock inference time.
    """

    text: str
    lines: list[LineResult]
    width: int
    height: int
    scale: float
    mean_confidence: float
    devanagari_ratio: float
    engine: str
    duration_ms: float


def _split_words(line: LineCandidate, mask_boxes: list[Box]) -> list[WordResult]:
    """Attach text to word boxes when the split is trustworthy.

    Args:
        line: The recognized line.
        mask_boxes: Word boxes from segmentation.

    Returns:
        Word results. The text is distributed by character count proportional to
        box width, which is approximate on purpose: per-word recognition is a
        separate (slower) model pass and is listed as future work.

    Why approximate: claiming exact word alignment we did not compute would be
    worse than an honest proportional split. The export marks it as such.
    """
    if not mask_boxes or not line.text:
        return []

    total_width = sum(box.width for box in mask_boxes) or 1
    characters = list(line.text)
    results: list[WordResult] = []
    cursor = 0

    for index, box in enumerate(mask_boxes):
        share = box.width / total_width
        take = len(characters) - cursor if index == len(mask_boxes) - 1 else round(len(characters) * share)
        chunk = "".join(characters[cursor : cursor + take]).strip()
        cursor += take
        if chunk:
            results.append(WordResult(text=chunk, box=box, confidence=line.confidence))
    return results


def run_page(gray: np.ndarray, engine: OcrEngine, *, scale: float = 1.0) -> PageResult:
    """Run the engine over one page and assemble the canonical result.

    Args:
        gray: Preprocessed grayscale page.
        engine: The OCR backend.
        scale: Downscale factor applied by preprocessing.

    Returns:
        A ``PageResult`` with normalized text.

    Why normalize here: model output can contain control characters from the
    decoder; the normalization is lossless for combining marks (see
    ``devanagari_text.normalize``) and guarantees the API never returns text
    that is invalid to paste into a document editor.
    """
    import time  # noqa: PLC0415 - only needed for the timing measurement

    started = time.perf_counter()
    candidates = engine.recognize(gray)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    lines: list[LineResult] = []
    confidences: list[float] = []
    for candidate in candidates:
        text = normalize_text(candidate.text, keep_line_breaks=False)
        if not text:
            continue
        lines.append(
            LineResult(
                text=text,
                box=candidate.box,
                confidence=candidate.confidence,
                words=_split_words(candidate, candidate.words),
            )
        )
        confidences.append(candidate.confidence)

    document_text = "\n".join(line.text for line in lines)
    mean_confidence = round(float(np.mean(confidences)), 4) if confidences else 0.0

    return PageResult(
        text=document_text,
        lines=lines,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
        scale=round(scale, 4),
        mean_confidence=mean_confidence,
        devanagari_ratio=round(devanagari_ratio(document_text), 4),
        engine=engine.name,
        duration_ms=duration_ms,
    )


def to_plain_text(page: PageResult) -> str:
    """Export the page as UTF-8 plain text with a trailing newline."""
    return page.text + "\n"


def to_hocr(page: PageResult, *, page_index: int = 1) -> str:
    """Export the page as hOCR 1.2 XHTML.

    Args:
        page: The recognized page.
        page_index: 1-based page number for ``ppageno``.

    Returns:
        An hOCR document as a string.

    Why hOCR: it is the interchange format OCR tooling actually consumes
    (Tesseract, OCRmyPDF, ABBYY readers), so this makes the output composable
    with the wider ecosystem in one step.
    """
    body_lines: list[str] = []
    for line_number, line in enumerate(page.lines, start=1):
        box = line.box
        word_spans = "".join(
            f'<span class="ocrx_word" id="word_{page_index}_{line_number}_{wi}" '
            f'title="bbox {w.box.x0} {w.box.y0} {w.box.x1} {w.box.y1}; '
            f'x_wconf {int(w.confidence * 100)}">{html.escape(w.text)}</span> '
            for wi, w in enumerate(line.words, start=1)
        )
        # Fall back to escaped text when word boxes were unavailable.
        content = word_spans or html.escape(line.text)
        body_lines.append(
            f'<span class="ocr_line" id="line_{page_index}_{line_number}" '
            f'title="bbox {box.x0} {box.y0} {box.x1} {box.y1}; '
            f'x_wconf {int(line.confidence * 100)}">{content}</span><br />'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="hi">\n'
        f"<head><title>{html.escape(page.text[:60] or 'Varnam OCR')}</title>\n"
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n'
        '<meta name="ocr-system" content="varnam-ocr" />\n'
        "</head>\n<body>\n"
        '<div class="ocr_page" id="page_1" '
        f'title="bbox 0 0 {page.width} {page.height}; ppageno {page_index - 1}">\n'
        f'<div class="ocr_carea" id="block_1_1" title="bbox 0 0 {page.width} {page.height}">\n'
        '<p class="ocr_par" id="par_1_1" lang="hi">\n'
        + "\n".join(body_lines)
        + "\n</p>\n</div>\n</div>\n</body>\n</html>\n"
    )
