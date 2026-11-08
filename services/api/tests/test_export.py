"""Tests for the export formats.

hOCR is a real interchange format, so a broken document is not cosmetic - it
means the output cannot be opened in Tesseract tooling or OCRmyPDF.
"""

from __future__ import annotations

from app.services.export import LineResult, PageResult, WordResult, to_hocr, to_plain_text
from app.services.segmentation import Box


def _page() -> PageResult:
    """Build a small page result with one line and two words."""
    line = LineResult(
        text="नमस्ते दुनिया",
        box=Box(10, 20, 300, 60),
        confidence=0.93,
        words=[
            WordResult(text="नमस्ते", box=Box(10, 20, 140, 60), confidence=0.93),
            WordResult(text="दुनिया", box=Box(160, 20, 300, 60), confidence=0.91),
        ],
    )
    return PageResult(
        text="नमस्ते दुनिया",
        lines=[line],
        width=400,
        height=100,
        scale=1.0,
        mean_confidence=0.93,
        devanagari_ratio=1.0,
        engine="fake",
        duration_ms=12.5,
    )


def test_plain_text_has_trailing_newline() -> None:
    """Text export ends with exactly one newline so concatenation is safe."""
    assert to_plain_text(_page()) == "नमस्ते दुनिया\n"


def test_hocr_contains_required_markup() -> None:
    """hOCR output carries the classes and metadata consumers require."""
    document = to_hocr(_page())
    assert document.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert 'class="ocr_page"' in document
    assert 'class="ocr_line"' in document
    assert 'class="ocrx_word"' in document
    assert "x_wconf 93" in document
    assert "bbox 10 20 300 60" in document


def test_hocr_escapes_user_text() -> None:
    """Text containing markup cannot inject into the document."""
    import html

    line = LineResult(text="<script>", box=Box(0, 0, 10, 10), confidence=0.5, words=[])
    page = PageResult(
        text="<script>",
        lines=[line],
        width=20,
        height=20,
        scale=1.0,
        mean_confidence=0.5,
        devanagari_ratio=0.0,
        engine="fake",
        duration_ms=1.0,
    )
    document = to_hocr(page)
    assert "<script>" not in document.split("<p", 1)[1]
    assert html.escape("<script>") in document
