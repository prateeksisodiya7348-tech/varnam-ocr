"""Tests for the NumPy segmentation layer.

These run with no model, no disk and no network - the entire reason
segmentation was kept dependency-free.
"""

from __future__ import annotations

import numpy as np

from app.services.preprocess import binarize, ensure_dark_text
from app.services.segmentation import (
    Box,
    LineCandidate,
    segment_lines,
    segment_words,
    sort_reading_order,
    split_columns,
)


def _page_with_bars(rows: list[tuple[int, int, int, int]], size: tuple[int, int] = (400, 300)) -> np.ndarray:
    """Create a white page with black bars.

    Args:
        rows: ``(x0, y0, x1, y1)`` rectangles to fill black.
        size: ``(height, width)`` of the page.

    Returns:
        Boolean ink mask.
    """
    height, width = size
    page = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in rows:
        page[y0:y1, x0:x1] = True
    return page


def test_binarize_separates_ink_from_paper() -> None:
    """Otsu splits a bimodal image cleanly."""
    gray = np.full((50, 50), 250, dtype=np.uint8)
    gray[10:20, 10:40] = 10
    mask = binarize(gray)
    assert mask[15, 20]
    assert not mask[0, 0]


def test_ensure_dark_text_inverts_dark_pages() -> None:
    """A dark-background page is flipped so text is always ink."""
    dark = np.full((40, 40), 20, dtype=np.uint8)
    assert ensure_dark_text(dark).mean() > 200
    light = np.full((40, 40), 240, dtype=np.uint8)
    assert ensure_dark_text(light).mean() == 240


def test_segment_lines_finds_each_bar() -> None:
    """Four bars must produce four line boxes, top to bottom."""
    mask = _page_with_bars([(50, 40, 350, 70), (50, 110, 350, 140), (50, 180, 350, 210)])
    lines = segment_lines(mask)
    assert len(lines) == 3
    assert lines[0].y0 < lines[1].y0 < lines[2].y0
    # Boxes must be tight around the ink, not the full page width.
    assert lines[0].x0 >= 45
    assert lines[0].x1 <= 355


def test_segment_lines_ignores_specks() -> None:
    """A single noisy row must not become a text line."""
    mask = _page_with_bars([(50, 100, 350, 130)])
    mask[20, 200] = True
    assert len(segment_lines(mask)) == 1


def test_segment_words_finds_gaps() -> None:
    """Two ink blocks separated by a wide gap become two words."""
    mask = _page_with_bars([(40, 50, 120, 80), (200, 50, 300, 80)])
    # Build the line box by hand so the test is independent of segment_lines.
    line = Box(x0=40, y0=50, x1=300, y1=80)
    words = segment_words(mask, line)
    assert len(words) == 2
    assert words[0].x1 <= words[1].x0


def test_reading_order_sorts_by_row_then_column() -> None:
    """Lines on the same visual row are ordered left to right."""
    top_left = LineCandidate(box=Box(10, 10, 100, 40))
    top_right = LineCandidate(box=Box(200, 12, 300, 42))
    bottom = LineCandidate(box=Box(10, 100, 300, 130))

    ordered = sort_reading_order([bottom, top_right, top_left])
    assert [line.box.x0 for line in ordered] == [10, 200, 10]
    assert ordered[0].box.y0 == 10


def test_reading_order_handles_empty_input() -> None:
    """Empty input is a valid page, not an error."""
    assert sort_reading_order([]) == []


def test_split_columns_detects_a_gutter() -> None:
    """A page-wide vertical gap produces two column groups."""
    left = [LineCandidate(box=Box(10, 10 + i * 40, 180, 40 + i * 40)) for i in range(5)]
    right = [LineCandidate(box=Box(240, 10 + i * 40, 420, 40 + i * 40)) for i in range(5)]
    columns = split_columns(left + right, page_width=430)
    assert len(columns) == 2
    assert all(line.box.x1 <= 200 for line in columns[0])
    assert all(line.box.x0 >= 200 for line in columns[1])


def test_split_columns_keeps_single_column_pages_intact() -> None:
    """Full-width lines are never split, even with ragged right edges."""
    lines = [LineCandidate(box=Box(10, 10 + i * 40, 400, 40 + i * 40)) for i in range(5)]
    assert len(split_columns(lines, page_width=420)) == 1
