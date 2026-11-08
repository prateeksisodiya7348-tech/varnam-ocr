"""Layout segmentation and reading-order reconstruction.

Devanagari adds a wrinkle that Latin OCR does not have: the *shirorekha*, the
continuous horizontal headline that joins most characters in a word. It makes
words look like solid rectangles connected by a single dark line, which breaks
naive connected-component word splitting. The projection-profile approach used
here survives that because it works on ink density per row/column rather than
on connectivity.

Everything in this module is deterministic, CPU-only and dependency-free
beyond NumPy - which is exactly what lets it run on a 1 vCPU free instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned rectangle in pixel coordinates.

    Coordinates are inclusive-exclusive ``[x0, x1)`` x ``[y0, y1)``, matching
    NumPy slicing so crops need no off-by-one adjustment.

    Attributes:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        """Rectangle width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Rectangle height in pixels."""
        return self.y1 - self.y0

    @property
    def center_y(self) -> float:
        """Vertical center, used for row clustering."""
        return (self.y0 + self.y1) / 2.0

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        """Serialize for the API response."""
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(slots=True)
class LineCandidate:
    """A detected text line with its word boxes and recognized content.

    Attributes:
        box: The line bounding box in preprocessed-image coordinates.
        words: Word-level boxes, same coordinate space.
        text: Recognized (already normalized) line text.
        confidence: Mean character probability in [0, 1].
    """

    box: Box
    words: list[Box] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0


def _runs(mask: np.ndarray, *, min_length: int) -> list[tuple[int, int]]:
    """Find contiguous True runs in a 1-D boolean mask.

    Args:
        mask: 1-D boolean array.
        min_length: Runs shorter than this are discarded as noise.

    Returns:
        List of ``(start, end)`` index pairs, end-exclusive.

    Why vectorized: a Python loop over 2000 rows is fine, but this same helper
    serves column projections of width 2000+; ``np.diff`` on the padded mask
    keeps it O(n) with no interpreter overhead.
    """
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= min_length
    ]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average used to damp single-pixel noise.

    Args:
        values: 1-D numeric array.
        window: Odd window size; forced odd so the average is centered.

    Returns:
        Smoothed array of the same length.

    Why: a single speck of dust can create a spurious ink row, splitting one
    text line into two. A 3-5px average removes that without merging real lines.
    """
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def segment_lines(mask: np.ndarray, *, min_height: int = 6) -> list[Box]:
    """Split a page mask into horizontal text lines.

    Args:
        mask: Boolean ink mask, True = text.
        min_height: Minimum line thickness in pixels; filters specks and rules.

    Returns:
        Line boxes top-to-bottom.

    Why projection profiles: for documents with a consistent baseline (printed
    text, forms, receipts) it is more reliable and 100x cheaper than a learned
    detector, and it needs no detector model on a memory-constrained host.
    """
    if mask.size == 0:
        return []

    row_ink = mask.sum(axis=1).astype(np.float64)
    smoothed = _smooth(row_ink, 3)

    # A row is "text" when it holds at least 0.5% of the page width in ink.
    # Below that, we are looking at noise or a border artifact.
    threshold = max(1.0, mask.shape[1] * 0.005)
    text_rows = smoothed > threshold

    raw_runs = _runs(text_rows, min_length=min_height)
    if not raw_runs:
        return []

    # Merge runs separated by a gap smaller than ~60% of the median line height.
    # Devanagari matras above and the shirorekha often create a 1-3px gap inside
    # what is visually one line; merging keeps "किताब" as one line, not two.
    heights = [end - start for start, end in raw_runs]
    merge_gap = max(2, int(np.median(heights) * 0.6))

    merged: list[list[int]] = [list(raw_runs[0])]
    for start, end in raw_runs[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Horizontal extent per line, so a box never spans empty page margins.
    boxes: list[Box] = []
    for y0, y1 in merged:
        band = mask[y0:y1, :]
        columns = np.flatnonzero(band.any(axis=0))
        if columns.size == 0:
            continue
        boxes.append(Box(x0=int(columns[0]), y0=int(y0), x1=int(columns[-1]) + 1, y1=int(y1)))
    return boxes


def segment_words(mask: np.ndarray, line: Box) -> list[Box]:
    """Split a line box into word boxes using vertical gaps.

    Args:
        mask: Boolean ink mask for the whole page.
        line: The line box to subdivide.

    Returns:
        Word boxes left-to-right; may be empty.

    Why the gap threshold scales with line height: Devanagari words are tall
    (headline + matras), so a fixed 5px gap would merge words on large print and
    over-split on small print. ``0.45 * height`` tracks the visual word space
    across font sizes.
    """
    band = mask[line.y0 : line.y1, :]
    if band.size == 0:
        return []

    column_ink = band.sum(axis=0).astype(np.float64)
    threshold = max(1.0, band.shape[0] * 0.05)
    inked = column_ink > threshold

    gap_threshold = max(3, int(line.height * 0.45))
    runs = _runs(inked, min_length=2)

    if not runs:
        return [line]

    grouped: list[list[int]] = [list(runs[0])]
    for start, end in runs[1:]:
        if start - grouped[-1][1] < gap_threshold:
            grouped[-1][1] = end
        else:
            grouped.append([start, end])

    return [
        Box(x0=x0, y0=line.y0, x1=x1, y1=line.y1)
        for x0, x1 in grouped
        if x1 - x0 >= 2
    ]


def sort_reading_order(lines: list[LineCandidate]) -> list[LineCandidate]:
    """Order lines the way a human reads a page.

    Args:
        lines: Unordered line candidates.

    Returns:
        Lines sorted by (row band, then x).

    Algorithm:
        1. Sort by vertical center.
        2. Group into rows: a line joins the current row when its vertical
           center is within 60% of the row's median height. This tolerates the
           baseline jitter of scanned pages.
        3. Sort each row left-to-right.

    Limitation: this is single-column plus stacked-row logic. True multi-column
    detection needs a learned layout model; the heuristic here splits columns
    only when a page-wide vertical gutter is visible (see ``split_columns``).
    """
    if not lines:
        return []

    ordered = sorted(lines, key=lambda item: (item.box.center_y, item.box.x0))
    median_height = float(np.median([item.box.height for item in ordered]) or 1.0)
    tolerance = max(4.0, median_height * 0.6)

    rows: list[list[LineCandidate]] = [[ordered[0]]]
    row_center = ordered[0].box.center_y
    for candidate in ordered[1:]:
        if abs(candidate.box.center_y - row_center) <= tolerance:
            rows[-1].append(candidate)
            # Rolling average keeps the row center stable across a long table row.
            row_center = float(np.mean([item.box.center_y for item in rows[-1]]))
        else:
            rows.append([candidate])
            row_center = candidate.box.center_y

    result: list[LineCandidate] = []
    for row in rows:
        result.extend(sorted(row, key=lambda item: item.box.x0))
    return result


def split_columns(lines: list[LineCandidate], page_width: int) -> list[list[LineCandidate]]:
    """Group lines into columns when a vertical gutter exists.

    Args:
        lines: Lines already in reading order.
        page_width: Width of the page in pixels.

    Returns:
        Column groups, left to right. Returns a single group when no clear
        gutter is found, which is the common case for letters and forms.

    Why a gutter detector: two-column newspaper scans are the classic failure
    case where line-by-line OCR interleaves the columns into nonsense. Requiring
    the gap to appear in >70% of lines avoids splitting on a wide table cell.
    """
    if len(lines) < 4:
        return [lines]

    occupied = np.zeros(page_width, dtype=bool)
    for line in lines:
        occupied[line.box.x0 : min(line.box.x1, page_width)] = True

    # Ignore margins: only interior gaps can be gutters.
    first_ink = int(np.argmax(occupied))
    last_ink = page_width - int(np.argmax(occupied[::-1]))

    gaps: list[tuple[int, int]] = []
    for start, end in _runs(~occupied[first_ink:last_ink], min_length=int(page_width * 0.03)):
        gaps.append((start + first_ink, end + first_ink))

    if not gaps:
        return [lines]

    # The widest interior gap is the gutter candidate.
    gutter_start, gutter_end = max(gaps, key=lambda gap: gap[1] - gap[0])
    left = [line for line in lines if line.box.x1 <= gutter_start + 2]
    right = [line for line in lines if line.box.x0 >= gutter_end - 2]
    ambiguous = len(lines) - len(left) - len(right)

    if len(left) < 2 or len(right) < 2 or ambiguous > 0.3 * len(lines):
        return [lines]
    return [left, right]
