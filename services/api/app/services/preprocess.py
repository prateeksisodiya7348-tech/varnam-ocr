"""Image preprocessing: decode, orient, grayscale, bound the pixel budget.

Every function here is pure and operates on NumPy arrays so the pipeline can be
unit-tested without a model, a disk, or a network. That is the only reason the
OCR tests can run in CI in under a second.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from app.core.errors import InvalidUploadError


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A page decoded and normalized, ready for segmentation.

    Attributes:
        gray: 8-bit single-channel array, shape ``(height, width)``.
        scale: Factor the original was multiplied by (<=1.0 when downscaled).
        original_size: ``(width, height)`` before any resizing. Needed to map
            predicted boxes back onto the user's original image in the viewer.
    """

    gray: np.ndarray
    scale: float
    original_size: tuple[int, int]

    @property
    def height(self) -> int:
        """Pixel height after preprocessing."""
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        """Pixel width after preprocessing."""
        return int(self.gray.shape[1])


def prepare_image(data: bytes, *, max_long_edge: int) -> PreparedImage:
    """Decode upload bytes into a bounded grayscale page.

    Args:
        data: Encoded PNG/JPEG/WEBP bytes.
        max_long_edge: Longest side allowed after resizing.

    Returns:
        A ``PreparedImage``.

    Raises:
        InvalidUploadError: When the bytes are not a decodable image.

    Why grayscale: the recognizer is trained on single-channel input, and color
    adds nothing for black-on-white text while tripling memory. Why resize: an
    unmodified 12MP phone photo is ~48MB as float32 and would OOM a 512MB box;
    2000px on the long edge keeps ~10px tall glyph strokes, which is above the
    32px canvas requirement.
    """
    try:
        with Image.open(io.BytesIO(data)) as source:
            # EXIF orientation is applied before anything else: a sideways scan
            # is the single most common cause of "OCR returned gibberish".
            oriented = ImageOps.exif_transpose(source) or source
            rgb = oriented.convert("RGB")
    except (OSError, ValueError) as exc:
        raise InvalidUploadError("The image could not be decoded.") from exc

    original_size = rgb.size

    # EXIF metadata may still be attached; stripping it removes the ICC profile
    # and any GPS data before the image is persisted (privacy + size).
    rgb.info.pop("exif", None)

    width, height = rgb.size
    long_edge = max(width, height)
    scale = 1.0
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        rgb = rgb.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            # LANCZOS is the best downscaling filter for text: it preserves
            # stroke separation where BILINEAR would blur adjacent matras.
            Image.Resampling.LANCZOS,
        )

    gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    return PreparedImage(gray=gray, scale=scale, original_size=original_size)


def binarize(gray: np.ndarray, *, invert: bool = False) -> np.ndarray:
    """Otsu-threshold an image into a boolean text mask.

    Args:
        gray: 8-bit grayscale array.
        invert: Force the polarity (for white-on-dark slides).

    Returns:
        Boolean array where True means "ink".

    Why Otsu by hand instead of OpenCV: it is ~25 lines, removes a 60MB
    dependency from the image, and the computation is transparent to a reader.
    Otsu maximizes inter-class variance, which behaves well on scanned pages
    with bimodal histograms.
    """
    if gray.dtype != np.uint8:
        raise ValueError("binarize expects a uint8 array")

    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return np.zeros(gray.shape, dtype=bool)

    probabilities = histogram / total
    omega = np.cumsum(probabilities)             # class-1 cumulative weight
    mu = np.cumsum(probabilities * np.arange(256))  # class-1 cumulative mean
    mu_total = mu[-1]

    # Guard against division by zero for degenerate (all-one-value) images.
    denominator = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(denominator > 0, (mu_total * omega - mu) ** 2 / denominator, 0.0)

    threshold = int(np.argmax(variance))
    ink = gray > threshold
    return ~ink if not invert else ink


def ensure_dark_text(gray: np.ndarray) -> np.ndarray:
    """Invert a light-on-dark page so text is always dark ink.

    Args:
        gray: 8-bit grayscale array.

    Returns:
        The original array, or its inverse when the border is dark.

    Why the border heuristic: OCR pages are usually light paper with a dark
    border only when inverted. Sampling the 8px frame is cheap and beats
    running the whole pipeline twice.
    """
    frame = np.concatenate(
        [gray[:8, :].ravel(), gray[-8:, :].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()]
    )
    return 255 - gray if float(np.median(frame)) < 128 else gray
