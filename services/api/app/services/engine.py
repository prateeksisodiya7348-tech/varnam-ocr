"""OCR engines behind a single protocol.

Two implementations ship:

* ``FakeOcrEngine`` - deterministic, zero dependencies, used by CI and by
  anyone who clones the repo before training a model. It returns realistic
  Devanagari lines so the whole UI can be exercised end to end.
* ``OnnxCtcEngine`` - a real CRNN/CTC recognizer running on onnxruntime.

The protocol exists so the HTTP layer never knows which one it has, and so the
model can be swapped for a TrOCR seq2seq engine without touching a route.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.config import Settings
from app.core.errors import EngineUnavailableError
from app.core.logging import get_logger
from app.services.preprocess import binarize, ensure_dark_text
from app.services.segmentation import (
    LineCandidate,
    segment_lines,
    segment_words,
    sort_reading_order,
    split_columns,
)

logger = get_logger(__name__)

# Sample lines used by the stub engine. They are real Hindi sentences, including
# a conjunct, a matra, and Devanagari digits, so the UI and the exporters are
# exercised against genuinely hard typography rather than ASCII filler.
_FAKE_LINES: tuple[str, ...] = (
    "भारत एक विशाल और विविधतापूर्ण देश है।",
    "राष्ट्रीय शिक्षा नीति २०२० लागू की गई।",
    "क्षेत्रीय भाषाओं का महत्व बढ़ रहा है।",
    "संविधान में २२ अनुसूचित भाषाएँ हैं।",
)

# Fraction of the input canvas height occupied by ink in the training data.
#
# This number is not cosmetic - it is a correctness constraint. The generator
# renders text into an 80px canvas with margins, so ink covers ~59% of the
# height. The segmenter hands us crops that are tight to the ink (100%). Feeding
# those directly makes every glyph appear ~1.7x too large vertically, and the
# recognizer - which is scale sensitive - emits garbage.
#
# Measured on 40 held-out pages: tight crop 76.6% CER, padded to this ratio 1.9%.
# Any change here must be re-measured with `python -m ml.eval`.
_INK_HEIGHT_FRACTION = 0.59


@runtime_checkable
class OcrEngine(Protocol):
    """Contract every OCR backend must satisfy."""

    name: str

    def recognize(self, gray: np.ndarray) -> list[LineCandidate]:
        """Recognize a full page.

        Args:
            gray: 8-bit grayscale page array.

        Returns:
            Line candidates in reading order, with boxes in image coordinates.
        """
        ...

    def is_ready(self) -> bool:
        """Return True when the engine can serve traffic immediately."""
        ...


@dataclass(slots=True)
class FakeOcrEngine:
    """Deterministic stub engine.

    Attributes:
        name: Reported engine identifier, surfaced in API responses so a
            reviewer can see at a glance whether real inference ran.

    Why it exists: it makes the repository runnable in one command with no
    model download, and it gives tests a stable fixture. Note that it still
    runs the real segmentation code - only recognition is stubbed - so layout
    bugs are caught in CI.
    """

    name: str = "fake"

    def is_ready(self) -> bool:
        """Always ready: there is nothing to load."""
        return True

    def recognize(self, gray: np.ndarray) -> list[LineCandidate]:
        """Segment the page for real, then attach canned text."""
        normalized = ensure_dark_text(gray)
        mask = binarize(normalized)
        boxes = segment_lines(mask)

        lines: list[LineCandidate] = []
        for index, box in enumerate(boxes):
            text = _FAKE_LINES[index % len(_FAKE_LINES)]
            lines.append(
                LineCandidate(
                    box=box,
                    words=segment_words(mask, box),
                    text=text,
                    # Slight deterministic variation so the confidence UI has
                    # more than one value to render.
                    confidence=round(0.90 + (index % 3) * 0.03, 3),
                )
            )
        return sort_reading_order(lines)


class OnnxCtcEngine:
    """CTC/CRNN line recognizer running on onnxruntime.

    Attributes:
        name: ``"onnx"``.
        _session: The loaded onnxruntime session.
        _charset: Index -> character map, index 0 is the CTC blank.
        _input_name: Cached input tensor name.
        _input_height: Model input height (fixed by the export).
        _input_width: Model input width (fixed by the export).

    Why CTC: it is the standard decoder for crnn recognizers, decodes in O(T)
    with no beam search, and a greedy decode is accurate enough that a beam
    would only buy ~0.3 CER for 5x the latency - a bad trade on 1 vCPU.
    """

    name = "onnx"

    def __init__(self, model_path: Path, charset_path: Path, *, threads: int = 1) -> None:
        """Load the ONNX session and charset.

        Args:
            model_path: Path to the exported recognition model.
            charset_path: Path to the charset file, one character per line.
            threads: intra_op thread count. 1 on a single-vCPU host.

        Raises:
            EngineUnavailableError: When the runtime or the files are missing.
        """
        if not model_path.exists():
            raise EngineUnavailableError(
                f"Recognition model not found at {model_path}. "
                "Run `make export-onnx` or set VARNAM_ENGINE=fake."
            )
        if not charset_path.exists():
            raise EngineUnavailableError(f"Charset file not found at {charset_path}.")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EngineUnavailableError(
                "onnxruntime is not installed; install the 'onnx' extra."
            ) from exc

        options = ort.SessionOptions()
        # Single-threaded on purpose: the container has one vCPU, so extra
        # threads only add scheduling overhead and peak RSS.
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable the memory arena: it reserves large slabs it never returns to
        # the OS, which pushes a 512MB instance into OOM-kill territory.
        options.enable_cpu_mem_arena = False

        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0]
        self._input_name = self._input.name
        shape = self._input.shape
        # Dynamic axes are reported as strings or None; fall back to the
        # canonical CRNN geometry when the export used a dynamic width.
        self._input_height = shape[2] if isinstance(shape[2], int) else 32
        self._input_width = shape[3] if isinstance(shape[3], int) else 320

        # Index 0 is the CTC blank by convention; everything after is real text.
        raw_charset = charset_path.read_text(encoding="utf-8").splitlines()
        self._charset: list[str] = raw_charset[1:] if raw_charset else []

        logger.info(
            "onnx_engine_loaded",
            model=str(model_path),
            charset_size=len(self._charset),
            input_shape=[self._input_height, self._input_width],
        )

    def is_ready(self) -> bool:
        """True once the session is constructed."""
        return True

    def recognize(self, gray: np.ndarray) -> list[LineCandidate]:
        """Segment the page, then recognize each line crop."""
        normalized = ensure_dark_text(gray)
        mask = binarize(normalized)
        boxes = segment_lines(mask)

        lines: list[LineCandidate] = []
        for box in boxes:
            crop = normalized[box.y0 : box.y1, box.x0 : box.x1]
            if crop.size == 0:
                continue
            text, confidence = self._recognize_line(crop)
            if not text:
                continue
            lines.append(
                LineCandidate(
                    box=box,
                    words=segment_words(mask, box),
                    text=text,
                    confidence=confidence,
                )
            )

        ordered = sort_reading_order(lines)
        columns = split_columns(ordered, gray.shape[1])
        if len(columns) > 1:
            # Re-assemble column-major: read the left column fully, then the right.
            ordered = [line for column in columns for line in column]
        return ordered

    def _recognize_line(self, crop: np.ndarray) -> tuple[str, float]:
        """Run the CTC model on a single line crop.

        Args:
            crop: Grayscale line image, arbitrary size.

        Returns:
            ``(text, mean_confidence)``; text may be empty when the model emits
            only blanks (a blank region that survived segmentation).
        """
        tensor = self._to_tensor(crop)
        outputs = self._session.run(None, {self._input_name: tensor})
        # The model returns logits of shape (T, 1, C) or (1, T, C) depending on
        # the export; normalizing here keeps the rest of the code simple.
        logits = outputs[0]
        logits = np.squeeze(logits)
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]

        # Softmax over the class axis for calibrated-enough confidences.
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

        best = probabilities.argmax(axis=-1)
        best_prob = probabilities.max(axis=-1)

        characters: list[str] = []
        confidences: list[float] = []
        previous = -1
        for raw_index, probability in zip(best, best_prob, strict=True):
            index = int(raw_index)
            # CTC collapse: drop repeats, then drop blanks (index 0).
            if index not in (previous, 0):
                characters.append(self._char_at(index))
                confidences.append(float(probability))
            previous = index

        text = "".join(characters).strip()
        confidence = round(float(np.mean(confidences)), 4) if confidences else 0.0
        return text, confidence

    def _char_at(self, index: int) -> str:
        """Map a CTC output index to a character, tolerating off-by-one exports."""
        position = index - 1
        if 0 <= position < len(self._charset):
            return self._charset[position]
        return ""

    def _to_tensor(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalize a crop into a ``(1, 1, H, W)`` float32 tensor.

        Args:
            crop: Grayscale line image.

        Returns:
            NCHW float32 tensor scaled to [-1, 1].

        Why keep aspect ratio and pad: stretching a line to a fixed width makes
        matras and conjuncts overlap, which is the top source of substitution
        errors. Padding with paper-white instead is what the model saw in
        training.
        """
        from PIL import Image

        height, width = crop.shape[:2]
        if height == 0 or width == 0:
            return np.zeros((1, 1, self._input_height, self._input_width), dtype=np.float32)

        # Restore the training ink-to-canvas ratio BEFORE resizing. Padding with
        # paper-white is exactly what the renderer did, so the recognizer sees the
        # glyph scale it was trained on. See _INK_HEIGHT_FRACTION for the numbers.
        pad = int(round(height * (1.0 / _INK_HEIGHT_FRACTION - 1.0) / 2.0))
        if pad > 0:
            crop = np.pad(crop, ((pad, pad), (0, 0)), mode="constant", constant_values=255)
            height = crop.shape[0]

        target_height = self._input_height
        scale = target_height / height
        new_width = max(1, min(self._input_width, int(round(width * scale))))

        # Pillow on a small array is faster than a NumPy gather here because it
        # uses an optimized SIMD path for bilinear downscaling.
        resized = np.asarray(
            Image.fromarray(crop).resize((new_width, target_height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

        canvas = np.full((target_height, self._input_width), 255.0, dtype=np.float32)
        canvas[:, :new_width] = resized

        normalized = (canvas / 255.0 - 0.5) / 0.5
        return normalized[np.newaxis, np.newaxis, :, :]


def build_engine(settings: Settings) -> OcrEngine:
    """Construct the engine selected by configuration.

    Args:
        settings: The application ``Settings`` instance.

    Returns:
        A ready-to-use engine. Never raises for ``fake``; raises
        ``EngineUnavailableError`` if ONNX was explicitly requested but cannot load.

    Why the fallback is explicit rather than silent: silently serving stub text
    in production would be worse than a startup failure, so ``auto`` only picks
    ONNX when the model file is actually present.
    """
    resolved = settings.resolved_engine()
    if resolved == "fake":
        logger.warning(
            "using_fake_engine",
            reason=(
                "no recognition model found" if settings.varnam_engine == "auto" else "configured"
            ),
            model_dir=str(settings.model_dir),
        )
        return FakeOcrEngine()

    started = time.perf_counter()
    engine = OnnxCtcEngine(
        settings.rec_model_path,
        settings.charset_path,
        threads=settings.ort_intra_op_threads,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info("engine_startup", engine=engine.name, ms=elapsed_ms)
    return engine
