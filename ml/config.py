"""Training-side configuration, kept separate from the serving settings.

Why two config modules: training is run on a free Colab/Kaggle GPU with a
completely different resource envelope (16GB VRAM vs 512MB RAM). Sharing one
config would mean a training knob could accidentally reach production.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters for the CRNN recognizer.

    Attributes:
        image_height: Input height in pixels. 32 is the CRNN standard and keeps
            Devanagari headline + matras legible without quadratic cost.
        image_width: Maximum input width. Longer lines are split at word gaps.
        max_label_length: CTC target length ceiling; a common Hindi sentence
            line is 40-60 characters, so 80 leaves headroom.
        batch_size: Chosen to fit 8GB VRAM at 32x320.
        epochs: Upper bound; early stopping on validation CER usually triggers first.
        learning_rate: AdamW default for a small transformer/RNN head.
        charset_path: Where the index -> character map is written.
        output_dir: Where checkpoints and the exported ONNX land.
    """

    image_height: int = 32
    image_width: int = 320
    max_label_length: int = 80
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    charset_path: Path = Path("data/charset.txt")
    output_dir: Path = Path("artifacts")

    @property
    def num_classes(self) -> int:
        """Number of CTC output classes including the blank at index 0."""
        if not self.charset_path.exists():
            # 1 blank + the Devanagari block minus unused slots is the rough size;
            # the real number is always read from the charset file at train time.
            return 1
        return len(self.charset_path.read_text(encoding="utf-8").splitlines())


# The five real-world slices that matter. Tracking per-slice CER instead of one
# aggregate number is what stops a model that is great on clean print and
# useless on phone photos from looking "done".
EVAL_SLICES: tuple[str, ...] = (
    "printed_clean",
    "printed_scan",
    "handwritten",
    "phone_photo",
    "mixed_layout",
)
