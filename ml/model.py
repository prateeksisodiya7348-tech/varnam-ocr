"""CRNN + CTC recognizer.

Architecture choice, and why it is defensible in a review:

* A CNN trunk downsamples the line image height 32 -> 1, turning the image into
  a sequence over the horizontal axis. For Devanagari this preserves the exact
  left-to-right order of the shirorekha and its matras, which is the property
  that matters.
* A bidirectional LSTM sees two characters of context on each side. Devanagari
  needs this badly, because a matra's meaning depends on the consonant after it.
* A linear head emits one distribution per horizontal step; CTC handles the
  alignment without per-character bounding boxes in the training data - which
  matters because line-level transcription is all most real datasets provide.

TrOCR (a full encoder-decoder transformer) is the accuracy escape hatch and is
documented in MODEL_CARD.md, but it is ~10x slower on CPU and 6x larger. On a
512MB free instance that trade does not pay off.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU with a configurable downsample.

    Attributes:
        block: The sequential stack.

    Pooling is split per axis on purpose. Pooling height collapses the image into
    a sequence; pooling width throws away horizontal resolution, which is exactly
    where Devanagari's matras live. Pooling width four times over a 320px input
    leaves 20 time steps - fewer than the characters in a single line, which makes
    the CTC alignment impossible. So width is pooled twice at most.
    """

    def __init__(self, in_channels: int, out_channels: int, *, pool: str = "both") -> None:
        """Build one convolutional block.

        Args:
            in_channels: Input feature maps.
            out_channels: Output feature maps.
            pool: ``"both"`` (halve H and W), ``"height"`` (halve H only), or
                ``"none"``.
        """
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool == "both":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif pool == "height":
            # Kernel (2,1): halves the vertical axis, leaves the sequence axis intact.
            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: ``(N, C, H, W)`` tensor.

        Returns:
            Feature map with the same batch dimension.
        """
        return self.block(x)


class CrnnRecognizer(nn.Module):
    """CTC line recognizer.

    Attributes:
        num_classes: Output classes including the blank at index 0.
        cnn: Feature extractor.
        rnn: Sequence model over horizontal steps.
        head: Per-step classifier.

    Shape trace for a 32x320 input:
        (N,1,32,320) -> CNN -> (N,256,2,80) -> squeeze H -> (N,80,512)
        -> BiLSTM -> (N,80,256) -> head -> (N,80,num_classes)

    The 80 output steps bound the label length: CTC needs T >= label length for
    a valid alignment, which is why ``max_label_length`` is 80. That only holds
    because the last two blocks pool height and NOT width - see ``ConvBlock``.
    """

    def __init__(self, num_classes: int, *, hidden_size: int = 256) -> None:
        """Assemble the network.

        Args:
            num_classes: Charset size including blank.
            hidden_size: LSTM hidden width per direction.
        """
        super().__init__()
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            ConvBlock(1, 64, pool="both"),      # 32x320 -> 16x160
            ConvBlock(64, 128, pool="both"),    # 16x160 -> 8x80
            ConvBlock(128, 256, pool="height"), # 8x80   -> 4x80
            ConvBlock(256, 256, pool="height"), # 4x80   -> 2x80
        )

        self.rnn = nn.LSTM(
            input_size=256 * 2,  # feature maps after collapsing the height axis
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: ``(N, 1, H, W)`` normalized line images.

        Returns:
            Log-probabilities of shape ``(T, N, num_classes)``, the layout
            ``torch.nn.CTCLoss`` expects (time-major).
        """
        features = self.cnn(x)
        batch, channels, height, width = features.shape

        # Collapse height into channels: (N, C, H, W) -> (N, W, C*H).
        # This is the step that converts an image into a sequence.
        features = features.permute(0, 3, 1, 2).reshape(batch, width, channels * height)

        sequence, _ = self.rnn(features)
        logits = self.head(sequence)

        # Log-softmax is required by CTCLoss (it works in log space).
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters.

    Args:
        model: Any PyTorch module.

    Returns:
        Number of parameters with ``requires_grad`` set.

    Why report this: a ~15M parameter model is ~60MB as float32 and ~15MB as
    INT8, which is the number that decides whether it fits the deployment budget.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
