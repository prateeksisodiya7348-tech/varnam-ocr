"""Training loop for the Devanagari CRNN recognizer.

Run on a free GPU (Colab T4 / Kaggle P100):

    python -m ml.data.prepare --images data/raw --manifest data/train_manifest.csv
    python -m ml.train --manifest data/train_manifest.csv --charset data/charset.txt

Why this file is short: every line here exists to fix a specific failure mode.
Longer training scripts mostly add logging, which is the one thing you should not
build yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.config import TrainConfig
from ml.data.manifest import ManifestRow, build_charset, read_manifest, split_manifest
from ml.metrics import character_error_rate
from ml.model import CrnnRecognizer, count_parameters


class LineDataset(Dataset):
    """Line crop dataset driven by a manifest.

    Attributes:
        rows: Labelled samples.
        charset: Index -> character map (index 0 is blank).
        label_to_index: Inverse map.
        config: Geometry settings.
    """

    def __init__(self, rows: list[ManifestRow], charset: list[str], config: TrainConfig) -> None:
        """Store rows and build the label index.

        Args:
            rows: Samples to serve.
            charset: Output of ``build_charset`` (blank at index 0).
            config: Geometry settings.
        """
        self.rows = rows
        self.charset = charset
        self.label_to_index = {character: index for index, character in enumerate(charset)}
        self.config = config

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load one normalized line image and its encoded label.

        Args:
            index: Sample index.

        Returns:
            ``(image_tensor, label_tensor, label_length)``.

        Why resize with padding instead of stretching: stretching changes the
        aspect ratio of Devanagari conjuncts, which teaches the model a
        distortion that never occurs at inference time.
        """
        row = self.rows[index]
        with Image.open(row.image_path) as source:
            image = source.convert("L")

        target_h = self.config.image_height
        target_w = self.config.image_width
        width, height = image.size
        scale = target_h / max(1, height)
        new_w = max(1, min(target_w, int(round(width * scale))))
        image = image.resize((new_w, target_h), Image.Resampling.BILINEAR)

        canvas = Image.new("L", (target_w, target_h), color=255)
        canvas.paste(image, (0, 0))

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        normalized = (array - 0.5) / 0.5

        # Unknown characters are dropped rather than mapped to blank; mapping to
        # blank would teach the model to silently delete them.
        indices = [self.label_to_index[ch] for ch in row.text if ch in self.label_to_index][
            : self.config.max_label_length
        ]
        label = torch.tensor(indices, dtype=torch.long)

        return torch.from_numpy(normalized)[None, :, :], label, len(indices)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    """Pad variable-length labels into a batch.

    Args:
        batch: Output of ``LineDataset.__getitem__``.

    Returns:
        ``(images, targets, input_lengths, target_lengths)`` as CTCLoss expects.
    """
    images, labels, lengths = zip(*batch, strict=True)
    max_length = max(1, max(lengths))
    padded = torch.zeros((len(labels), max_length), dtype=torch.long)
    for row, label in enumerate(labels):
        padded[row, : len(label)] = label

    images_stacked = torch.stack(images)
    # Every line image has the same width here, so the CTC input length is the
    # CNN's output width (80 for a 320px input with 4 pooling stages).
    input_lengths = torch.full((len(images),), images_stacked.shape[3] // 4, dtype=torch.long)
    return images_stacked, padded, input_lengths, torch.tensor(lengths, dtype=torch.long)


def decode_greedy(logits: np.ndarray, charset: list[str]) -> str:
    """Greedy CTC decode for validation.

    Args:
        logits: ``(T, num_classes)`` logits.
        charset: Index -> character map with blank at 0.

    Returns:
        Decoded string.
    """
    best = logits.argmax(axis=-1)
    characters: list[str] = []
    previous = -1
    for index in best:
        index = int(index)
        if index != previous and index != 0 and index < len(charset):
            characters.append(charset[index])
        previous = index
    return "".join(characters)


@torch.no_grad()
def validate(model: CrnnRecognizer, loader: DataLoader, charset: list[str]) -> float:
    """Compute mean CER over the validation loader.

    Args:
        model: Model in eval mode.
        loader: Validation loader.
        charset: Index -> character map.

    Returns:
        Mean character error rate.
    """
    model.eval()
    scores: list[float] = []
    for images, labels, _input_lengths, _target_lengths in loader:
        logits = model(images).permute(1, 0, 2)  # (N, T, C) for decoding
        for row, label in enumerate(labels):
            reference = "".join(charset[index] for index in label.tolist() if index > 0)
            hypothesis = decode_greedy(logits[row].numpy(), charset)
            scores.append(character_error_rate(reference, hypothesis))
    return float(np.mean(scores)) if scores else 1.0


def main(argv: list[str] | None = None) -> int:
    """Train, validate, and checkpoint.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Train the Varnam CRNN recognizer.")
    parser.add_argument("--manifest", type=Path, default=Path("data/train_manifest.csv"))
    parser.add_argument("--charset", type=Path, default=Path("data/charset.txt"))
    parser.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig().learning_rate)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    if not rows:
        print("error: manifest is empty")
        return 2

    charset = (
        args.charset.read_text(encoding="utf-8").splitlines()
        if args.charset.exists()
        else build_charset([row.text for row in rows], args.charset)
    )
    # Blank must be index 0; a charset without it produces an untrainable model.
    if not charset or charset[0] != "<blank>":
        print("error: charset must start with <blank> as index 0")
        return 2

    config = TrainConfig(charset_path=args.charset)
    train_rows, val_rows = split_manifest(rows)
    print(f"train={len(train_rows)} val={len(val_rows)} classes={len(charset)}")

    train_loader = DataLoader(
        LineDataset(train_rows, charset, config),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        LineDataset(val_rows, charset, config),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrnnRecognizer(num_classes=len(charset)).to(device)
    print(f"device={device} parameters={count_parameters(model):,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_cer = 1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels, input_lengths, target_lengths in train_loader:
            images = images.to(device)
            logits = model(images)  # (T, N, C)

            loss = criterion(logits, labels, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping is not optional for LSTMs: a single exploding
            # batch can permanently wreck the weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.item())

        scheduler.step()
        val_cer = validate(model, val_loader, charset)
        mean_loss = running / max(1, len(train_loader))
        print(f"epoch {epoch:02d} loss={mean_loss:.4f} val_cer={val_cer:.2%}")

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(
                {"model_state": model.state_dict(), "num_classes": len(charset), "val_cer": val_cer},
                args.out / "best.pt",
            )
            print(f"  checkpoint saved (best CER {best_cer:.2%})")

    print(f"done. best validation CER: {best_cer:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
