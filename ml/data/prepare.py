"""Build a manifest CSV from a folder of image/text pairs.

    python -m ml.data.prepare --images data/raw --manifest data/train_manifest.csv

Expected layout (one transcription per image, same basename):

    data/raw/scan/page_001.png
    data/raw/scan/page_001.txt     <- UTF-8 Devanagari transcription

Why pairs on disk instead of a fancy dataset class: every public Devanagari
dataset (IIIT-ILST, CVIT-OCR, DHCD) ships as images plus text somewhere. A folder
of pairs is the lowest common denominator that all of them can be converted to,
and converting is a 30-line shell script rather than a library dependency.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Find image/transcription pairs under ``root``.

    Args:
        root: Directory to scan recursively.

    Returns:
        Pairs where a ``.txt`` sibling exists and is non-empty.

    Why require non-empty: an empty transcription is usually a failed extraction,
    and training on it teaches the model to output nothing.
    """
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(root.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        transcript = image.with_suffix(".txt")
        if transcript.exists() and transcript.read_text(encoding="utf-8").strip():
            pairs.append((image, transcript))
    return pairs


def write_manifest(
    pairs: list[tuple[Path, Path]], destination: Path, *, slice_name: str
) -> int:
    """Write pairs to a manifest CSV.

    Args:
        pairs: Image/transcription pairs.
        destination: Output CSV path.
        slice_name: Label applied to every row, used for per-slice metrics.

    Returns:
        Number of rows written.

    Why paths are relative to the manifest's own folder: ``read_manifest``
    resolves each ``image_path`` against the manifest directory. Writing paths
    relative to the current working directory instead silently produces
    ``data/data/synth/...`` on the next read - a bug that only shows up as
    "every image is missing" after the manifest has already been handed to
    someone else.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = destination.parent.resolve()

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "text", "slice"])
        for image, transcript in pairs:
            # A manifest can legitimately point outside its own folder, so fall
            # back to an absolute path rather than raising.
            try:
                relative = Path(os.path.relpath(image.resolve(), base)).as_posix()
            except ValueError:
                relative = image.resolve().as_posix()
            writer.writerow([relative, transcript.read_text(encoding="utf-8").strip(), slice_name])
    return len(pairs)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Build an OCR manifest from image/text pairs.")
    parser.add_argument("--images", type=Path, required=True, help="Root folder of pairs.")
    parser.add_argument("--manifest", type=Path, default=Path("data/train_manifest.csv"))
    parser.add_argument("--slice", default="unlabeled", help="Slice label for these rows.")
    args = parser.parse_args(argv)

    if not args.images.exists():
        print(f"error: {args.images} does not exist")
        return 2

    pairs = collect_pairs(args.images)
    if not pairs:
        print(f"error: no image/text pairs found under {args.images}")
        print("       each image needs a same-named .txt with its transcription")
        return 2

    count = write_manifest(pairs, args.manifest, slice_name=args.slice)
    print(f"wrote {count} rows to {args.manifest} (slice={args.slice})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
