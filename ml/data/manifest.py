"""Dataset manifests and charset construction.

The manifest is the single contract between "whatever data you have" and the
evaluation code. It is intentionally a flat CSV so it can be produced by a
script, a spreadsheet export, or by hand when you only have 50 labelled samples.

Schema (header required):

    image_path,text,slice
    data/scan/page_001.png,"भारत एक विशाल देश है।",printed_scan
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from devanagari_text.normalize import normalize_text

REQUIRED_COLUMNS = ("image_path", "text", "slice")

# Printable-ASCII fallback characters. A recognizer trained only on Devanagari
# still meets English words, numbers and punctuation inside real documents; if
# those classes are missing the model is forced to emit the nearest Devanagari
# glyph, which corrupts mixed lines.
ASCII_FALLBACK = "0123456789.,:;!?()[]{}-/%&'\"+*=@#"


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One labelled sample.

    Attributes:
        image_path: Path to the image, resolved against the manifest directory.
        text: Ground-truth transcription, already normalized.
        slice_name: Evaluation slice, e.g. ``printed_scan``.
    """

    image_path: Path
    text: str
    slice_name: str


def read_manifest(path: Path) -> list[ManifestRow]:
    """Load and validate a manifest CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        Parsed rows with normalized text.

    Raises:
        ValueError: When required columns are missing or a row is malformed.

    Why normalize on read: the ground truth must be in exactly the same form as
    model output, otherwise the CER is measuring our text-formatting choices
    instead of recognition accuracy.
    """
    if not path.exists():
        raise ValueError(f"Manifest not found: {path}")

    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")

        for line_number, raw in enumerate(reader, start=2):
            text = normalize_text(raw.get("text") or "", keep_line_breaks=True)
            image = (raw.get("image_path") or "").strip()
            if not image or not text:
                # Skip rather than fail: a real scrape always has a few empty rows,
                # and dropping them silently would inflate the score.
                continue
            rows.append(
                ManifestRow(
                    image_path=(path.parent / image).resolve(),
                    text=text,
                    slice_name=(raw.get("slice") or "unlabeled").strip(),
                )
            )
    return rows


def build_charset(texts: list[str], destination: Path, *, include_ascii: bool = True) -> list[str]:
    """Derive the CTC class list from a corpus and write it to disk.

    Args:
        texts: All training transcriptions.
        destination: Output path, one character per line.
        include_ascii: Append the ASCII fallback set.

    Returns:
        The charset list, with ``""`` at index 0 as the CTC blank.

    Why derive instead of hardcoding the block: the Devanagari block contains
    dozens of slots that never appear in real text. Training a class that has
    zero examples wastes capacity and makes the softmax slightly less confident
    for no benefit. Derived means every class is learnable.

    Why the blank is line 1: ``OnnxCtcEngine`` reads the file positionally and
    treats index 0 as blank, which is the universal CTC convention. Writing the
    blank as the literal first line (rather than an offset) keeps the file
    human-readable and diffable.
    """
    seen: set[str] = set()
    for text in texts:
        seen.update(text)

    # Sort by codepoint so the file is stable across runs and diffs cleanly in
    # git. A random order would make every regeneration look like a huge change.
    #
    # Whitespace is dropped EXCEPT the space character: word spacing is a real,
    # learnable class. Excluding it makes the decoder structurally incapable of
    # emitting a space, so every line collapses into one unreadable token
    # ("तथा प्रत्येक" -> "तथाप्रत्येक"). That is a charset bug, not a model bug,
    # and it looked like a 14% CER before anyone noticed.
    characters = sorted(
        ch for ch in seen if ch != "\x00" and (ch == " " or not ch.isspace())
    )
    if include_ascii:
        characters = sorted(set(characters) | set(ASCII_FALLBACK))
    characters = sorted(set(characters) | {" "})

    lines = ["<blank>", *characters]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def split_manifest(
    rows: list[ManifestRow], *, val_fraction: float = 0.15, seed: int = 1337
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Split rows into train/validation deterministically.

    Args:
        rows: All rows.
        val_fraction: Fraction held out for validation.
        seed: RNG seed, so every run sees the same split.

    Returns:
        ``(train_rows, val_rows)``.

    Why not sklearn: one ``random.Random`` call does the same thing without a
    dependency, and the determinism is explicit rather than implied.
    """
    import random  # noqa: PLC0415 - only needed here

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * val_fraction))
    return shuffled[cut:], shuffled[:cut]
