"""Shared Devanagari text utilities used by BOTH training and serving.

Why this package exists
-----------------------
The single most common failure mode in an Indic OCR project is *train/serve
skew*: the training pipeline normalizes text one way (say, it strips the
danda), the serving pipeline normalizes a different way, and the character
error rate silently degrades. Keeping one importable module for both sides
removes that entire class of bug.

This is deliberately a plain package on ``PYTHONPATH`` rather than a pip
installable distribution: it has zero third-party dependencies, so adding
packaging metadata would cost more than it saves. ``PYTHONPATH`` includes
the repository root and ``packages/`` (see Makefile and the Dockerfile).
"""

from devanagari_text.normalize import (
    DEVANAGARI_DIGITS,
    DevanagariText,
    contains_devanagari,
    normalize_text,
    to_devanagari_digits,
    to_latin_digits,
)

__all__ = [
    "DEVANAGARI_DIGITS",
    "DevanagariText",
    "contains_devanagari",
    "normalize_text",
    "to_devanagari_digits",
    "to_latin_digits",
]
