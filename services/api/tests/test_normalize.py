"""Tests for Devanagari-safe normalization.

These are the most important tests in the repository. A silent change to this
module would corrupt every OCR result while every other test still passed, so
each combining-mark case is pinned explicitly.
"""

from __future__ import annotations

import pytest
from devanagari_text.normalize import (
    contains_devanagari,
    devanagari_ratio,
    normalize_text,
    to_devanagari_digits,
    to_latin_digits,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  नमस्ते   दुनिया  ", "नमस्ते दुनिया"),
        ("भारत\n\n\nएक देश", "भारत\n\nएक देश"),
        ("पंक्ति\tएक", "पंक्ति एक"),
    ],
)
def test_whitespace_normalization(raw: str, expected: str) -> None:
    """Collapse whitespace without touching the characters themselves."""
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "कि",  # i-matra (combining vowel sign)
        "क्‍ष",  # virama + ZWJ conjunct
        "ज्ञ",  # jna conjunct
        "र्क",  # reph (ra + virama above ka)
        "कँ",  # chandrabindu
        "कः",  # visarga
        "क़",  # nukta
        "हिन्दी।",  # danda terminator
    ],
)
def test_combining_marks_are_preserved(text: str) -> None:
    """Every combining mark must survive normalization byte for byte.

    This is the regression guard against anyone "helpfully" adding a
    mark-stripping step to post-processing.
    """
    assert normalize_text(text) == text


def test_normalization_is_idempotent() -> None:
    """Normalizing twice must equal normalizing once.

    Non-idempotent normalization makes cached results drift every time a saved
    job is re-served, which is invisible until a user compares two exports.
    """
    raw = "  राष्ट्रीय   शिक्षा नीति २०२० \n\n\n लागू  "
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_digit_translation_round_trip() -> None:
    """Latin and Devanagari digits convert losslessly in both directions."""
    assert to_latin_digits("२०२०") == "2020"
    assert to_devanagari_digits("2020") == "२०२०"
    assert to_latin_digits(to_devanagari_digits("1947")) == "1947"


def test_digit_translation_leaves_text_untouched() -> None:
    """Only digit codepoints change; matras and conjuncts are unaffected."""
    assert to_latin_digits("कक्षा ५ में") == "कक्षा 5 में"


def test_contains_devanagari() -> None:
    """Script detection works for text, and rejects Latin and empty input."""
    assert contains_devanagari("हिन्दी")
    assert not contains_devanagari("hindi")
    assert not contains_devanagari("")


def test_devanagari_ratio_flags_latin_garbage() -> None:
    """A Latin response for a Hindi page must score near zero.

    This is the signal the API exposes to the UI so a wrong-orientation scan
    shows a warning instead of confidently displaying nonsense.
    """
    assert devanagari_ratio("पूरी तरह हिन्दी") == 1.0
    assert devanagari_ratio("completely latin") == 0.0
    assert devanagari_ratio("") == 0.0
    mixed = devanagari_ratio("हिन्दी text")
    assert 0.0 < mixed < 1.0


def test_control_characters_are_removed() -> None:
    """Decoder artifacts (NUL, BEL) never reach a client."""
    assert normalize_text("क\x00ख\x07ग") == "कखग"
