"""Devanagari-safe text normalization.

The hard rule
-------------
Devanagari is an abugida: a consonant carries an inherent vowel, and matras
(vowel signs), the virama/halant, nukta, anusvara and visarga are *combining*
code points that change the reading of the base character. Any "cleanup" that
drops them corrupts the text:

    कि  (ki)        -> dropping the i-matra yields  क  (ka)
    क्ष  (ksha)      -> dropping virama yields  कष  ("kash")
    कँ  (kan, nasal) -> dropping chandrabindu yields  क

Therefore this module only ever does three things:

1. Unicode **NFC** composition (a canonical, lossless transform).
2. Whitespace collapsing and control-character removal (never marks).
3. Optional, explicit digit translation between scripts.

Everything else is left to the caller. There is no ``strip_marks`` helper on
purpose: it is the single most damaging function an OCR post-processor can
have, and if it does not exist nobody can accidentally call it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Devanagari digits ० १ २ ३ ४ ५ ६ ७ ८ ९ map 1:1 onto 0-9 by codepoint offset.
DEVANAGARI_ZERO = "\u0966"  # ०
DEVANAGARI_NINE = "\u096f"  # ९
DEVANAGARI_DIGITS = "".join(chr(0x0966 + i) for i in range(10))

# Characters that are legitimate in Devanagari text and must never be removed.
# Used by tests as an explicit contract, and by `contains_devanagari`.
_DEVANAGARI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari block (matras, virama, nukta, danda...)
    (0xA8E0, 0xA8FF),  # Devanagari Extended (vedic accents, combining marks)
    (0x1CD0, 0x1CFF),  # Vedic Extensions
    (0x200C, 0x200D),  # ZWNJ / ZWJ - meaningful inside conjuncts
)

# Danda and double danda are sentence terminators unique to Indic scripts.
DANDA = "\u0964"
DOUBLE_DANDA = "\u0965"

# Control characters (C0/C1) except tab/newline, which we normalize separately.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Any run of horizontal whitespace -> a single space. We intentionally keep the
# newline out of this class so that line structure survives for hOCR export.
_HSPACE_RE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class DevanagariText:
    """A normalized Devanagari string plus cheap, cached analysis flags.

    Attributes:
        text: The NFC-composed, whitespace-collapsed string.
        has_devanagari: True when at least one Devanagari codepoint is present.
        char_count: Number of Unicode code points (not grapheme clusters).
    """

    text: str
    has_devanagari: bool
    char_count: int

    @classmethod
    def from_raw(cls, raw: str) -> DevanagariText:
        """Normalize ``raw`` and precompute the analysis flags."""
        cleaned = normalize_text(raw)
        return cls(
            text=cleaned,
            has_devanagari=contains_devanagari(cleaned),
            char_count=len(cleaned),
        )


def normalize_text(raw: str, *, keep_line_breaks: bool = True) -> str:
    """Normalize model output without ever destroying combining marks.

    Args:
        raw: Raw string produced by the recognizer or typed by a user.
        keep_line_breaks: Preserve ``\\n`` so multi-line pages keep their
            reading order. Set False for single-line recognition output.

    Returns:
        NFC-composed text with control characters removed, horizontal
        whitespace collapsed to single spaces, and trailing spaces stripped
        from each line.

    Notes:
        Idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
        This property is asserted in ``tests/test_normalize.py`` because
        non-idempotent normalization makes cached results drift on re-save.
    """
    if not raw:
        return ""

    # NFC first: composition is canonical, so it cannot lose information.
    # We deliberately do not use NFKC - compatibility decomposition would
    # rewrite some Indic forms in ways the recognizer never saw in training.
    composed = unicodedata.normalize("NFC", raw)

    # Remove control characters but keep \n (and \t, converted below).
    no_controls = _CONTROL_RE.sub("", composed)

    # Tabs become spaces; runs of horizontal whitespace collapse to one space.
    collapsed = _HSPACE_RE.sub(" ", no_controls.replace("\t", " "))

    if not keep_line_breaks:
        return collapsed.strip()

    # Normalize line-by-line so indentation does not leak into exported text.
    lines = [line.strip() for line in collapsed.split("\n")]
    joined = "\n".join(lines)
    return _MULTI_NEWLINE_RE.sub("\n\n", joined).strip()


def contains_devanagari(text: str) -> bool:
    """Return True if ``text`` holds at least one Devanagari codepoint.

    Args:
        text: Any string.

    Returns:
        True when a code point falls inside a Devanagari/Vedic range.
        ZWNJ/ZWJ count, because they only appear in Indic conjuncts.

    Why not a regex: a compiled ``str`` scan is faster for the short strings
    we evaluate per line, and the range table doubles as documentation.
    """
    return any(
        any(low <= ord(ch) <= high for low, high in _DEVANAGARI_RANGES)
        for ch in text
    )


def to_latin_digits(text: str) -> str:
    """Translate Devanagari digits (०-९) to ASCII digits (0-9).

    Args:
        text: Input string, may mix scripts.

    Returns:
        A new string with only the digit codepoints swapped; every other
        character - including matras - is untouched.
    """
    return text.translate(str.maketrans(DEVANAGARI_DIGITS, "0123456789"))


def to_devanagari_digits(text: str) -> str:
    """Translate ASCII digits (0-9) to Devanagari digits (०-९).

    Args:
        text: Input string, may mix scripts.

    Returns:
        A new string with only ASCII digit codepoints swapped.
    """
    return text.translate(str.maketrans("0123456789", DEVANAGARI_DIGITS))


def devanagari_ratio(text: str) -> float:
    """Fraction of non-space characters that are Devanagari.

    Used by the API to flag likely misdetections (e.g. the engine returning
    Latin garbage because the page was rotated). A low ratio on a document the
    user marked as Hindi is a better signal than raw model confidence.

    Args:
        text: Candidate OCR output.

    Returns:
        Value in [0.0, 1.0]; 0.0 for empty or whitespace-only input.
    """
    meaningful = [ch for ch in text if not ch.isspace()]
    if not meaningful:
        return 0.0
    hits = sum(1 for ch in meaningful if contains_devanagari(ch))
    return hits / len(meaningful)
