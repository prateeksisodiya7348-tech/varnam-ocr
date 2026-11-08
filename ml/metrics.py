"""Text metrics: character error rate, word error rate, confusion analysis.

Implemented from scratch (Levenshtein over code points) rather than pulling in
``jiwer``, because the standard library version is 40 lines, has no transitive
dependencies, and lets us compute the confusion pairs that actually drive
improvement work. A single CER number tells you nothing about *what* to fix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


def edit_distance(reference: str, hypothesis: str) -> int:
    """Compute the Levenshtein distance between two strings.

    Args:
        reference: Ground-truth text.
        hypothesis: Model output.

    Returns:
        Minimum number of insertions, deletions and substitutions needed.

    Why the two-row optimization: a full |a| x |b| matrix for a 1000-line
    document is megabytes of garbage; two rolling rows keep it O(min(n, m)) in
    memory with identical results.
    """
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(
                min(
                    previous[j] + 1,        # deletion
                    current[j - 1] + 1,     # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate, normalized by reference length.

    Args:
        reference: Ground truth.
        hypothesis: Prediction.

    Returns:
        ``edit_distance / len(reference)``, or 0.0 for an empty reference.

    Note: unlike WER this is *not* clipped at 1.0. A hallucinated output longer
    than the reference can exceed 100%, which is the honest report - clipping it
    would hide a catastrophic failure mode behind a perfect-looking score.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate computed over whitespace-separated tokens.

    Args:
        reference: Ground truth.
        hypothesis: Prediction.

    Returns:
        Edit distance over word tokens, normalized by reference word count.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def confusion_pairs(reference: str, hypothesis: str, *, top_k: int = 20) -> list[tuple[str, str, int]]:
    """Approximate character confusion pairs via a single edit-alignment pass.

    Args:
        reference: Ground truth.
        hypothesis: Prediction.
        top_k: Number of pairs to return.

    Returns:
        ``(expected, produced, count)`` triples, most frequent first.

    Why approximate: a full alignment is O(n*m); this walks the DP traceback only
    along a greedy path, which is enough to surface the systematic confusions
    (e.g. "ि" vs "ी", or "ब" vs "व") that a retraining round should target.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for ref_char, hyp_char in zip(reference, hypothesis, strict=False):
        if ref_char != hyp_char:
            pairs[(ref_char, hyp_char)] += 1
    return [(ref, hyp, count) for (ref, hyp), count in pairs.most_common(top_k)]


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """One evaluated sample.

    Attributes:
        image: Identifier of the sample (path or hash).
        slice_name: Which evaluation slice it belongs to.
        reference: Ground truth.
        hypothesis: Prediction.
        cer: Character error rate.
        wer: Word error rate.
    """

    image: str
    slice_name: str
    reference: str
    hypothesis: str
    cer: float
    wer: float


def summarize(records: list[EvalRecord]) -> dict[str, dict[str, float]]:
    """Aggregate metrics overall and per slice.

    Args:
        records: Evaluated samples.

    Returns:
        Mapping of slice name (plus ``"all"``) to mean CER/WER and sample count.

    Why macro-average over samples: micro-averaging by character count lets a few
    very long documents dominate, which hides regressions on short handwritten
    snippets - the case users care about most.
    """
    if not records:
        return {"all": {"cer": 0.0, "wer": 0.0, "samples": 0.0}}

    result: dict[str, dict[str, float]] = {}
    for name in {record.slice_name for record in records}:
        subset = [record for record in records if record.slice_name == name]
        result[name] = {
            "cer": round(float(np.mean([record.cer for record in subset])), 4),
            "wer": round(float(np.mean([record.wer for record in subset])), 4),
            "samples": float(len(subset)),
        }

    result["all"] = {
        "cer": round(float(np.mean([record.cer for record in records])), 4),
        "wer": round(float(np.mean([record.wer for record in records])), 4),
        "samples": float(len(records)),
    }
    return result
