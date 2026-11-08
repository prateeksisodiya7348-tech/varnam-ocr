"""Smoke-test an exported model before deploying it.

    python -m ml.verify --model services/api/models/rec.onnx \
                        --charset services/api/models/charset.txt

This is the check that catches the three failures which otherwise only appear in
production: wrong axis order, a charset with no blank at index 0, and an input
shape that does not match what the API feeds the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def verify(model_path: Path, charset_path: Path, *, expected_width: int = 320) -> tuple[bool, list[str]]:
    """Run structural checks against the exported artifacts.

    Args:
        model_path: Path to ``rec.onnx``.
        charset_path: Path to ``charset.txt``.
        expected_width: Input width the API will send.

    Returns:
        ``(ok, problems)`` where ``problems`` is a human-readable list.
    """
    problems: list[str] = []

    if not model_path.exists():
        return False, [f"model not found: {model_path}"]
    if not charset_path.exists():
        return False, [f"charset not found: {charset_path}"]

    charset = charset_path.read_text(encoding="utf-8").splitlines()
    if len(charset) < 10:
        problems.append(f"charset has only {len(charset)} entries; expected the Devanagari set")
    if not charset or charset[0] != "<blank>":
        problems.append("charset line 1 must be <blank> (CTC index 0)")

    try:
        import onnxruntime as ort  # noqa: PLC0415 - optional dependency
    except ImportError:
        return False, ["onnxruntime is not installed"]

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    # Dynamic axes are represented as strings; only check the fixed ones.
    dims = [dim if isinstance(dim, int) else None for dim in shape]
    if dims[1] not in (1, None):
        problems.append(f"expected 1 input channel, model wants {dims[1]}")
    if dims[2] not in (32, None):
        problems.append(f"expected input height 32, model wants {dims[2]}")
    if dims[3] not in (expected_width, None):
        problems.append(f"expected input width {expected_width}, model wants {dims[3]}")

    probe = np.zeros((1, 1, 32, expected_width), dtype=np.float32)
    try:
        output = session.run(None, {model_input.name: probe})[0]
    except Exception as exc:  # noqa: BLE001 - the runtime raises its own types
        return False, [*problems, f"inference failed: {exc}"]

    squeezed = np.squeeze(output)
    if squeezed.ndim != 2:
        problems.append(f"expected a 2-D (time, classes) output after squeeze, got {squeezed.shape}")
    else:
        time_steps, classes = squeezed.shape
        if classes != len(charset):
            problems.append(f"model outputs {classes} classes but charset has {len(charset)} entries")
        if time_steps < 10:
            problems.append(f"only {time_steps} time steps; the model may be over-downsampling")

    return not problems, problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        0 when the artifacts are usable, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Verify exported ONNX artifacts.")
    parser.add_argument("--model", type=Path, default=Path("services/api/models/rec.onnx"))
    parser.add_argument("--charset", type=Path, default=Path("services/api/models/charset.txt"))
    args = parser.parse_args(argv)

    ok, problems = verify(args.model, args.charset)
    if ok:
        print("OK: model and charset are consistent and inference runs.")
        return 0

    print("FAILED:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
