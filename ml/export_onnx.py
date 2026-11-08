"""Export a trained checkpoint to INT8 ONNX.

    python -m ml.export_onnx --checkpoint artifacts/best.pt \
        --charset data/charset.txt --out services/api/models

Why ONNX at all: it removes PyTorch (~2GB of wheels, slow cold start) from the
serving image and lets onnxruntime use its own integer kernels. On a 512MB
instance this is the difference between "deploys" and "OOM-killed".

Why INT8 dynamic quantization: it shrinks the weights 4x and speeds up the LSTM
matmuls roughly 2-3x on x86, at a typical CER cost of well under half a point.
Dynamic (per-batch) is chosen over static because static needs a calibration set
and its accuracy win does not justify the operational complexity here.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from ml.config import TrainConfig
from ml.model import CrnnRecognizer, count_parameters


def load_checkpoint(path: Path, *, num_classes: int) -> CrnnRecognizer:
    """Rebuild the model and load its weights.

    Args:
        path: Checkpoint produced by ``ml.train``.
        num_classes: Charset size, must match the checkpoint.

    Returns:
        A model in eval mode.

    Raises:
        ValueError: When the checkpoint's class count disagrees with the charset.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_classes = int(checkpoint["num_classes"])
    if checkpoint_classes != num_classes:
        raise ValueError(
            f"Checkpoint was trained with {checkpoint_classes} classes but the charset "
            f"has {num_classes}. Regenerate the charset or retrain; silently proceeding "
            "would produce garbage output with no error."
        )

    model = CrnnRecognizer(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _session_runs(path: Path, config: TrainConfig) -> bool:
    """Check that onnxruntime can actually load and run an exported graph.

    Args:
        path: ONNX file to test.
        config: Geometry for the probe input.

    Returns:
        True when a full inference completes.

    Why this exists: dynamic INT8 quantization happily rewrites Conv layers into
    ``ConvInteger`` nodes that onnxruntime's CPU provider does not implement. The
    resulting file is a valid .onnx that cannot be loaded - the service would die
    at startup with a ``NOT_IMPLEMENTED`` error. Validating with a real session is
    the only reliable way to catch it before deployment.
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415 - heavy import

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        probe = np.zeros((1, 1, config.image_height, config.image_width), dtype=np.float32)
        session.run(None, {"input": probe})
        return True
    except Exception:  # noqa: BLE001 - any failure means "not shippable"
        return False


def export(model: CrnnRecognizer, config: TrainConfig, destination: Path) -> Path:
    """Export the model to ONNX, quantizing only when the result still loads.

    Args:
        model: Trained model in eval mode.
        config: Geometry used for the dummy input.
        destination: Where to write ``rec.onnx``.

    Returns:
        Path to the shippable model (INT8 when it works, otherwise float32).

    Why the fallback: a 16MB float model that runs beats a 4MB model that crashes
    the service. On a 512MB instance both fit; correctness wins.
    """
    destination.mkdir(parents=True, exist_ok=True)
    float_path = destination / "rec_float.onnx"
    quantized_path = destination / "rec_quantized.onnx"
    final_path = destination / "rec.onnx"

    dummy = torch.randn(1, 1, config.image_height, config.image_width)

    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["logits"],
        # dynamic_axes: batch and time are dynamic so the runtime can accept any
        # line width; height and channels stay fixed because the CNN downsamples by
        # a fixed factor.
        "dynamic_axes": {"input": {0: "batch", 3: "width"}, "logits": {1: "batch", 0: "time"}},
        "opset_version": 17,
        "do_constant_folding": True,
    }

    try:
        # dynamo=False selects the legacy TorchScript exporter. The dynamo path
        # (default in newer torch) imports `onnxscript`, an extra dependency that
        # exists only for the exporter - our graph is fully static and the legacy
        # exporter traces it correctly. Depending on it would mean every training
        # install needs a package we never otherwise use.
        torch.onnx.export(model, dummy, float_path.as_posix(), dynamo=False, **export_kwargs)
    except TypeError:
        # torch versions that predate the `dynamo` keyword.
        torch.onnx.export(model, dummy, float_path.as_posix(), **export_kwargs)

    quantized_ok = False
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415 - heavy import

        quantize_dynamic(
            model_input=float_path.as_posix(),
            model_output=quantized_path.as_posix(),
            weight_type=QuantType.QInt8,
        )
        # Dynamic quantization rewrites Conv into ConvInteger, which the CPU
        # execution provider does not implement. Only accept the INT8 model if a
        # real session can actually run it.
        quantized_ok = _session_runs(quantized_path, config)
    except Exception as exc:  # noqa: BLE001 - quantization is best-effort
        print(f"  quantization skipped: {exc}")

    if quantized_ok:
        quantized_path.replace(final_path)
        float_path.unlink(missing_ok=True)
        print("  model format: INT8 (dynamic quantized)")
        return final_path

    # Fall back to the float graph. A model that loads is worth more than a third
    # of the size that crashes the service at startup.
    print("  model format: float32 (INT8 graph was not loadable)")
    quantized_path.unlink(missing_ok=True)
    float_path.replace(final_path)
    return final_path


def _parity_check(original: CrnnRecognizer, onnx_path: Path, config: TrainConfig) -> float:
    """Compare the ONNX output against PyTorch on the same input.

    Args:
        original: The PyTorch model.
        onnx_path: Exported model path.
        config: Geometry for the probe input.

    Returns:
        Maximum absolute difference between the two argmax decodings, expressed
        as the fraction of positions that disagree.

    Why a parity check: quantization is lossy by design. If it changes more than
    a few percent of the predicted characters, the export is broken (usually a
    transposed axis) and must fail loudly here rather than in production.
    """
    import onnxruntime as ort  # noqa: PLC0415 - heavy import

    probe = np.random.default_rng(0).random((1, 1, config.image_height, config.image_width))
    probe = (probe * 2 - 1).astype(np.float32)

    with torch.no_grad():
        torch_logits = original(torch.from_numpy(probe)).numpy()
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": probe})[0]

    torch_best = torch_logits.argmax(axis=-1).ravel()
    onnx_best = onnx_logits.argmax(axis=-1).ravel()
    length = min(len(torch_best), len(onnx_best))
    disagreements = int((torch_best[:length] != onnx_best[:length]).sum())
    return disagreements / max(1, length)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Export the CRNN recognizer to INT8 ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--charset", type=Path, default=Path("data/charset.txt"))
    parser.add_argument("--out", type=Path, default=Path("services/api/models"))
    parser.add_argument(
        "--max-disagreement",
        type=float,
        default=0.05,
        help="Fail the export when quantization changes more than this fraction of predictions.",
    )
    args = parser.parse_args(argv)

    if not args.charset.exists():
        print(f"error: charset not found at {args.charset}")
        return 2

    charset_lines = args.charset.read_text(encoding="utf-8").splitlines()
    config = TrainConfig(charset_path=args.charset)

    model = load_checkpoint(args.checkpoint, num_classes=len(charset_lines))
    print(f"parameters: {count_parameters(model):,}")

    onnx_path = export(model, config, args.out)
    size_mb = onnx_path.stat().st_size / 1_048_576
    print(f"exported: {onnx_path} ({size_mb:.1f} MB)")

    disagreement = _parity_check(model, onnx_path, config)
    print(f"quantization disagreement: {disagreement:.2%}")
    if disagreement > args.max_disagreement:
        print("error: quantization changed too many predictions; investigate before deploying.")
        return 1

    # The charset must sit next to the model: the engine resolves both from
    # MODEL_DIR, and shipping only the model produces an unbootable container.
    target_charset = args.out / "charset.txt"
    if args.charset.resolve() != target_charset.resolve():
        shutil.copyfile(args.charset, target_charset)
    print(f"charset: {target_charset} ({len(charset_lines)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
