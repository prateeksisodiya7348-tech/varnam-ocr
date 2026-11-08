"""Training and evaluation tooling for Varnam OCR.

This package never ships to production. The API only ever loads the exported
``.onnx`` artifact, which is deliberate: PyTorch is ~2GB of wheels and would
blow the free-tier image budget, while onnxruntime is ~40MB.
"""
