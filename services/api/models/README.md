# Model artifacts
#
# This directory is (intentionally) empty in the repository. Model weights do
# not belong in git, and the API is designed to run without them.
#
# VARNAM_ENGINE=auto behaviour:
#   * rec.onnx present  -> the real ONNX recognizer is loaded
#   * rec.onnx missing  -> the deterministic stub engine is used, and the API
#                          reports `"engine": "fake"` so nothing is hidden
#
# To add a real model:
#   1. uv pip install -r requirements-train.txt   (repo root: ml/requirements-train.txt)
#   2. python -m ml.export_onnx --checkpoint <path> --out services/api/models
#   3. Restart the API.
#
# Expected files:
#   rec.onnx    - CTC/CRNN recognizer, NCHW float32, input (1, 1, 32, 320)
#   charset.txt - one character per line; line 1 is the CTC blank index 0
#
# Verify a model before deploying:
#   python -m ml.verify --model services/api/models/rec.onnx --charset services/api/models/charset.txt
