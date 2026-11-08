#!/usr/bin/env bash
# End-to-end smoke test against a live deployment.
#
#   bash scripts/smoke_test.sh https://varnam-ocr-api.onrender.com
#
# Checks, in order:
#   1. /health/ready returns 200 (model loaded, not just process alive)
#   2. /openapi.json is served
#   3. POST /v1/ocr with a generated Devanagari page returns non-empty text
#
# The sample page is generated with the Python standard library only, so this
# script has no dependency on Pillow, ImageMagick, or a checked-in binary.
set -euo pipefail

API_URL="${1:-http://127.0.0.1:8000}"
API_URL="${API_URL%/}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Fail fast with a clear message instead of a confusing connection error.
command -v curl >/dev/null || { echo "error: curl is required"; exit 1; }
command -v python3 >/dev/null || command -v python >/dev/null || { echo "error: python is required"; exit 1; }
PY="$(command -v python3 || command -v python)"

echo "==> 1/3 health"
for attempt in $(seq 1 20); do
  if curl -fsS -o /dev/null "$API_URL/health/ready"; then
    echo "    ready after ${attempt} attempt(s)"
    break
  fi
  if [ "$attempt" -eq 20 ]; then
    echo "error: $API_URL/health/ready never became healthy"
    exit 1
  fi
  sleep 3
done

echo "==> 2/3 openapi"
curl -fsS -o /dev/null "$API_URL/openapi.json"
echo "    schema served"

echo "==> 3/3 ocr"
"$PY" - "$TMP_DIR/page.png" <<'PYTHON'
import struct, sys, zlib

# Build a 240x120 grayscale PNG with three horizontal bars, entirely with the
# standard library. Three bars exercise segmentation; a 1x1 pixel would not.
W, H = 240, 120
rows = []
for y in range(H):
    row = bytearray([0])  # filter byte: none
    for x in range(W):
        inker = (20 <= y < 34) or (52 <= y < 66) or (84 <= y < 98)
        inside = 20 <= x < 220
        row.append(0 if (inker and inside) else 255)
    rows.append(bytes(row))
raw = b"".join(rows)

def chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )

png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 0, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(raw, 9))
    + chunk(b"IEND", b"")
)
open(sys.argv[1], "wb").write(png)
PYTHON

RESPONSE="$(curl -fsS -X POST "$API_URL/v1/ocr" -F "file=@$TMP_DIR/page.png;type=image/png")"

"$PY" - "$RESPONSE" <<'PYTHON'
import json, sys

payload = json.loads(sys.argv[1])
if payload.get("status") != "succeeded":
    print(f"    FAIL: {payload}")
    raise SystemExit(1)
if not payload.get("text", "").strip():
    print("    FAIL: empty text")
    raise SystemExit(1)
print(f"    ok: engine={payload['engine']} lines={len(payload['lines'])} "
      f"confidence={payload['mean_confidence']:.2f} {payload['duration_ms']:.0f}ms")
PYTHON

echo "smoke test passed against $API_URL"
