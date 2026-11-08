# RUNBOOK — Varnam OCR, start to finish

Everything needed to take this project from a fresh clone to a deployed service,
in the order it must happen. Each phase states what to run, what it touches in
the code, and what "success" looks like.

**Time:** ~45 minutes total (30 of which is model training).

---

## Part 0 — What the system is

```
Browser  ──►  Next.js (3000)  ──HTTP──►  FastAPI (8010)  ──►  ONNX model
                                              │
                                              └──►  SQLite / Postgres (jobs, feedback)
```

Four layers, each in its own directory:

| Layer | Directory | Language | Responsibility |
| --- | --- | --- | --- |
| Web | `apps/web` | TypeScript | upload, viewer, editor, exports |
| API | `services/api` | Python | validation, pipeline, persistence |
| ML | `ml` | Python | training, evaluation, ONNX export |
| Shared | `packages/devanagari_text` | Python | text normalization used by both ML and API |

The shared package exists so training and serving normalize text identically.
Without it, the two drift and CER silently degrades.

---

## Part 1 — One-time environment setup

### 1.1 Tools

| Tool | Why |
| --- | --- |
| Python 3.12 | the pinned wheels (numpy, onnxruntime) have no 3.13+/3.14 builds |
| Node 20+ | Next.js 15 |
| uv | creates the venv and installs packages (uv venvs have no pip) |
| pnpm | frontend package manager |

```powershell
# Python 3.12 without touching the system install
uv python install 3.12
```

### 1.2 Python environment

```powershell
cd C:\dev\varnam-ocr\services\api
$uv = "$env:USERPROFILE\.local\bin\uv.exe"

& $uv venv --python 3.12 .venv
& $uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
& $uv pip install --python .\.venv\Scripts\python.exe -r requirements-dev.txt
```

**Touches:** creates `services/api/.venv`.

**Why requirements.txt and not pyproject:** the runtime set is a deployment
contract. A flat, pinned file makes the Docker build reproducible and needs no
build backend.

### 1.3 Training extras (only if you will train)

```powershell
& $uv pip install --python .\.venv\Scripts\python.exe torch onnx onnxscript
```

~250 MB. Skipped by anyone who only serves an existing `rec.onnx`.

### 1.4 Frontend packages

```powershell
cd C:\dev\varnam-ocr
& "$env:APPDATA\npm\pnpm.cmd" install
```

### 1.5 Environment files

```powershell
Copy-Item .env.example services\api\.env              # optional; defaults work
Copy-Item apps\web\.env.example apps\web\.env.local   # points web at the API
```

`apps\web\.env.local` must contain:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8010
```

**Verify:** `Test-Path services\api\.venv\Scripts\python.exe` → `True`

---

## Part 2 — Generate training data (2 min)

Synthetic Devanagari is rendered by **Windows GDI**, not Pillow. Pillow on Windows
ships without libraqm, so it cannot shape Devanagari — it draws the i-matra after
the consonant instead of before it, producing images no real page contains.

```powershell
cd C:\dev\varnam-ocr

powershell -ExecutionPolicy Bypass -File scripts\gen_synthetic.ps1 `
    -Out data\synth\train -Count 2500 -Seed 11

powershell -ExecutionPolicy Bypass -File scripts\gen_synthetic.ps1 `
    -Out data\synth\eval -Count 400 -Seed 99
```

**Touches:** `scripts/gen_synthetic.ps1` reads `data/corpus_hi.txt`, writes
`<Out>/00001.png` + `00001.txt` pairs.

**Success:** `2500` and `400` PNG files, `skipped=0` in the output.

### 2.1 Build manifests

```powershell
$env:PYTHONPATH = "C:\dev\varnam-ocr;C:\dev\varnam-ocr\packages"
$py = ".\services\api\.venv\Scripts\python.exe"

& $py -m ml.data.prepare --images data\synth\train --manifest data\train_manifest.csv --slice synthetic
& $py -m ml.data.prepare --images data\synth\eval  --manifest data\eval_manifest.csv  --slice synthetic
```

**Touches:** `ml/data/prepare.py` writes CSV rows `image_path,text,slice` with
paths relative to the manifest's own folder — so the CSV stays portable.

---

## Part 3 — Train (25-30 min on CPU)

```powershell
$env:PYTHONPATH = "C:\dev\varnam-ocr;C:\dev\varnam-ocr\packages"
$env:PYTHONIOENCODING = "utf-8"

& $py -u -m ml.train `
    --manifest data\train_manifest.csv `
    --charset data\charset.txt `
    --epochs 20 --batch-size 24 --lr 0.0006 `
    --out artifacts
```

### What the code does, in order

| File | Step |
| --- | --- |
| `ml/data/manifest.py` | reads CSV, normalizes ground truth via the shared module |
| `ml/data/manifest.py::build_charset` | derives the 89-class charset from the corpus (blank first, space included) |
| `ml/model.py::CrnnRecognizer` | CNN → BiLSTM → linear head |
| `ml/train.py::LineDataset` | resizes each line to 32×320, pads with paper-white |
| `ml/train.py::collate` | pads labels, computes CTC input lengths |
| `ml/train.py::validate` | greedy CTC decode → CER |

**Expected log:**

```
train=2125 val=375 classes=89
device=cpu parameters=4,160,025
epoch 01 loss=3.7782 val_cer=100.00%
epoch 05 loss=0.2073 val_cer=2.61%
epoch 07 loss=0.0488 val_cer=0.58%   <- checkpoint saved
```

**Why the CER stays at 100% for three epochs:** CTC has to discover the blank
class and rough alignment before any character is correct. This is normal, not a
bug.

**Output:** `artifacts/best.pt` (~16 MB).

---

## Part 4 — Export to ONNX (1 min)

```powershell
& $py -m ml.export_onnx `
    --checkpoint artifacts\best.pt `
    --charset data\charset.txt `
    --out services\api\models
```

### What happens inside `ml/export_onnx.py`

1. Load the checkpoint and verify its class count matches the charset.
2. Trace the model to ONNX with **`dynamo=False`** (the legacy exporter needs no
   extra dependency; the dynamo path imports `onnxscript`).
3. Attempt INT8 dynamic quantization.
4. **Load-test the quantized graph.** Dynamic INT8 rewrites Conv into
   `ConvInteger`, which onnxruntime's CPU provider does not implement. If the
   graph cannot run, the exporter keeps the float model.
5. Parity-check predictions against PyTorch; fail if >5% disagree.
6. Copy `charset.txt` next to the model.

**Expected:**

```
parameters: 4,160,025
model format: float32 (INT8 graph was not loadable)
exported: services\api\models\rec.onnx (15.9 MB)
quantization disagreement: 0.00%
charset: services\api\models\charset.txt (89 classes)
```

---

## Part 5 — Run locally (10 sec)

```powershell
cd C:\dev\varnam-ocr
powershell -ExecutionPolicy Bypass -File scripts\stop-all.ps1   # free the ports
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1  # start both
```

`start-all.ps1` opens two windows, waits for both health checks, then opens the
browser.

**Stop with:** `scripts\stop-all.ps1` (kills by port, not by process name, so it
never touches your other projects).

---

## Part 6 — Code walkthrough: what happens on one upload

```
POST /v1/ocr  (multipart)
│
├─ app/main.py :: _observe            request id, timing, metrics
├─ app/main.py :: _security_headers   CSP, nosniff, frame-deny
│
├─ app/routes/ocr.py :: create_ocr
│    └─ _enforce_rate_limit           token bucket, per client hash
│
├─ app/services/runner.py :: process_image_upload
│    ├─ validation.validate_upload    magic bytes → size cap → full decode
│    ├─ storage.put                   content-addressed key (sha256)
│    ├─ preprocess.prepare_image      EXIF transpose → grayscale → 2000px cap
│    ├─ export.run_page
│    │    ├─ engine.recognize
│    │    │    ├─ segmentation.binarize      Otsu (hand-written NumPy)
│    │    │    ├─ segmentation.segment_lines row projection
│    │    │    ├─ segmentation.sort_reading_order  row clustering
│    │    │    ├─ engine._to_tensor          pad to ink ratio → 32×320
│    │    │    └─ engine._recognize_line     softmax → CTC collapse
│    │    └─ devanagari_text.normalize_text  NFC, marks preserved
│    └─ session.add(OcrJob)           result JSON persisted
│
└─ OcrResponse (Pydantic)             boxes inverse-scaled to original pixels
```

Two details that are easy to get wrong and are handled explicitly:

- **Coordinate space.** Boxes are produced in preprocessed pixels and inverse-scaled
  in `runner._scaled_bbox` before serialization. Skipping this makes every overlay
  drift toward the top-left on a downscaled photo.
- **Ink ratio.** Segments are cropped tight to ink, but the model was trained on
  canvases where ink was ~59% of the height. `engine._INK_HEIGHT_FRACTION` pads the
  crop back to that ratio. Getting this wrong took CER from 0.3% to 80%.

---

## Part 7 — Verify end to end (2 min)

```powershell
# 1. model loaded?
curl.exe http://127.0.0.1:8010/health/ready
#    {"status":"ok","engine":"onnx","model_loaded":true}

# 2. real inference
curl.exe -s -o "$env:TEMP\ocr.json" http://127.0.0.1:8010/v1/ocr `
    -F "file=@data\synth\eval\00001.png"

# 3. read the answer
& $py -c "import json,os; d=json.load(open(os.environ['TEMP']+r'\ocr.json',encoding='utf-8')); print(d['engine'], len(d['lines']), d['mean_confidence'])"
```

**Then the UI:** <http://localhost:3000/app> → drop `data\synth\eval\00001.png`.

---

## Part 8 — Measure accuracy (3 min)

```powershell
$env:PYTHONPATH = "C:\dev\varnam-ocr;C:\dev\varnam-ocr\packages"
$env:MODEL_DIR  = "C:\dev\varnam-ocr\services\api\models"

& $py -m ml.eval --manifest data\eval_manifest.csv --engine onnx --out docs\EVALUATION.md
```

**Expected:** `overall CER=7.42% WER=18.62%` on 400 independent images.

`docs/EVALUATION.md` is generated, never hand-written. Quote the file, not a memory
of a number.

---

## Part 9 — Quality gates (2 min)

```powershell
cd C:\dev\varnam-ocr\services\api
& $py -m ruff check .
& $py -m ruff format --check .
& $py -m mypy app
& $py -m pytest -q

cd C:\dev\varnam-ocr
& "$env:APPDATA\npm\pnpm.cmd" --filter @varnam/web typecheck
& "$env:APPDATA\npm\pnpm.cmd" --filter @varnam/web lint
& "$env:APPDATA\npm\pnpm.cmd" --filter @varnam/web test
& "$env:APPDATA\npm\pnpm.cmd" --filter @varnam/web build
```

All eight must pass before a deploy. The CI workflow runs the same set.

---

## Part 10 — Deploy (~20 min)

Full detail in `docs/DEPLOYMENT.md`. Order matters:

1. **GitHub** — create an empty repo, then `git push -u origin main`.
2. **Neon** — Postgres, copy the **pooled** connection string.
3. **Render** — New → Blueprint → `render.yaml`; set `DATABASE_URL` and
   `CORS_ORIGINS`. Build takes 5-8 min.
4. **Vercel** — import repo, **Root Directory = `apps/web`**, set
   `NEXT_PUBLIC_API_BASE_URL` to the Render URL.
5. **CORS** — set `CORS_ORIGINS` on Render to the exact Vercel URL, no trailing slash.
6. **Verify** — `bash scripts/smoke_test.sh https://<api-url>`.

---

## Part 11 — Daily use

```powershell
cd C:\dev\varnam-ocr
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1   # work
powershell -ExecutionPolicy Bypass -File scripts\stop-all.ps1    # done
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `No module named 'app'` | wrong directory | run from `services\api` |
| `No module named 'devanagari_text'` | `PYTHONPATH` missing | include `<repo>;<repo>\packages` |
| `No module named pip` | uv venv has no pip | use `uv pip install --python .\.venv\Scripts\python.exe` |
| `engine: "fake"` | `rec.onnx` or `charset.txt` missing | export again; both files must be in `services\api\models` |
| `NOT_IMPLEMENTED: ConvInteger` | INT8 graph shipped | re-export; the load-test should have caught it |
| CER ~80% | ink-ratio padding missing | check `_INK_HEIGHT_FRACTION` in `engine.py` |
| gibberish on real photos | trained on synthetic only | fine-tune on real labelled lines |
| CORS error in browser | origin mismatch | exact Vercel URL, no trailing slash |
| `NEXT_PUBLIC_*` change ignored | inlined at build time | redeploy on Vercel |
