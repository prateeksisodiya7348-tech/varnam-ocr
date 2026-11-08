# Resume material

Use only the bullets whose numbers you have actually produced. Anything marked
`[fill from EVALUATION.md]` must be replaced with a real measurement — an
interviewer who asks "how did you measure that?" will find an invented number in
about ten seconds.

---

## Project entry

**Varnam OCR — Devanagari OCR platform** · TypeScript, Python, FastAPI, Next.js, ONNX
[github.com/&lt;you&gt;/varnam-ocr] · [live demo]

## Bullets

- Built and deployed an end-to-end Devanagari OCR system (printed + handwritten)
  serving a Next.js 15 / React 19 frontend and a FastAPI + ONNX Runtime inference
  service, running entirely on free-tier infrastructure (Vercel, Render, Neon, R2).

- Trained a CRNN + CTC recognizer on IIIT-ILST, CVIT-OCR and DHCD, exported to
  ONNX, and added both a quantization parity gate and a load-test that falls back
  to float32 when the INT8 graph cannot run on the CPU provider.

- Achieved **[X]% CER on printed and [Y]% on handwritten held-out slices**,
  measured per slice by a reproducible evaluation CLI wired to the same pipeline
  that serves production — not a separate notebook path.

- Cut inference memory to fit a 512 MB free instance by disabling the ONNX memory
  arena, pinning a single intra-op thread, and capping input at 2000 px, keeping
  peak RSS at ~200 MB per A4 page at a p95 of **[Z] ms**.

- Eliminated train/serve skew by extracting Devanagari normalization — matras,
  virama, nukta, anusvara, danda — into one shared module imported by both the
  training pipeline and the API, with tests pinning each combining-mark case.

- Designed a human-in-the-loop feedback loop that stores user corrections
  alongside model predictions as line-attributed training samples, turning
  production traffic into a domain-specific fine-tuning corpus.

- Hardened the upload boundary with magic-byte validation, full-decode checks,
  decompression-bomb guards and content-addressed storage, and shipped structured
  JSON logging with request correlation plus Prometheus latency histograms.

- Containerised the API in a multi-stage, non-root image under 200 MB with a
  readiness-gated health check, and gated `main` behind CI covering ruff, mypy,
  pytest (75% coverage floor), eslint, tsc, Vitest and a Docker build.

## Condensed (3 lines, for a one-page resume)

- End-to-end Devanagari OCR (print + handwriting): CRNN/CTC in PyTorch, INT8 ONNX
  export, FastAPI serving, Next.js UI with bbox overlay, editable transcript and
  hOCR export — deployed free on Vercel + Render + Neon.
- **[X]% CER printed / [Y]% handwritten** on held-out slices, measured per slice by
  a reproducible CLI that exercises the production inference path.
- Fit a full OCR stack into a 512 MB instance: INT8 weights, 1 ORT thread, arena
  disabled, 2000 px input cap, semaphore-bounded inference returning clean 503s
  instead of OOM kills.

## Talking points to prepare (not resume text)

1. **Why the model is not on Vercel** — bundle and timeout limits; this one answer
   demonstrates systems thinking more than any accuracy number.
2. **Why no mark stripping** — the Devanagari abugida argument. Concrete, memorable,
   and shows domain care.
3. **The inverse coordinate transform** — a bug that looks like a model failure but
   is a units failure.
4. **What you left out** — Alembic, word-level recognition, handwriting data. Being
   able to name your own gaps is a senior signal.
