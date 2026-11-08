# Limitations

Written before anyone asks, because a system that hides its boundaries is a
system nobody should trust with real documents.

---

## 1. Recognition accuracy

| Area | Status | Why | Next step |
| --- | --- | --- | --- |
| Clean printed Devanagari | Good | In-distribution | — |
| Scanned books | Moderate | Noise, skew, bleed-through | Add scan-augmentation to training |
| Handwriting | **Weak** | Isolated-character datasets do not teach connected writing | Collect line-level handwriting data |
| Rare conjuncts (`स्त्र`, `द्व्य`) | Weak | Under-represented in training distributions | Weighted sampling by conjunct frequency |
| Mixed Latin + Devanagari | Weak | ASCII class set is small and under-trained | Train a dedicated code-switch slice |
| Very small text (<14px) | Weak | 32px canvas throws away detail | Raise input height to 48 and retrain |

**The honest headline:** this is a credible *printed* Devanagari OCR. Handwriting
is a demo, not a product. Saying so is more valuable than inflating a number.

## 2. Layout

- No learned layout model. Line finding is a projection profile, which assumes
  roughly horizontal text.
- Multi-column splitting is a gutter heuristic: it needs a visible vertical gap
  in >70% of lines, and it does nothing for tables.
- Reading order across sidebars, footnotes, headers and captions is not modelled.
- Rotated pages beyond ~5° are not deskewed. A 90° rotation fails completely
  (EXIF orientation is handled; arbitrary rotation is not).
- Vertical Devanagari text is not supported.

## 3. Word-level output

Word boxes come from vertical projection gaps, but **word text is distributed
proportionally by box width**, not produced by a separate word recognizer. That
means word boundaries are approximate on tightly-spaced text. The JSON says so;
a consumer that needs precise word alignment needs a second model pass.

## 4. Infrastructure (free tier)

- **512MB RAM** is the binding constraint. Batch inference is impossible; a
  single A4 page at 2000px already uses ~200MB peak.
- **Cold start of 15-30s** after ~15 minutes idle. Handled in the UI, not hidden.
- **Single instance** → the rate limiter is per-process and resets on restart.
  Correct for one instance, wrong the moment there are two.
- **No persistent disk** → uploads go to R2 in production; the local backend is
  development-only.
- **No background worker** → PDF jobs run as FastAPI background tasks inside the
  web process. A long PDF competes with API traffic. A Celery/RQ worker is the
  correct fix.

## 5. Database

- **No Alembic migrations.** Tables are created with `SQLModel.metadata.create_all`.
  Fine for a first deploy and a demo, wrong the first time a column changes on a
  database with data. Adding Alembic is a one-file change and it should be done
  before any real traffic.
- No retention policy: `ocr_jobs.result` grows forever. A production deployment
  needs a TTL or a nightly prune.
- No per-user isolation. Jobs are addressable by UUID, which is
  unguessable-but-public. Real multi-tenancy needs auth on `GET /v1/jobs/{id}`.

## 6. Security

- No authentication on the OCR endpoint (deliberately, for a public demo). The
  only protection is per-IP rate limiting.
- `API_SHARED_SECRET` exists but is used only to salt the client hash, not to
  gate access.
- Feedback samples are unauthenticated, so a malicious client could poison the
  training corpus. Any production system needs review before those samples enter
  a training run.
- No antivirus/AV scanning on uploads. Magic-byte checks are not malware scanning.

## 7. Scripts

Only Devanagari (Hindi, Marathi, Nepali, Sanskrit). Bengali, Gujarati, Gurmukhi,
Tamil, Telugu and others are **not** in the charset. Feeding them in produces
confident-looking wrong glyphs, which is worse than an error — the API does not
currently detect "wrong script" beyond reporting `devanagari_ratio`.

## 8. What is deliberately not built

- **A trained public checkpoint.** Weights are not distributed here; the training
  and export pipeline is. Datasets carry their own licenses and shipping weights
  derived from them is a legal question, not a technical one.
- **A mobile app.** The web app is responsive and supports camera capture via the
  file input; that was judged sufficient.
- **User accounts and history.** Jobs are listed by id only.
- **Real-time streaming OCR** of a camera feed.

## 9. Known rough edges

- `ml/eval.py` prints `engine=auto` in the report header when `--engine` is
  omitted, even though the resolved engine may be `fake`. The per-sample results
  are correct; only the label is imprecise.
- `FakeOcrEngine` reuses four fixed sentences cyclically. Segment count and boxes
  are real; only the text is canned. This is intentional (it exercises the UI
  without a model) but it means a screenshot from the stub engine is not evidence
  of accuracy.
- The `docker-compose.yml` Postgres URL uses the `psycopg` driver, which requires
  `psycopg[binary]` if you switch away from SQLite. It is not in
  `requirements.txt` because the default deployment path does not need it.
