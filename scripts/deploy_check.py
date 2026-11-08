#!/usr/bin/env python
"""Pre-deployment configuration check.

    python scripts/deploy_check.py

Why this exists: the failure modes that actually bite on a free tier are all
configuration, not code - SQLite on an ephemeral container, a CORS list missing
the Vercel domain, a model that was never copied into the image. Each of those
turns into a 30-minute debugging session if it is discovered after deploy. This
script turns them into a 1-second local check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from the repo root without installing anything.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "api"))
sys.path.insert(0, str(REPO_ROOT / "packages"))

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def main() -> int:
    """Run every check and print a summary table.

    Returns:
        0 when nothing failed, 1 otherwise.
    """
    results: list[tuple[str, str, str]] = []

    def record(status: str, name: str, detail: str) -> None:
        results.append((status, name, detail))

    env = os.environ.get("APP_ENV", "development").lower()
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./varnam.db")
    cors = os.environ.get("CORS_ORIGINS", "")
    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    engine = os.environ.get("VARNAM_ENGINE", "auto")

    # --- environment -------------------------------------------------------
    record(PASS, "APP_ENV", env)
    if env != "production":
        record(WARN, "APP_ENV", "not 'production'; security headers and JSON logs stay relaxed")

    # --- database ----------------------------------------------------------
    if database_url.startswith("sqlite"):
        if env == "production":
            record(FAIL, "DATABASE_URL", "SQLite in production loses data on every redeploy")
        else:
            record(WARN, "DATABASE_URL", "SQLite (fine locally, never in production)")
    elif "sslmode=require" in database_url or "neon.tech" in database_url:
        record(PASS, "DATABASE_URL", "managed Postgres with TLS")
    else:
        record(WARN, "DATABASE_URL", "Postgres without an explicit sslmode; verify TLS is on")

    # --- CORS --------------------------------------------------------------
    if not cors:
        record(FAIL, "CORS_ORIGINS", "empty; the browser will block every request")
    elif "localhost" in cors and env == "production":
        record(WARN, "CORS_ORIGINS", "still includes localhost")
    elif "https://" not in cors:
        record(WARN, "CORS_ORIGINS", "no https origin configured")
    else:
        record(PASS, "CORS_ORIGINS", cors)

    # --- model -------------------------------------------------------------
    model_path = model_dir / os.environ.get("REC_MODEL_FILE", "rec.onnx")
    charset_path = model_dir / os.environ.get("CHARSET_FILE", "charset.txt")
    if engine == "fake":
        record(WARN, "VARNAM_ENGINE", "explicitly 'fake' - stub text will be served")
    elif model_path.exists() and charset_path.exists():
        size_mb = model_path.stat().st_size / 1_048_576
        if size_mb > 200:
            record(WARN, "Model size", f"{size_mb:.0f}MB - risky on a 512MB instance")
        else:
            record(PASS, "Model size", f"{size_mb:.1f}MB")
        record(PASS, "Charset", f"{len(charset_path.read_text(encoding='utf-8').splitlines())} classes")
    else:
        status = FAIL if engine == "onnx" else WARN
        record(status, "Model", f"missing under {model_dir} (engine will fall back to stub)")

    # --- storage -----------------------------------------------------------
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if env == "production" and backend == "local":
        record(WARN, "STORAGE_BACKEND", "local disk on an ephemeral container; uploads vanish")
    elif backend == "r2":
        missing = [key for key in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.environ.get(key)]
        if missing:
            record(FAIL, "R2 credentials", f"missing {', '.join(missing)}")
        else:
            record(PASS, "STORAGE_BACKEND", "r2")
    else:
        record(PASS, "STORAGE_BACKEND", backend)

    # --- secrets -----------------------------------------------------------
    record(
        PASS if os.environ.get("API_SHARED_SECRET") else WARN,
        "API_SHARED_SECRET",
        "set" if os.environ.get("API_SHARED_SECRET") else "empty; per-IP limiting only",
    )

    width = max(len(name) for _, name, _ in results)
    print("\nVarnam OCR - deployment check\n")
    for status, name, detail in results:
        print(f"  [{status}] {name.ljust(width)}  {detail}")

    failures = sum(1 for status, _, _ in results if status == FAIL)
    warnings = sum(1 for status, _, _ in results if status == WARN)
    print(f"\n{failures} failure(s), {warnings} warning(s)\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
