"""Pre-deployment validation for the Render blueprint and Docker image inputs.

Run before clicking Deploy. It catches the failures that only surface 6 minutes
into a cloud build:

  * render.yaml missing/renamed keys or a database reference that does not match
  * a Dockerfile COPY path that exists locally but was never committed
  * a missing model or charset (the image builds fine, then crashes at startup)

    python scripts/predeploy_check.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
problems: list[str] = []
notes: list[str] = []


def require(condition: bool, message: str) -> None:
    """Record a failure when ``condition`` is false."""
    if not condition:
        problems.append(message)


# --------------------------------------------------------------- render.yaml
blueprint_path = ROOT / "render.yaml"
require(blueprint_path.exists(), "render.yaml is missing from the repo root")

if blueprint_path.exists():
    try:
        import yaml  # type: ignore[import-untyped]

        blueprint = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse failure is fatal here
        problems.append(f"render.yaml is not valid YAML: {exc}")
        blueprint = {}

    databases = blueprint.get("databases") or []
    services = blueprint.get("services") or []

    require(bool(databases), "no `databases:` block - DATABASE_URL would come from nowhere")
    require(len(services) == 1, f"expected exactly one service, found {len(services)}")

    db_name = databases[0].get("name") if databases else None
    notes.append(f"database: {db_name}")

    if services:
        service = services[0]
        require(service.get("runtime") == "docker", "service runtime must be `docker`")
        require(
            service.get("dockerfilePath") == "./services/api/Dockerfile",
            "dockerfilePath must be ./services/api/Dockerfile",
        )
        # The image needs packages/ as well as services/api/, so the build context
        # has to be the repo root.
        require(service.get("dockerContext") == ".", "dockerContext must be `.` (repo root)")
        require(
            service.get("healthCheckPath") == "/health/ready",
            "healthCheckPath must be /health/ready",
        )
        notes.append(f"service : {service.get('name')} ({service.get('plan')})")

        env_vars = {item.get("key"): item for item in service.get("envVars", [])}
        database_url = env_vars.get("DATABASE_URL", {})
        from_db = database_url.get("fromDatabase", {})
        require(
            from_db.get("name") == db_name,
            "DATABASE_URL.fromDatabase must reference the database above",
        )
        require("CORS_ORIGINS" in env_vars, "CORS_ORIGINS is not configured")

# ------------------------------------------------------------------ Dockerfile
dockerfile = ROOT / "services" / "api" / "Dockerfile"
require(dockerfile.exists(), "services/api/Dockerfile is missing")

if dockerfile.exists():
    text = dockerfile.read_text(encoding="utf-8")
    for copy_source in ("packages/", "services/api/app/", "services/api/models/"):
        require(copy_source in text, f"Dockerfile never copies {copy_source}")
        require((ROOT / copy_source.rstrip("/")).exists(), f"{copy_source} does not exist locally")
    # The CMD is a JSON array, so the flag appears as two quoted elements:
    #   ..., "--workers", "1", ...
    require('"--workers"' in text, "Dockerfile CMD does not pin --workers")
    require('"--workers", "1"' in text, "uvicorn must run a single worker on 512MB")

# --------------------------------------------------------------- db driver
# A Postgres URL without a driver installed is a guaranteed runtime crash: the
# image builds, the service starts, then the first DB call raises
# ModuleNotFoundError. Catch it here instead of 8 minutes into a cloud build.
requirements = ROOT / "services" / "api" / "requirements.txt"
if requirements.exists():
    pinned = requirements.read_text(encoding="utf-8")
    if blueprint:
        env_map = {
            item.get("key"): item
            for item in (blueprint.get("services") or [{}])[0].get("envVars", [])
        }
        uses_postgres = "DATABASE_URL" in env_map and "fromDatabase" in env_map["DATABASE_URL"]
        if uses_postgres:
            require(
                "psycopg2-binary" in pinned or "psycopg[" in pinned,
                "DATABASE_URL points at Postgres but requirements.txt has no driver "
                "(add psycopg2-binary)",
            )
            notes.append("db driver: present")

# ------------------------------------------------------------------- artifacts
model = ROOT / "services" / "api" / "models" / "rec.onnx"
charset = ROOT / "services" / "api" / "models" / "charset.txt"

require(model.exists(), "services/api/models/rec.onnx is missing (image would fall back to the stub)")
if model.exists():
    size_mb = model.stat().st_size / 1_048_576
    notes.append(f"model   : {size_mb:.1f} MB")
    require(size_mb < 100, f"model is {size_mb:.0f} MB; too large for the free image budget")

require(charset.exists(), "services/api/models/charset.txt is missing")
if charset.exists():
    classes = len(charset.read_text(encoding="utf-8").splitlines())
    notes.append(f"charset : {classes} classes")
    require(classes > 10, f"charset has only {classes} entries")

# -------------------------------------------------------------- git coverage
try:
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "origin/main", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.splitlines()
except Exception as exc:  # noqa: BLE001 - git may be unavailable offline
    notes.append(f"git check skipped: {exc}")
    tracked = []

if tracked:
    notes.append(f"committed: {len(tracked)} files")
    for required_file in (
        "render.yaml",
        "services/api/Dockerfile",
        "services/api/models/rec.onnx",
        "services/api/models/charset.txt",
        "services/api/requirements.txt",
    ):
        require(
            required_file in tracked,
            f"{required_file} is not committed - Render will not see it",
        )

    # Every importable module must be in git. A `.gitignore` entry like `models/`
    # or `data/` also matches Python packages (services/api/app/models,
    # ml/data); they keep working locally and vanish from the image, so the
    # container dies at import time with ModuleNotFoundError.
    missing_modules: list[str] = []
    for package_dir in (
        ROOT / "services" / "api" / "app",
        ROOT / "ml",
        ROOT / "packages",
    ):
        for path in package_dir.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative not in tracked:
                missing_modules.append(relative)

    require(
        not missing_modules,
        f"{len(missing_modules)} Python file(s) are not committed and will break the "
        f"container: {', '.join(sorted(missing_modules)[:5])}",
    )

# --------------------------------------------------------------------- report
print("Pre-deploy check\n")
for note in notes:
    print(f"  {note}")

if problems:
    print("\nFAILED:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print("\nAll checks passed - safe to deploy.")
