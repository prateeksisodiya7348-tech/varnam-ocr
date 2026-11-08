"""Shared pytest fixtures.

The environment is configured *before* the application package is imported,
because ``app.config.get_settings`` is cached and ``app.db.get_engine`` is
cached too. Setting env vars later would silently reuse the developer's real
database, which is exactly the kind of test-isolation bug that erodes trust in
a suite.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

# ---- environment must be set before importing `app` ------------------------
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="varnam-test-"))
os.environ["APP_ENV"] = "development"
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_ROOT / 'test.db').as_posix()}"
os.environ["VARNAM_ENGINE"] = "fake"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = str(_TMP_ROOT / "uploads")
# A high limit keeps the rate-limit tests independent of the functional tests.
os.environ["RATE_LIMIT_PER_MINUTE"] = "500"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    """Session-wide temporary directory for uploads and the test database."""
    return _TMP_ROOT


@pytest.fixture
def client() -> TestClient:
    """A TestClient bound to an app with a fake engine and temp storage.

    Yields:
        A client whose context manager triggers the real lifespan, so startup
        logic (DB table creation, service wiring) is exercised by every test.
    """
    from app.main import create_app

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def sample_png() -> bytes:
    """A synthetic 'page' with four solid text-like bars.

    Returns:
        PNG bytes.

    Why bars and not real text: rendering Devanagari needs a font file, which
    would make the suite depend on the host's font set. Bars exercise the exact
    code path (binarize -> row projection -> line boxes) without that coupling.
    """
    image = Image.new("L", (900, 500), color=255)
    draw = ImageDraw.Draw(image)
    for index in range(4):
        top = 60 + index * 100
        draw.rectangle([80, top, 820, top + 34], fill=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
