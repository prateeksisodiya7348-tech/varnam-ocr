"""Typed application settings, loaded once from the environment.

Every tunable lives here so that reviewers can answer "where does this number
come from?" in one file. Nothing else in the codebase reads ``os.environ``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EngineName = Literal["fake", "onnx", "auto"]
StorageName = Literal["local", "r2"]


class Settings(BaseSettings):
    """Immutable-by-convention runtime configuration.

    Values are read from the process environment (and a local ``.env``).
    Defaults are chosen so that ``uv run uvicorn app.main:app`` works on a
    fresh clone with no model files and no database server.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- runtime ---------------------------------------------------------
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # --- database --------------------------------------------------------
    database_url: str = "sqlite:///./varnam.db"

    # --- inference -------------------------------------------------------
    varnam_engine: EngineName = "auto"
    model_dir: Path = Path("./models")
    rec_model_file: str = "rec.onnx"
    charset_file: str = "charset.txt"
    ort_intra_op_threads: int = 1
    # Recognition canvas. 32px height x 320px width is the standard CRNN/CTC
    # geometry; wider crops are split so long lines keep their resolution.
    rec_image_height: int = 32
    rec_image_width: int = 320
    rec_max_width_ratio: float = 40.0

    # --- limits ----------------------------------------------------------
    max_upload_mb: int = 10
    max_image_long_edge: int = 2000
    max_pages_per_job: int = 20
    request_timeout_seconds: int = 25
    rate_limit_per_minute: int = 20

    # --- cors ------------------------------------------------------------
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- storage ---------------------------------------------------------
    storage_backend: StorageName = "local"
    local_storage_dir: Path = Path("./uploads")
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = "varnam-ocr"

    # --- auth ------------------------------------------------------------
    api_shared_secret: str = ""

    @field_validator("database_url")
    @classmethod
    def _require_explicit_db_in_prod(cls, value: str, info) -> str:
        """Fail fast when production would silently fall back to SQLite.

        A SQLite file on an ephemeral Render container loses every OCR result
        on redeploy. That is a data-loss bug, not a config typo, so we refuse
        to start instead of logging a warning nobody reads.
        """
        if info.data.get("app_env") == "production" and value.startswith("sqlite"):
            raise ValueError(
                "DATABASE_URL must point at Postgres in production; "
                "SQLite on an ephemeral container loses data on redeploy."
            )
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse the comma-separated CORS allowlist into a clean list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """True when running with production hardening enabled."""
        return self.app_env == "production"

    @property
    def rec_model_path(self) -> Path:
        """Absolute-ish path to the ONNX recognition model."""
        return self.model_dir / self.rec_model_file

    @property
    def charset_path(self) -> Path:
        """Path to the CTC index -> Devanagari character map."""
        return self.model_dir / self.charset_file

    def resolved_engine(self) -> Literal["fake", "onnx"]:
        """Collapse ``auto`` into a concrete engine name.

        ``auto`` picks ONNX only when the model file actually exists, which is
        what makes the repository demo-able before any training run: clone,
        run, get readable output from the stub engine.
        """
        if self.varnam_engine == "auto":
            return "onnx" if self.rec_model_path.exists() else "fake"
        return self.varnam_engine


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached because pydantic validation plus ``.env`` parsing is measurable
    overhead when called per request, and settings never change at runtime.
    """
    return Settings()
