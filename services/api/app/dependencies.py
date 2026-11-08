"""Process-wide singletons exposed as FastAPI dependencies.

Why an app-state container instead of module globals: tests build an app with a
fake engine and a temp directory, and module-level globals would leak between
tests. Holding them on ``app.state`` keeps each app instance isolated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.core.ratelimit import TokenBucketLimiter
from app.services.engine import OcrEngine
from app.services.storage import StorageBackend


@dataclass(slots=True)
class AppServices:
    """Everything a route needs that is expensive to build.

    Attributes:
        engine: The OCR backend.
        storage: Upload persistence.
        limiter: Per-client token bucket.
        settings: Validated configuration.
    """

    engine: OcrEngine
    storage: StorageBackend
    limiter: TokenBucketLimiter
    settings: Settings

    def client_key(self, request: Request) -> str:
        """Derive a stable, non-reversible client identifier.

        Args:
            request: The incoming request.

        Returns:
            A truncated SHA-256 of the client host plus a salt.

        Why hashed: rate limiting needs a stable key, but storing raw IPs is
        personal data. Hashing keeps the limiter exact while making the log
        line non-identifying.
        """
        host = request.client.host if request.client else "unknown"
        # Trusting X-Forwarded-For here would let a client bypass the bucket by
        # spoofing the header. Render terminates TLS and sets client.host, so we
        # use that instead of parsing headers ourselves.
        digest = hashlib.sha256(f"{host}:{self.settings.api_shared_secret}".encode()).hexdigest()
        return digest[:16]


def get_services(request: Request) -> AppServices:
    """FastAPI dependency returning the app's service container.

    Args:
        request: Used to reach ``request.app.state``.

    Returns:
        The container built during application startup.
    """
    return request.app.state.services


SettingsDep = Annotated[Settings, Depends(get_settings)]
ServicesDep = Annotated[AppServices, Depends(get_services)]
