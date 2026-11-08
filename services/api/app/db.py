"""Database engine and session management.

Why SQLModel: one class defines the table *and* the response shape, so the API
cannot drift from the schema without a test failing. SQLite is the local default
to keep a fresh clone zero-setup; production points ``DATABASE_URL`` at Postgres.

Migrations: ``create_all`` is used for the demo deployment because it is
idempotent and needs no separate step on a free tier. A production system with
schema evolution should add Alembic - see docs/LIMITATIONS.md for that and the
one-line change it requires.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine as SAEngine
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings
from app.core.logging import get_logger
from app.models.db import OcrJob  # noqa: F401 - imported so metadata sees the table

logger = get_logger(__name__)

# A managed Postgres can take a few seconds to accept connections right after a
# deploy or after waking from an idle suspension. Crashing on the first failed
# connect turns a transient race into a "failed deploy", so we retry.
_INIT_ATTEMPTS = 6
_INIT_BASE_DELAY_SECONDS = 1.5


def _configure_sqlite(engine: SAEngine) -> None:
    """Apply SQLite pragmas that make it behave under concurrent writes.

    Args:
        engine: The SQLAlchemy engine being created.

    Why: SQLite defaults to a rollback journal and no WAL, which produces
    "database is locked" the moment two OCR jobs finish simultaneously. WAL plus
    a busy timeout is the standard fix and costs two PRAGMAs.
    """

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


@lru_cache(maxsize=1)
def get_engine() -> SAEngine:
    """Create (once) and return the process-wide SQLAlchemy engine.

    Returns:
        A configured engine. Connection pooling is left at SQLAlchemy defaults,
        which are correct for both SQLite and a pooled Neon/Render URL.
    """
    settings = get_settings()
    url = settings.database_url

    # check_same_thread=False is required because FastAPI runs sync routes in a
    # threadpool; SQLAlchemy still serializes access per connection.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    # pool_pre_ping costs one round trip per checkout and saves the entire class
    # of "server closed the connection unexpectedly" errors that a sleeping
    # managed database produces after idle periods.
    engine = create_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args=connect_args,
    )

    if url.startswith("sqlite"):
        _configure_sqlite(engine)

    logger.info("database_engine_created", dialect=engine.dialect.name)
    return engine


def init_db(*, attempts: int = _INIT_ATTEMPTS) -> bool:
    """Create missing tables, retrying while the database becomes reachable.

    Args:
        attempts: How many times to try before giving up.

    Returns:
        True when the schema is ready, False when every attempt failed.

    Why this does not raise: a free-tier Postgres is briefly unreachable right
    after a deploy, and raising here kills the process - which Render reports as
    a failed deploy even though the code is fine. Returning False keeps the
    service up, so ``/health/ready`` answers and a later request can retry the
    connection instead of the whole container dying.
    """
    last_error = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            SQLModel.metadata.create_all(get_engine())
            logger.info("database_ready", attempt=attempt)
            return True
        except Exception as exc:  # noqa: BLE001 - any driver error is retryable here
            last_error = str(exc)[:200]
            if attempt == attempts:
                break
            delay = _INIT_BASE_DELAY_SECONDS * attempt
            logger.warning(
                "database_init_retry",
                attempt=attempt,
                of=attempts,
                retry_in_seconds=delay,
                error=last_error,
            )
            time.sleep(delay)

    logger.error("database_init_failed", error=last_error)
    return False


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session.

    Yields:
        A session that is always closed, even when the route raises.
    """
    with Session(get_engine()) as session:
        yield session
