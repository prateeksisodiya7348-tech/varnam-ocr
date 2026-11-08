"""Database engine and session management.

Why SQLModel: one class defines the table *and* the response shape, so the API
cannot drift from the schema without a test failing. SQLite is the local
default to keep a fresh clone zero-setup; production points ``DATABASE_URL`` at
Neon Postgres.

Migrations: ``create_all`` is used for the demo deployment because it is
idempotent and needs no separate step on a free tier. A production system with
schema evolution should add Alembic - see docs/LIMITATIONS.md for that and the
one-line change it requires.
"""

from __future__ import annotations

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
        which are correct for both SQLite and a pooled Neon URL.
    """
    settings = get_settings()
    url = settings.database_url

    # check_same_thread=False is required because FastAPI runs sync routes in a
    # threadpool; SQLAlchemy still serializes access per connection.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, echo=False, pool_pre_ping=True, connect_args=connect_args)

    if url.startswith("sqlite"):
        _configure_sqlite(engine)

    logger.info("database_ready", dialect=engine.dialect.name)
    return engine


def init_db() -> None:
    """Create missing tables.

    Idempotent: safe to call on every container start, which is exactly what a
    free-tier host that restarts containers needs.
    """
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session.

    Yields:
        A session that is always closed, even when the route raises.
    """
    with Session(get_engine()) as session:
        yield session
