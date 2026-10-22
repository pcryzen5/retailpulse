"""Database engine, session management, and connectivity verification."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from retailpulse.config import get_settings
from retailpulse.errors import DatabaseError
from retailpulse.logging import get_logger

logger = get_logger("retailpulse.db.session")


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create a new SQLAlchemy sync engine with production connection pool defaults."""
    db_url = url or get_settings().database_url
    engine_kwargs: dict[str, Any] = {
        "echo": echo,
        "pool_pre_ping": True,
    }

    # PostgreSQL-specific connection pool tuning
    if "postgres" in db_url:
        engine_kwargs["pool_size"] = 10
        engine_kwargs["max_overflow"] = 20
        engine_kwargs["connect_args"] = {"connect_timeout": 5}

    return sa.create_engine(db_url, **engine_kwargs)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the cached singleton database engine configured from application settings."""
    return create_db_engine()


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a configured session factory bound to the provided or default engine."""
    eng = engine or get_engine()
    return sessionmaker(
        bind=eng,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(engine: Engine | None = None) -> Generator[Session, None, None]:
    """Provide a transactional scope around operations with automatic commit and rollback."""
    factory = get_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Database session transaction rolled back due to error", error=str(exc))
        raise
    finally:
        session.close()


def get_db(engine: Engine | None = None) -> Generator[Session, None, None]:
    """Yield a database session for request lifecycle (FastAPI dependency)."""
    factory = get_session_factory(engine)
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


def check_database(
    engine: Engine | None = None,
    config_path: str | Path = "alembic.ini",
) -> dict[str, Any]:
    """Verify database connectivity and check migration state against Alembic head.

    Args:
        engine: Optional Engine instance; uses default if omitted.
        config_path: Path to alembic.ini configuration.

    Returns:
        Dictionary containing connectivity, current revision, head revision, and up-to-date status.

    Raises:
        DatabaseError: If connectivity cannot be established.
    """
    eng = engine or get_engine()
    result: dict[str, Any] = {
        "connected": False,
        "current_revision": None,
        "head_revision": None,
        "is_up_to_date": False,
        "error": None,
    }

    try:
        with eng.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
            result["connected"] = True

            migration_ctx = MigrationContext.configure(conn)
            result["current_revision"] = migration_ctx.get_current_revision()

        alembic_cfg_path = Path(config_path)
        if alembic_cfg_path.exists():
            cfg = Config(str(alembic_cfg_path))
            script = ScriptDirectory.from_config(cfg)
            result["head_revision"] = script.get_current_head()

            # Up to date when current matches head (and both are not None)
            if (
                result["current_revision"] is not None
                and result["current_revision"] == result["head_revision"]
            ):
                result["is_up_to_date"] = True
            elif result["current_revision"] is None:
                result["is_up_to_date"] = False
        else:
            logger.warning("Alembic configuration not found at path", path=str(config_path))

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Database health check failed", error=str(exc))
        raise DatabaseError(f"Database health check failed: {exc}") from exc

    return result
