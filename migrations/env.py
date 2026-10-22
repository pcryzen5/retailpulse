"""Alembic runtime environment configuration for RetailPulse migrations."""

from __future__ import annotations

import logging
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure all models are registered with Base.metadata
import retailpulse.db.models  # noqa: F401
from retailpulse.config import get_settings
from retailpulse.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

target_metadata = Base.metadata


def get_url() -> str:
    """Retrieve database connection URL from environment settings or config."""
    # Check if custom RP_DB_OWNER_URL or RP_DATABASE_URL is provided in settings
    try:
        settings = get_settings()
        # If an owner URL is present in env, prefer it for migrations
        owner_url = os.environ.get("RP_DB_OWNER_URL")
        if owner_url:
            return owner_url
        if settings.database_url:
            return settings.database_url
    except Exception as exc:
        logger.warning("Could not read settings from config.py: %s; using alembic.ini URL", exc)

    return str(config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an
    Engine is acceptable here as well. By skipping the Engine creation we don't
    even need a DBAPI to be available. Calls to context.execute() here emit the
    given string to the script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection with
    the context.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
