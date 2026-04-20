"""
Alembic environment configuration.

Uses psycopg2 (synchronous) for migrations — asyncpg is the runtime driver
but Alembic requires a synchronous connection.

The DATABASE_URL is read from the .env file via app.core.config.settings so
there is a single source of truth for the connection string.
"""

import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Make sure the backend package is on sys.path when running
# `alembic` from the backend/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Interpret the config file's logging section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Pull DATABASE_URL from app settings ───────────────────────────────────────
from app.core.config import settings  # noqa: E402

# asyncpg uses postgresql:// — psycopg2 also accepts postgresql://, so we just
# make sure it's not the asyncpg-only postgresql+asyncpg:// scheme.
db_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
# configparser treats % as an interpolation character — escape all % in the URL
config.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))

# We don't use SQLAlchemy ORM models, so target_metadata stays None.
# Alembic will rely solely on the explicit op.create_table() calls in each
# migration file to determine what the schema should look like.
target_metadata = None


# ── Migration runners ─────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection (outputs SQL to stdout).
    Useful for reviewing what will be executed before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
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
