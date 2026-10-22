"""Create PostgreSQL extensions and application schemas.

Revision ID: 0001
Revises: None
Create Date: 2026-10-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create extensions citext and pg_trgm, and schemas app, ops, core, and ml."""
    op.execute("CREATE EXTENSION IF NOT EXISTS citext;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    op.execute("CREATE SCHEMA IF NOT EXISTS app;")
    op.execute("CREATE SCHEMA IF NOT EXISTS ops;")
    op.execute("CREATE SCHEMA IF NOT EXISTS core;")
    op.execute("CREATE SCHEMA IF NOT EXISTS ml;")


def downgrade() -> None:
    """Drop schemas and extensions."""
    op.execute("DROP SCHEMA IF EXISTS ml CASCADE;")
    op.execute("DROP SCHEMA IF EXISTS core CASCADE;")
    op.execute("DROP SCHEMA IF EXISTS ops CASCADE;")
    op.execute("DROP SCHEMA IF EXISTS app CASCADE;")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm;")
    op.execute("DROP EXTENSION IF EXISTS citext;")
