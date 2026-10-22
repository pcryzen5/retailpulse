"""Unit tests for database session, transactions, and repository abstractions."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from retailpulse.db.repositories.base import BaseRepository
from retailpulse.db.session import (
    create_db_engine,
    get_db,
    get_session_factory,
    session_scope,
)

test_metadata = sa.MetaData()


class DummyBase(DeclarativeBase):
    """Declarative base for unit test dummy models."""

    metadata = test_metadata


class DummyItem(DummyBase):
    """Dummy entity model for repository testing."""

    __tablename__ = "dummy_item"
    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    name: Mapped[str] = mapped_column(sa.String, nullable=False)


@pytest.fixture
def sqlite_engine() -> sa.Engine:
    """Create in-memory SQLite engine for fast unit tests."""
    engine = create_db_engine("sqlite:///:memory:")
    test_metadata.create_all(engine)
    return engine


def test_session_scope_commit(sqlite_engine: sa.Engine) -> None:
    """Verify that session_scope commits successfully on normal completion."""
    with session_scope(sqlite_engine) as session:
        session.add(DummyItem(id=1, name="Item 1"))

    # Verify persisted in new session
    factory = get_session_factory(sqlite_engine)
    with factory() as session:
        item = session.get(DummyItem, 1)
        assert item is not None
        assert item.name == "Item 1"


def test_session_scope_rollback(sqlite_engine: sa.Engine) -> None:
    """Verify that session_scope automatically rolls back when an exception is raised."""

    def _trigger_rollback() -> None:
        with session_scope(sqlite_engine) as session:
            session.add(DummyItem(id=2, name="Failed Item"))
            raise RuntimeError("deliberate failure")

    with pytest.raises(RuntimeError, match="deliberate failure"):
        _trigger_rollback()

    # Verify not persisted
    factory = get_session_factory(sqlite_engine)
    with factory() as session:
        item = session.get(DummyItem, 2)
        assert item is None


def test_get_db_generator(sqlite_engine: sa.Engine) -> None:
    """Verify get_db generator yields active session and closes upon exit."""
    gen = get_db(sqlite_engine)
    session = next(gen)
    assert session.is_active
    with pytest.raises(StopIteration):
        next(gen)


def test_base_repository_operations(sqlite_engine: sa.Engine) -> None:
    """Verify BaseRepository CRUD operations."""
    factory = get_session_factory(sqlite_engine)
    with factory() as session:
        repo: BaseRepository[DummyItem] = BaseRepository(session, DummyItem)

        # Add single
        repo.add(DummyItem(id=10, name="Alpha"))
        session.commit()

        # Add multiple
        repo.add_all([DummyItem(id=11, name="Beta"), DummyItem(id=12, name="Gamma")])
        session.commit()

        # Count & list_all
        assert repo.count() == 3
        items = repo.list_all(limit=2, offset=0)
        assert len(items) == 2
        assert items[0].name == "Alpha"

        # Get by id
        fetched = repo.get_by_id(11)
        assert fetched is not None
        assert fetched.name == "Beta"

        # Delete
        repo.delete(fetched)
        session.commit()
        assert repo.get_by_id(11) is None
        assert repo.count() == 2
