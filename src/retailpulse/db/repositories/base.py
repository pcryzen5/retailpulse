"""Generic base repository for entity persistence and querying."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import DeclarativeBase, Session

ModelT = TypeVar("ModelT", bound=DeclarativeBase)


class BaseRepository(Generic[ModelT]):
    """Generic repository providing standardized CRUD operations over SQLAlchemy models."""

    def __init__(self, session: Session, model_cls: type[ModelT]) -> None:
        """Initialize repository with an active session and model class.

        Args:
            session: Active SQLAlchemy session.
            model_cls: Target entity model class.
        """
        self.session = session
        self.model_cls = model_cls

    def get_by_id(self, ident: Any) -> ModelT | None:
        """Retrieve an entity by primary key identifier."""
        return self.session.get(self.model_cls, ident)

    def list_all(self, *, limit: int | None = None, offset: int = 0) -> Sequence[ModelT]:
        """List entities with optional pagination."""
        stmt = select(self.model_cls).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def count(self) -> int:
        """Count total entities in table."""
        stmt = select(func.count()).select_from(self.model_cls)
        return self.session.scalar(stmt) or 0

    def add(self, entity: ModelT) -> ModelT:
        """Add an entity instance to the session."""
        self.session.add(entity)
        return entity

    def add_all(self, entities: Sequence[ModelT]) -> None:
        """Add multiple entity instances to the session."""
        self.session.add_all(entities)

    def delete(self, entity: ModelT) -> None:
        """Mark an entity for deletion."""
        self.session.delete(entity)

    def flush(self) -> None:
        """Flush pending changes to the database within the current transaction."""
        self.session.flush()
