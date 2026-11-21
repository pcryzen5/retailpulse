"""User management administrative API endpoints.

Implements API §2.1 and SECURITY §4:
- GET /api/v1/users: Paginated list of users (Admin only).
- POST /api/v1/users: Create user with role and Argon2id hashed password (Admin only).
- PATCH /api/v1/users/{user_id}: Update user attributes with last active admin
  protection (Admin only).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import require_role
from retailpulse.db.models.app import AppUser
from retailpulse.db.session import get_db
from retailpulse.errors import ConflictError, DomainValidationError, NotFoundError
from retailpulse.security.auth import ensure_not_last_admin
from retailpulse.security.password import hash_password, validate_password

router = APIRouter(prefix="/users", tags=["Users"])

VALID_ROLES = frozenset({"admin", "analyst", "viewer"})


class UserDetailResponse(BaseModel):
    """Admin-view user details response model."""

    model_config = ConfigDict(extra="forbid")

    user_id: uuid.UUID
    email: str
    full_name: str | None
    role: str
    is_active: bool
    failed_login_count: int
    locked_until: datetime.datetime | None
    last_login_at: datetime.datetime | None
    created_at: datetime.datetime


class UserListEnvelope(BaseModel):
    """Paginated user listing envelope."""

    model_config = ConfigDict(extra="forbid")

    items: list[UserDetailResponse]
    page: int
    page_size: int
    total: int


class CreateUserRequest(BaseModel):
    """User creation payload."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    full_name: str | None = None
    role: str = Field(default="analyst")
    password: str = Field(min_length=12, max_length=128)


class UpdateUserRequest(BaseModel):
    """User modification payload."""

    model_config = ConfigDict(extra="forbid")

    role: str | None = None
    is_active: bool | None = None
    full_name: str | None = None
    password: str | None = Field(default=None, min_length=12, max_length=128)


@router.get(
    "",
    summary="List Users",
    response_model=UserListEnvelope,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("admin"))],
)
async def list_users(
    session: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, description="Items per page")] = 50,
) -> UserListEnvelope:
    """Return paginated list of all users (Admin only)."""
    offset = (page - 1) * page_size
    total = session.scalar(sa.select(sa.func.count(AppUser.user_id))) or 0

    users = (
        session.execute(
            sa.select(AppUser).order_by(AppUser.created_at.desc()).offset(offset).limit(page_size)
        )
        .scalars()
        .all()
    )

    items = [
        UserDetailResponse(
            user_id=u.user_id,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            failed_login_count=u.failed_login_count,
            locked_until=u.locked_until,
            last_login_at=u.last_login_at,
            created_at=u.created_at,
        )
        for u in users
    ]

    return UserListEnvelope(
        items=items,
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post(
    "",
    summary="Create User",
    response_model=UserDetailResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("admin"))],
)
async def create_user(
    body: CreateUserRequest,
    session: Annotated[Session, Depends(get_db)],
) -> UserDetailResponse:
    """Create a new user account with validated role and password (Admin only)."""
    clean_email = body.email.strip().lower()
    clean_role = body.role.strip().lower()

    if clean_role not in VALID_ROLES:
        raise DomainValidationError(
            f"Invalid role '{body.role}'. Must be one of: {sorted(VALID_ROLES)}"
        )

    # Check email uniqueness
    existing = session.execute(
        sa.select(AppUser.user_id).where(AppUser.email == clean_email)
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError(f"User with email '{clean_email}' already exists")

    validate_password(body.password)
    pwd_hash = hash_password(body.password, validate=False)

    new_user = AppUser(
        user_id=uuid.uuid4(),
        email=clean_email,
        full_name=body.full_name.strip() if body.full_name else None,
        role=clean_role,
        password_hash=pwd_hash,
        is_active=True,
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return UserDetailResponse(
        user_id=new_user.user_id,
        email=new_user.email,
        full_name=new_user.full_name,
        role=new_user.role,
        is_active=new_user.is_active,
        failed_login_count=new_user.failed_login_count,
        locked_until=new_user.locked_until,
        last_login_at=new_user.last_login_at,
        created_at=new_user.created_at,
    )


@router.patch(
    "/{user_id}",
    summary="Update User",
    response_model=UserDetailResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("admin"))],
)
async def update_user(
    user_id: uuid.UUID,
    body: UpdateUserRequest,
    session: Annotated[Session, Depends(get_db)],
) -> UserDetailResponse:
    """Update role, status, full name, or password with last-admin protection (Admin only)."""
    user = session.execute(
        sa.select(AppUser).where(AppUser.user_id == user_id)
    ).scalar_one_or_none()

    if user is None:
        raise NotFoundError(f"User with ID '{user_id}' not found")

    new_role = body.role.strip().lower() if body.role else None
    if new_role is not None and new_role not in VALID_ROLES:
        raise DomainValidationError(
            f"Invalid role '{body.role}'. Must be one of: {sorted(VALID_ROLES)}"
        )

    # SEC-AUTHZ-03: Last active admin protection
    ensure_not_last_admin(
        session=session,
        target_user=user,
        new_role=new_role,
        new_is_active=body.is_active,
    )

    if new_role is not None:
        user.role = new_role
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.full_name is not None:
        user.full_name = body.full_name.strip() if body.full_name else None
    if body.password is not None:
        validate_password(body.password)
        user.password_hash = hash_password(body.password, validate=False)

    session.commit()
    session.refresh(user)

    return UserDetailResponse(
        user_id=user.user_id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        failed_login_count=user.failed_login_count,
        locked_until=user.locked_until,
        last_login_at=user.last_login_at,
        created_at=user.created_at,
    )
