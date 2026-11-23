"""Authentication API endpoints.

Implements API §2.1, API §3.1, and SECURITY §3:
- POST /api/v1/auth/login: Authenticates credentials, evaluates lockout, returns JWT access token.
- GET /api/v1/auth/me: Returns authenticated user profile.
- POST /api/v1/auth/change-password: Validates current credentials and applies new password.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import get_current_user
from retailpulse.api.limiter import limiter
from retailpulse.config import get_settings
from retailpulse.db.models.app import AppUser
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError
from retailpulse.security.auth import authenticate_user, change_user_password
from retailpulse.security.jwt import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])


class TokenResponse(BaseModel):
    """OAuth2 access token response model."""

    model_config = ConfigDict(extra="forbid")

    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int


class UserProfileResponse(BaseModel):
    """Current authenticated user profile model."""

    model_config = ConfigDict(extra="forbid")

    user_id: uuid.UUID
    email: str
    full_name: str | None
    role: str


class ChangePasswordRequest(BaseModel):
    """Password modification payload."""

    model_config = ConfigDict(extra="forbid")

    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=12, max_length=128)


@router.post(
    "/login",
    summary="User Login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("5/minute", key_func=get_remote_address)
async def login(
    request: Request,
    session: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Authenticate with username (email) and password, returning an access token.

    Accepts both application/x-www-form-urlencoded (OAuth2 standard) and application/json.
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        email = str(body.get("username") or body.get("email") or "")
        password = str(body.get("password") or "")
    else:
        form = await request.form()
        email = str(form.get("username") or form.get("email") or "")
        password = str(form.get("password") or "")

    if not email or not password:
        raise DomainValidationError("Both username/email and password are required")

    user = authenticate_user(session, email, password)

    settings = get_settings()
    expires_seconds = settings.jwt_expire_minutes * 60
    access_token = create_access_token(user_id=user.user_id, role=user.role)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",  # noqa: S106
        expires_in=expires_seconds,
    )


@router.get(
    "/me",
    summary="Current User Profile",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
)
async def get_me(
    current_user: Annotated[AppUser, Depends(get_current_user)],
) -> UserProfileResponse:
    """Return profile details for the currently authenticated user."""
    return UserProfileResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
    )


@router.post(
    "/change-password",
    summary="Change Password",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def change_password(
    body: ChangePasswordRequest,
    current_user: Annotated[AppUser, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
) -> Response:
    """Update current user's password with old password verification and policy enforcement."""
    change_user_password(
        session=session,
        user=current_user,
        current_password=body.current_password,
        new_password=body.new_password,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
