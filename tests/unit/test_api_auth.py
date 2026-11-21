"""Unit tests for authentication, Argon2id hashing, JWT, account lockout, and RBAC.

Implements P5-02 verification:
- Argon2id password hashing and deny-list policy.
- JWT creation, standard claims, tampering, wrong algorithm, and expiration.
- Account lockout after 5 consecutive failures.
- Generic failure message across unknown users, wrong passwords, and locked accounts.
- /api/v1/auth/* endpoints (login, me, change-password).
- /api/v1/users/* endpoints (list, create, update, last active admin protection).
- require_role dependency enforcing viewer < analyst < admin hierarchy.
- Dynamic DB check (deactivation/role change applies immediately).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Annotated

import jwt
import pytest
import sqlalchemy as sa
from fastapi import APIRouter, Depends
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from retailpulse.api.app import create_app
from retailpulse.api.dependencies import require_role
from retailpulse.db.models.app import AppUser
from retailpulse.db.session import get_db
from retailpulse.errors import (
    AuthenticationError,
    DomainValidationError,
)
from retailpulse.security.auth import (
    authenticate_user,
)
from retailpulse.security.jwt import (
    create_access_token,
    decode_access_token,
)
from retailpulse.security.password import (
    hash_password,
    validate_password,
    verify_password,
)
from retailpulse.timeutil import utc_now


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def auth_db_session():
    """Create an in-memory SQLite database populated with the AppUser schema."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    # Map schemas to None for SQLite in-memory execution
    engine = engine.execution_options(
        schema_translate_map={"app": None, "core": None, "ops": None, "ml": None}
    )
    AppUser.__table__.create(bind=engine, checkfirst=True)

    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def sample_users(auth_db_session: Session) -> dict[str, AppUser]:
    """Seed three users representing Viewer, Analyst, and Admin roles."""
    pwd_hash = hash_password("ValidPassword123!")

    viewer = AppUser(
        user_id=uuid.uuid4(),
        email="viewer@example.com",
        full_name="Test Viewer",
        role="viewer",
        password_hash=pwd_hash,
        is_active=True,
    )
    analyst = AppUser(
        user_id=uuid.uuid4(),
        email="analyst@example.com",
        full_name="Test Analyst",
        role="analyst",
        password_hash=pwd_hash,
        is_active=True,
    )
    admin = AppUser(
        user_id=uuid.uuid4(),
        email="admin@example.com",
        full_name="Test Admin",
        role="admin",
        password_hash=pwd_hash,
        is_active=True,
    )
    auth_db_session.add_all([viewer, analyst, admin])
    auth_db_session.commit()

    return {"viewer": viewer, "analyst": analyst, "admin": admin}


@pytest.fixture
def test_client(auth_db_session: Session):
    """Create a TestClient with overridden get_db dependency and test RBAC endpoints."""
    app = create_app()

    # Route group to test require_role hierarchy
    rbac_router = APIRouter(prefix="/test-rbac")

    @rbac_router.get("/viewer", dependencies=[Depends(require_role("viewer"))])
    def viewer_endpoint(user: Annotated[AppUser, Depends(require_role("viewer"))]):
        return {"role": user.role, "access": "viewer_granted"}

    @rbac_router.get("/analyst", dependencies=[Depends(require_role("analyst"))])
    def analyst_endpoint(user: Annotated[AppUser, Depends(require_role("analyst"))]):
        return {"role": user.role, "access": "analyst_granted"}

    @rbac_router.get("/admin", dependencies=[Depends(require_role("admin"))])
    def admin_endpoint(user: Annotated[AppUser, Depends(require_role("admin"))]):
        return {"role": user.role, "access": "admin_granted"}

    app.include_router(rbac_router)

    session_cls = sessionmaker(bind=auth_db_session.get_bind())

    def override_get_db():
        session = session_cls()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    return TestClient(app, raise_server_exceptions=False)


# =========================================================================
# 1. Password Policy & Argon2id Hashing Tests (SEC-AUTH-01)
# =========================================================================


def test_password_policy_min_length() -> None:
    """Verify passwords shorter than 12 characters are rejected."""
    with pytest.raises(DomainValidationError, match="at least 12 characters"):
        validate_password("short123")


def test_password_policy_max_length() -> None:
    """Verify passwords longer than 128 characters are rejected."""
    with pytest.raises(DomainValidationError, match="must not exceed 128"):
        validate_password("a" * 129)


def test_password_policy_denylist() -> None:
    """Verify predictable passwords in the deny-list are rejected."""
    with pytest.raises(DomainValidationError, match="too common or easily guessable"):
        validate_password("password12345")

    with pytest.raises(DomainValidationError, match="too common or easily guessable"):
        validate_password("123456789012")


def test_argon2id_hashing_and_verification() -> None:
    """Verify Argon2id hashing generates valid hashes and verifies correctly."""
    pwd = "SecurePassphrase123!"
    p_hash = hash_password(pwd)
    assert p_hash.startswith("$argon2id$")

    is_valid, _ = verify_password(p_hash, pwd)
    assert is_valid is True

    is_wrong, _ = verify_password(p_hash, "WrongPassword123!")
    assert is_wrong is False


# =========================================================================
# 2. JWT Verification, Expiry, Tampering, and Algorithm Tests (SEC-AUTH-02)
# =========================================================================


def test_jwt_standard_claims_and_decode() -> None:
    """Verify access token contains sub, role, iat, exp, and jti."""
    u_id = uuid.uuid4()
    token = create_access_token(user_id=u_id, role="analyst")
    payload = decode_access_token(token)

    assert payload["sub"] == str(u_id)
    assert payload["role"] == "analyst"
    assert "iat" in payload
    assert "exp" in payload
    assert "jti" in payload


def test_jwt_expiry_rejected() -> None:
    """Verify expired tokens are rejected with AuthenticationError."""
    u_id = uuid.uuid4()
    # Expired 10 seconds ago
    expired_token = create_access_token(
        user_id=u_id,
        role="viewer",
        expires_delta=datetime.timedelta(seconds=-10),
    )

    with pytest.raises(AuthenticationError, match="expired"):
        decode_access_token(expired_token)


def test_jwt_tampering_signature_rejected() -> None:
    """Verify tampered JWT tokens fail signature verification."""
    token = create_access_token(user_id=uuid.uuid4(), role="analyst")
    # Mutate the payload portion of header.payload.signature
    parts = token.split(".")
    tampered_token = f"{parts[0]}.eyJhZG1pbiI6dHJ1ZX0.{parts[2]}"

    with pytest.raises(AuthenticationError, match="Invalid authentication token"):
        decode_access_token(tampered_token)


def test_jwt_wrong_algorithm_rejected() -> None:
    """Verify tokens signed with unauthorized algorithms (e.g. none, HS384) are rejected."""
    payload = {
        "sub": str(uuid.uuid4()),
        "role": "admin",
        "iat": int(utc_now().timestamp()),
        "exp": int((utc_now() + datetime.timedelta(minutes=30)).timestamp()),
        "jti": str(uuid.uuid4()),
    }
    # Encode with 'none' or different key/algo
    secret = "test-secret-at-least-64-bytes-long-super-secure-key-12345678901234567890!"
    unauthorized_token = jwt.encode(payload, secret, algorithm="HS384")

    with pytest.raises(AuthenticationError, match="Invalid authentication token"):
        # Explicit allowlist in decode_access_token only permits HS256
        decode_access_token(unauthorized_token, secret_key=secret)


# =========================================================================
# 3. Authentication & Account Lockout Tests (SEC-AUTH-03)
# =========================================================================


def test_authenticate_success(auth_db_session: Session, sample_users: dict[str, AppUser]) -> None:
    """Verify successful authentication resets failure count."""
    user = authenticate_user(auth_db_session, "analyst@example.com", "ValidPassword123!")
    assert user.email == "analyst@example.com"
    assert user.failed_login_count == 0
    assert user.last_login_at is not None


def test_authenticate_unknown_user_generic_error(auth_db_session: Session) -> None:
    """Verify unknown user fails with uniform generic error message (SEC-AUTH-03)."""
    with pytest.raises(AuthenticationError, match="Invalid credentials or authentication token"):
        authenticate_user(auth_db_session, "nonexistent@example.com", "ValidPassword123!")


def test_authenticate_wrong_password_lockout_after_5_failures(
    auth_db_session: Session,
    sample_users: dict[str, AppUser],
) -> None:
    """Verify account locks for 15 minutes after 5 consecutive failures."""
    email = "viewer@example.com"

    # 4 consecutive failures
    for i in range(1, 5):
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            authenticate_user(auth_db_session, email, "WrongPassword123!")
        u = auth_db_session.execute(sa.select(AppUser).where(AppUser.email == email)).scalar_one()
        assert u.failed_login_count == i
        assert u.locked_until is None

    # 5th failure: triggers lockout
    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        authenticate_user(auth_db_session, email, "WrongPassword123!")

    locked_user = auth_db_session.execute(
        sa.select(AppUser).where(AppUser.email == email)
    ).scalar_one()
    assert locked_user.failed_login_count == 5
    assert locked_user.locked_until is not None
    locked_dt = (
        locked_user.locked_until.replace(tzinfo=datetime.UTC)
        if locked_user.locked_until.tzinfo is None
        else locked_user.locked_until
    )
    assert locked_dt > utc_now()

    # Subsequent attempt with correct password fails identically while locked
    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        authenticate_user(auth_db_session, email, "ValidPassword123!")


def test_authenticate_inactive_user_generic_error(
    auth_db_session: Session,
    sample_users: dict[str, AppUser],
) -> None:
    """Verify deactivated user fails with uniform generic error message."""
    user = sample_users["viewer"]
    user.is_active = False
    auth_db_session.commit()

    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        authenticate_user(auth_db_session, "viewer@example.com", "ValidPassword123!")


# =========================================================================
# 4. Auth API Endpoints (/api/v1/auth/*)
# =========================================================================


def test_api_login_form_and_json(
    test_client: TestClient,
    sample_users: dict[str, AppUser],
) -> None:
    """Verify /api/v1/auth/login works with both form-urlencoded and JSON."""
    # 1. Form urlencoded
    form_resp = test_client.post(
        "/api/v1/auth/login",
        data={"username": "analyst@example.com", "password": "ValidPassword123!"},
    )
    assert form_resp.status_code == 200
    token_data = form_resp.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["expires_in"] == 1800

    # 2. JSON
    json_resp = test_client.post(
        "/api/v1/auth/login",
        json={"email": "analyst@example.com", "password": "ValidPassword123!"},
    )
    assert json_resp.status_code == 200
    assert "access_token" in json_resp.json()


def test_api_login_failure_returns_401_problem(test_client: TestClient) -> None:
    """Verify failed login returns RFC 9457 401 Problem Details."""
    resp = test_client.post(
        "/api/v1/auth/login",
        json={"email": "baduser@example.com", "password": "WrongPassword123!"},
    )
    assert resp.status_code == 401
    assert resp.headers["Content-Type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:unauthorized"
    assert body["detail"] == "Invalid credentials or authentication token"


def test_api_get_me(
    test_client: TestClient,
    sample_users: dict[str, AppUser],
) -> None:
    """Verify /api/v1/auth/me returns current user profile."""
    user = sample_users["analyst"]
    token = create_access_token(user.user_id, user.role)

    resp = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "analyst@example.com"
    assert body["role"] == "analyst"
    assert body["user_id"] == str(user.user_id)


def test_api_change_password(
    test_client: TestClient,
    sample_users: dict[str, AppUser],
    auth_db_session: Session,
) -> None:
    """Verify /api/v1/auth/change-password validates old password and applies new."""
    user = sample_users["viewer"]
    token = create_access_token(user.user_id, user.role)

    # 1. Invalid old password fails
    fail_resp = test_client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "WrongPassword123!", "new_password": "NewValidPassword123!"},
    )
    assert fail_resp.status_code == 401

    # 2. Valid change succeeds (204)
    succ_resp = test_client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "ValidPassword123!", "new_password": "NewValidPassword123!"},
    )
    assert succ_resp.status_code == 204

    # 3. New password authenticates
    auth_db_session.expire_all()
    auth_user = authenticate_user(auth_db_session, "viewer@example.com", "NewValidPassword123!")
    assert auth_user.user_id == user.user_id


# =========================================================================
# 5. Role-Based Access Control (require_role) Hierarchy Tests
# =========================================================================


def test_rbac_hierarchy(
    test_client: TestClient,
    sample_users: dict[str, AppUser],
) -> None:
    """Verify viewer < analyst < admin hierarchy on protected routes."""
    viewer_token = create_access_token(sample_users["viewer"].user_id, "viewer")
    analyst_token = create_access_token(sample_users["analyst"].user_id, "analyst")
    admin_token = create_access_token(sample_users["admin"].user_id, "admin")

    # Viewer permissions
    assert (
        test_client.get(
            "/test-rbac/viewer", headers={"Authorization": f"Bearer {viewer_token}"}
        ).status_code
        == 200
    )
    assert (
        test_client.get(
            "/test-rbac/analyst", headers={"Authorization": f"Bearer {viewer_token}"}
        ).status_code
        == 403
    )
    assert (
        test_client.get(
            "/test-rbac/admin", headers={"Authorization": f"Bearer {viewer_token}"}
        ).status_code
        == 403
    )

    # Analyst permissions
    assert (
        test_client.get(
            "/test-rbac/viewer", headers={"Authorization": f"Bearer {analyst_token}"}
        ).status_code
        == 200
    )
    assert (
        test_client.get(
            "/test-rbac/analyst", headers={"Authorization": f"Bearer {analyst_token}"}
        ).status_code
        == 200
    )
    assert (
        test_client.get(
            "/test-rbac/admin", headers={"Authorization": f"Bearer {analyst_token}"}
        ).status_code
        == 403
    )

    # Admin permissions
    assert (
        test_client.get(
            "/test-rbac/viewer", headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code
        == 200
    )
    assert (
        test_client.get(
            "/test-rbac/analyst", headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code
        == 200
    )
    assert (
        test_client.get(
            "/test-rbac/admin", headers={"Authorization": f"Bearer {admin_token}"}
        ).status_code
        == 200
    )


def test_rbac_dynamic_db_check(
    test_client: TestClient,
    sample_users: dict[str, AppUser],
    auth_db_session: Session,
) -> None:
    """Verify deactivation and role demotion apply immediately (SEC-AUTH-05)."""
    analyst = sample_users["analyst"]
    analyst_token = create_access_token(analyst.user_id, "analyst")

    # Initially has analyst access
    resp1 = test_client.get(
        "/test-rbac/analyst", headers={"Authorization": f"Bearer {analyst_token}"}
    )
    assert resp1.status_code == 200

    # Demote user in database to viewer
    analyst.role = "viewer"
    auth_db_session.commit()

    # Immediate rejection on analyst endpoint with same token
    resp2 = test_client.get(
        "/test-rbac/analyst", headers={"Authorization": f"Bearer {analyst_token}"}
    )
    assert resp2.status_code == 403

    # Deactivate user
    analyst.is_active = False
    auth_db_session.commit()

    # Immediate rejection across all endpoints with 401
    resp3 = test_client.get(
        "/test-rbac/viewer", headers={"Authorization": f"Bearer {analyst_token}"}
    )
    assert resp3.status_code == 401


# =========================================================================
# 6. User Management API & Last-Admin Protection (/api/v1/users)
# =========================================================================


def test_api_users_crud_and_last_admin_protection(
    test_client: TestClient,
    sample_users: dict[str, AppUser],
    auth_db_session: Session,
) -> None:
    """Verify user management endpoints and last active administrator protection (SEC-AUTHZ-03)."""
    admin_token = create_access_token(sample_users["admin"].user_id, "admin")
    viewer_token = create_access_token(sample_users["viewer"].user_id, "viewer")

    # 1. Non-admin forbidden from listing users
    assert (
        test_client.get(
            "/api/v1/users", headers={"Authorization": f"Bearer {viewer_token}"}
        ).status_code
        == 403
    )

    # 2. Admin can list users
    list_resp = test_client.get("/api/v1/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 3

    # 3. Create user
    create_resp = test_client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "email": "new.user@example.com",
            "full_name": "New User",
            "role": "analyst",
            "password": "ValidPassword123!",
        },
    )
    assert create_resp.status_code == 201
    new_user_data = create_resp.json()
    assert new_user_data["email"] == "new.user@example.com"
    new_user_id = new_user_data["user_id"]

    # 4. Duplicate email rejected with 409
    dup_resp = test_client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "email": "new.user@example.com",
            "role": "viewer",
            "password": "ValidPassword123!",
        },
    )
    assert dup_resp.status_code == 409

    # 5. Last active admin cannot be deactivated or demoted (SEC-AUTHZ-03)
    admin_user = sample_users["admin"]
    patch_fail = test_client.patch(
        f"/api/v1/users/{admin_user.user_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"is_active": False},
    )
    assert patch_fail.status_code == 409
    assert "last active administrator" in patch_fail.json()["detail"]

    patch_demote_fail = test_client.patch(
        f"/api/v1/users/{admin_user.user_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "analyst"},
    )
    assert patch_demote_fail.status_code == 409

    # 6. Deactivating non-admin user succeeds
    patch_ok = test_client.patch(
        f"/api/v1/users/{new_user_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"is_active": False},
    )
    assert patch_ok.status_code == 200
    assert patch_ok.json()["is_active"] is False
