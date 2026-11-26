"""Post-deployment smoke test suite.

Verifies P8-04 exit criteria:
1. /health/ready returns 200 OK with database connected and migrations current.
2. Login authenticates an active user and returns an access token.
3. Authenticated read succeeds (e.g. GET /api/v1/kpis/overview or /api/v1/auth/me).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import requests


def _check_readiness(ready_url: str, max_retries: int, retry_delay: float) -> bool:
    print(f"--> Step 1: Checking readiness probe ({ready_url})...")
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(ready_url, timeout=5)
            if resp.status_code == 200:
                body = resp.json()
                print(f"    [OK] /health/ready returned 200 OK: {body}")
                return True
            print(f"    Attempt {attempt}/{max_retries}: status {resp.status_code}, retrying...")
        except requests.RequestException as exc:
            print(f"    Attempt {attempt}/{max_retries}: connection error ({exc}), retrying...")

        time.sleep(retry_delay)

    print("    [FAIL] Readiness probe did not become ready in time.")
    return False


def _check_login(login_url: str, email: str, password: str) -> str | None:
    print(f"\n--> Step 2: Testing authentication and token issuance ({login_url})...")
    try:
        login_resp = requests.post(
            login_url,
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if login_resp.status_code != 200:
            print(f"    [FAIL] Login returned status {login_resp.status_code}: {login_resp.text}")
            return None

        payload: dict[str, Any] = login_resp.json()
        token = payload.get("access_token")
        if not token:
            print("    [FAIL] Login succeeded but response did not contain access_token.")
            return None

        print("    [OK] Authenticated successfully. Received Bearer token.")
        return str(token)
    except requests.RequestException as exc:
        print(f"    [FAIL] Login request failed with exception: {exc}")
        return None


def _check_authenticated_reads(read_url: str, overview_url: str, token: str) -> bool:
    print(f"\n--> Step 3: Testing authenticated read endpoint ({read_url})...")
    auth_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        me_resp = requests.get(read_url, headers=auth_headers, timeout=5)
        if me_resp.status_code != 200:
            print(f"    [FAIL] /api/v1/auth/me returned status {me_resp.status_code}: {me_resp.text}")
            return False

        user_data = me_resp.json()
        print(
            f"    [OK] Read user profile: email={user_data.get('email')}, role={user_data.get('role')}"
        )

        overview_resp = requests.get(overview_url, headers=auth_headers, timeout=5)
        if overview_resp.status_code == 200:
            print(f"    [OK] Read KPI overview: {overview_resp.json()}")
        else:
            print(f"    [WARN] KPI overview returned {overview_resp.status_code} (non-fatal)")
        return True
    except requests.RequestException as exc:
        print(f"    [FAIL] Authenticated read failed with exception: {exc}")
        return False


def run_smoke_test(
    base_url: str,
    email: str,
    password: str,
    max_retries: int = 15,
    retry_delay_seconds: float = 2.0,
) -> bool:
    """Run post-deployment smoke test with retries for container readiness.

    Args:
        base_url: Base HTTP(S) URL of the deployment (e.g. http://localhost or https://example.com).
        email: Smoke test user email.
        password: Smoke test user password.
        max_retries: Maximum number of polling retries.
        retry_delay_seconds: Delay between retries in seconds.

    Returns:
        True if all smoke checks pass, False otherwise.
    """
    clean_base = base_url.rstrip("/")
    ready_url = f"{clean_base}/health/ready"
    login_url = f"{clean_base}/api/v1/auth/login"
    read_url = f"{clean_base}/api/v1/auth/me"
    overview_url = f"{clean_base}/api/v1/kpis/overview"

    print("=== RetailPulse Post-Deployment Smoke Test ===")
    print(f"Target Base URL: {clean_base}")
    print(f"Target User:     {email}\n")

    if not _check_readiness(ready_url, max_retries, retry_delay_seconds):
        return False

    token = _check_login(login_url, email, password)
    if not token:
        return False

    if not _check_authenticated_reads(read_url, overview_url, token):
        return False

    print("\n=== ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY ===")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPulse Deployment Smoke Test")
    parser.add_argument(
        "--url",
        default="http://localhost",
        help="Base deployment URL (default: http://localhost)",
    )
    parser.add_argument("--email", default="admin@retailpulse.local", help="User email")
    parser.add_argument(
        "--password", default="AdminSecretPassword123!", help="User password"
    )
    parser.add_argument("--retries", type=int, default=15, help="Readiness retry attempts")
    parser.add_argument("--delay", type=float, default=2.0, help="Retry delay in seconds")

    args = parser.parse_args()
    success = run_smoke_test(
        base_url=args.url,
        email=args.email,
        password=args.password,
        max_retries=args.retries,
        retry_delay_seconds=args.delay,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
