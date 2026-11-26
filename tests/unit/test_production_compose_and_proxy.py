"""Verification test for P8-03 Production compose and proxy configuration.

Verifies:
1. docker-compose.prod.yml publishes ONLY ports 80 and 443 (bound exclusively by Caddy).
2. All backend services (postgres, mlflow, prefect, worker, api, dashboard) have no published ports.
3. Caddy reverse proxy blocks /metrics*, /docs*, /redoc*, and /openapi.json*.
4. Caddy sets security headers:
   X-Content-Type-Options nosniff, X-Frame-Options DENY, Referrer-Policy no-referrer.
5. All services specify restart policies and resource limits.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def test_production_compose_only_publishes_ports_80_and_443() -> None:
    """Verify that port scan / exposed ports on docker-compose.prod.yml only include 80/443."""
    prod_compose_path = Path("docker-compose.prod.yml")
    assert prod_compose_path.exists(), "docker-compose.prod.yml must exist"

    with prod_compose_path.open("r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    published_ports: dict[str, list[str]] = {}

    for service_name, service_cfg in services.items():
        if "ports" in service_cfg:
            published_ports[service_name] = [str(p) for p in service_cfg["ports"]]

    # Only caddy should have published ports
    exposed = list(published_ports.keys())
    assert set(published_ports.keys()) == {"caddy"}, (
        f"Only caddy should publish ports in production. Found services: {exposed}"
    )

    # Caddy ports must only be 80 and 443
    caddy_ports = published_ports["caddy"]
    expected_ports = {"80:80", "443:443"}
    assert set(caddy_ports) == expected_ports, f"Unexpected ports published by Caddy: {caddy_ports}"


def test_production_compose_resource_limits_and_restart_policy() -> None:
    """Verify all services define restart policies and deploy resource limits."""
    prod_compose_path = Path("docker-compose.prod.yml")
    with prod_compose_path.open("r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    for s_name, s_cfg in services.items():
        assert s_cfg.get("restart") in ("always", "unless-stopped"), (
            f"Service {s_name} missing safe restart policy"
        )
        deploy = s_cfg.get("deploy", {})
        resources = deploy.get("resources", {})
        limits = resources.get("limits", {})
        assert "cpus" in limits, f"Service {s_name} missing cpus resource limits"
        assert "memory" in limits, f"Service {s_name} missing memory resource limits"


def test_caddyfile_blocks_metrics_and_docs_and_adds_security_headers() -> None:
    """Verify Caddyfile enforces security headers, blocks /metrics and docs, and routes cleanly."""
    caddyfile_path = Path("infra/caddy/Caddyfile")
    assert caddyfile_path.exists(), "infra/caddy/Caddyfile must exist"

    content = caddyfile_path.read_text(encoding="utf-8")

    # 1. Blocked routes
    assert "respond /metrics* 403" in content
    assert "respond /docs* 403" in content
    assert "respond /redoc* 403" in content
    assert "respond /openapi.json* 403" in content

    # 2. Security headers (SEC-WEB-01)
    assert 'X-Content-Type-Options "nosniff"' in content
    assert 'X-Frame-Options "DENY"' in content
    assert 'Referrer-Policy "no-referrer"' in content

    # 3. Drift sandbox CSP (SEC-WEB-06)
    assert "Content-Security-Policy" in content
    assert "sandbox" in content

    # 4. Upload size limit (SEC-INP-03)
    assert "max_size 52428800" in content
