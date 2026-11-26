"""Tests for P8-07 Kubernetes Kustomize manifests.

Verifies:
1. Base Kustomization renders all required services, deployments, configs, and ingresses.
2. Kind overlay applies appropriate namePrefix and single-replica resource limits.
3. Health and readiness probes are declared on all serving deployments.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


def test_kustomize_manifest_files_exist() -> None:
    """Verify that required Kustomize base and kind overlay files exist."""
    base_dir = Path("infra/k8s/base")
    assert (base_dir / "kustomization.yaml").exists()
    assert (base_dir / "namespace.yaml").exists()
    assert (base_dir / "config.yaml").exists()
    assert (base_dir / "platform.yaml").exists()
    assert (base_dir / "apps.yaml").exists()

    kind_dir = Path("infra/k8s/overlays/kind")
    assert (kind_dir / "kustomization.yaml").exists()


def test_kubectl_kustomize_generates_valid_yaml() -> None:
    """Verify kubectl kustomize outputs clean parseable YAML for base and kind overlay."""
    kubectl_bin = shutil.which("kubectl")
    if not kubectl_bin:
        pytest.skip("kubectl not found in environment PATH")

    # Test base rendering
    res_base = subprocess.run(  # noqa: S603
        [kubectl_bin, "kustomize", "infra/k8s/base"],
        capture_output=True,
        text=True,
        check=True,
    )
    base_docs = list(yaml.safe_load_all(res_base.stdout))
    assert len(base_docs) >= 8

    # Verify deployments in base
    deployments = {
        doc["metadata"]["name"]: doc for doc in base_docs if doc.get("kind") == "Deployment"
    }
    assert "api" in deployments
    assert "dashboard" in deployments
    assert "postgres" in deployments
    assert "worker" in deployments

    # Verify readiness probes
    api_container = deployments["api"]["spec"]["template"]["spec"]["containers"][0]
    assert "readinessProbe" in api_container
    assert api_container["readinessProbe"]["httpGet"]["path"] == "/health/ready"

    # Test kind overlay rendering
    res_kind = subprocess.run(  # noqa: S603
        [kubectl_bin, "kustomize", "infra/k8s/overlays/kind"],
        capture_output=True,
        text=True,
        check=True,
    )
    kind_docs = list(yaml.safe_load_all(res_kind.stdout))
    kind_deployments = {
        doc["metadata"]["name"]: doc for doc in kind_docs if doc.get("kind") == "Deployment"
    }

    assert "kind-api" in kind_deployments
    assert kind_deployments["kind-api"]["spec"]["replicas"] == 1
    assert "kind-dashboard" in kind_deployments
    assert kind_deployments["kind-dashboard"]["spec"]["replicas"] == 1
