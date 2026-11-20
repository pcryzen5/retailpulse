"""Uvicorn ASGI entrypoint for RetailPulse API service.

Usage:
    uv run uvicorn retailpulse.api.main:app --reload --port 8000
"""

from __future__ import annotations

from retailpulse.api.app import create_app

app = create_app()
