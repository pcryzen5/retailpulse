# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for RetailPulse API service (Lightweight)

FROM ghcr.io/astral-sh/uv:0.5.21 AS uv_bin
FROM python:3.15.1-slim-bookworm AS builder

COPY --from=uv_bin /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml uv.lock README.md /app/

# Install core and api dependencies into virtual environment
RUN uv sync --frozen --no-dev --no-install-project --group core --group api

# Copy application source and install package
COPY src/ /app/src/
RUN uv sync --frozen --no-dev --group core --group api

# Final runtime image
FROM python:3.15.1-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RP_ENV=prod

WORKDIR /app

# Create unprivileged application user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/false -m appuser

# Copy virtual environment and application code from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --chown=appuser:appuser pyproject.toml README.md /app/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

CMD ["uvicorn", "retailpulse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
