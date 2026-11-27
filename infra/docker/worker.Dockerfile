# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for RetailPulse Worker service (Full ML Workload)

FROM ghcr.io/astral-sh/uv:0.5.21 AS uv_bin
FROM python:3.15.1-slim-bookworm AS builder

COPY --from=uv_bin /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies required by heavy ML packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency definition files
COPY pyproject.toml uv.lock README.md /app/

# Install core, ml, and api dependencies into virtual environment
RUN uv sync --frozen --no-dev --no-install-project --group core --group ml --group api

# Copy application source and install package
COPY src/ /app/src/
RUN uv sync --frozen --no-dev --group core --group ml --group api

# Final runtime image
FROM python:3.15.1-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RP_ENV=prod

WORKDIR /app

# Install runtime system libraries (e.g. libgomp for LightGBM/Torch)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/false -m appuser

# Copy virtual environment and application code from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --chown=appuser:appuser pyproject.toml README.md /app/

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import retailpulse; print(retailpulse.__version__)"

CMD ["retailpulse", "worker"]
