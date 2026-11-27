# syntax=docker/dockerfile:1
# Dockerfile for MLflow tracking and registry server

FROM python:3.15.1-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install MLflow and PostgreSQL adapter
RUN pip install --no-cache-dir \
    mlflow==2.12.0 \
    psycopg[binary]>=3.1.18

# Create unprivileged service user
RUN groupadd -g 10001 mlflow && \
    useradd -u 10001 -g mlflow -s /bin/false -m mlflow

WORKDIR /mlflow

RUN chown -R mlflow:mlflow /mlflow

USER mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
