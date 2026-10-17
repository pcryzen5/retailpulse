.DEFAULT_GOAL := help

.PHONY: help setup up down logs migrate seed etl train score api dashboard lint fmt typecheck test test-unit test-int e2e cov security clean

help: ## Show this help message
	@python -c "import re; print('Usage: make <target>\n\nTargets:'); [print(f'  {m[1]:<16} {m[2]}') for line in open('Makefile') for m in [re.match(r'^([a-zA-Z0-9_-]+):.*?## (.*)$$', line)] if m]"

setup: ## Install dependencies and pre-commit hooks
	uv sync
	uv run pre-commit install

up: ## Start platform services with docker compose
	docker compose up -d

down: ## Stop platform services with docker compose
	docker compose down

logs: ## View logs from platform services
	docker compose logs -f

migrate: ## Run database migrations to head
	uv run alembic upgrade head

seed: ## Seed initial admin user and synthetic parameters
	uv run retailpulse seed

etl: ## Run data ingestion and transformation pipeline
	uv run retailpulse etl run

train: ## Train machine learning models
	uv run retailpulse train

score: ## Run batch scoring and generate recommendations
	uv run retailpulse score

api: ## Run the FastAPI development server with reload
	uv run uvicorn retailpulse.api.main:app --reload --port 8000

dashboard: ## Run the Streamlit dashboard
	uv run streamlit run src/retailpulse/dashboard/app.py

lint: ## Run linters (ruff, ruff format check, import-linter)
	uv run ruff check .
	uv run ruff format --check .
	uv run lint-imports

fmt: ## Auto-format code and fix lint issues
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Run static type checking with mypy
	uv run mypy src/

test: ## Run unit and integration tests
	uv run pytest tests/

test-unit: ## Run unit tests only
	uv run pytest -m "unit and not slow" tests/

test-int: ## Run integration tests only
	uv run pytest -m "integration or ml" tests/

e2e: ## Run end-to-end tests
	uv run pytest tests/e2e/

cov: ## Run tests with branch coverage report
	uv run pytest --cov=src/retailpulse --cov-report=term-missing tests/

security: ## Run security static analysis and vulnerability scans
	uv run bandit -r src/
	uv run pip-audit

clean: ## Clean cache directories and build artifacts
	@python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('.mypy_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('build'), pathlib.Path('dist'), pathlib.Path('.coverage')]]"
