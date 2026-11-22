"""FastAPI Application Factory for RetailPulse REST Service.

Implements P5-01, TRD §7, API §1-§2, and SECURITY §7:
- Settings resolution and application state binding.
- Async lifespan handler for structured startup/shutdown lifecycle.
- Production-hardened middleware stack: RequestID, Timing, SecurityHeaders, optional CORS.
- RFC 9457 error handlers.
- Health probes (/health/live, /health/ready, /health/startup).
- OpenAPI documentation toggling based on RP_ENABLE_DOCS / environment.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retailpulse.api.errors import register_error_handlers
from retailpulse.api.middleware import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    TimingMiddleware,
)
from retailpulse.api.routers.auth import router as auth_router
from retailpulse.api.routers.customers import router as customers_router
from retailpulse.api.routers.forecasts import router as forecasts_router
from retailpulse.api.routers.health import router as health_router
from retailpulse.api.routers.inventory import router as inventory_router
from retailpulse.api.routers.kpis import router as kpis_router
from retailpulse.api.routers.reference import router as reference_router
from retailpulse.api.routers.users import router as users_router
from retailpulse.config import Settings, get_settings
from retailpulse.logging import get_logger

logger = get_logger("retailpulse.api.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()

    logger.info(
        "Application startup",
        env=settings.env,
        docs_enabled=settings.docs_enabled,
        currency=settings.currency,
    )

    yield

    logger.info("Application shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and configure the RetailPulse FastAPI application instance.

    Args:
        settings: Optional custom Settings instance; defaults to environment-loaded settings.

    Returns:
        Configured FastAPI application with middleware, routes, and error handlers.
    """
    cfg = settings or get_settings()

    docs_url = "/docs" if cfg.docs_enabled else None
    redoc_url = "/redoc" if cfg.docs_enabled else None
    openapi_url = "/openapi.json" if cfg.docs_enabled else None

    app = FastAPI(
        title="RetailPulse API",
        description=(
            "High-Throughput Demand Forecasting, Customer Intelligence, "
            "and Inventory Optimization REST Service."
        ),
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    app.state.settings = cfg

    # 1. Register RFC 9457 error handlers
    register_error_handlers(app)

    # 2. Add middleware stack (LIFO: RequestID is outermost, then Timing, then SecurityHeaders)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # 3. Add CORS middleware if origins configured (SEC-WEB-02: exact allowlist, never * with creds)
    if cfg.cors_origins:
        if isinstance(cfg.cors_origins, str):
            origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
        else:
            origins = list(cfg.cors_origins)

        if origins:
            allow_creds = "*" not in origins
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=allow_creds,
                allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
                allow_headers=["*"],
            )

    # 4. Mount routers
    app.include_router(health_router)
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(users_router, prefix="/api/v1")
    app.include_router(reference_router, prefix="/api/v1")
    app.include_router(kpis_router, prefix="/api/v1")
    app.include_router(forecasts_router, prefix="/api/v1")
    app.include_router(customers_router, prefix="/api/v1")
    app.include_router(inventory_router, prefix="/api/v1")

    return app
