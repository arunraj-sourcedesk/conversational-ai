"""
Application factory and entrypoint.

Architecture overview:
  - Lifespan context manager handles startup / shutdown hooks
  - Middleware is applied in order: CORS → exception handler → logging
  - All routes are mounted under versioned prefixes via APIRouter
  - Dependency injection is handled via FastAPI's Depends() mechanism

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.routes import chat_router, health_router, voice_router
from app.utils.middleware import ExceptionHandlerMiddleware, RequestLoggingMiddleware

# Configure logging before anything else
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup / shutdown lifecycle.

    Add database pool acquisition, cache warmup, etc. here.
    """
    logger.info(
        "Starting %s v%s [debug=%s]",
        settings.app_name, settings.app_version, settings.debug,
    )
    # Initialize database
    try:
        from app.utils.db import init_db
        await init_db()
    except Exception as exc:
        logger.exception("Failed to initialize database during startup: %s", exc)
    yield
    logger.info("Shutting down %s", settings.app_name)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade conversational AI service with text and voice APIs. "
            "Supports streaming (SSE for text, chunked for audio) and "
            "non-streaming modes."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware (applied in reverse order — last added = outermost)
    # ------------------------------------------------------------------

    # 1. CORS (outermost — must run before any custom logic)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Exception handler (catches domain errors → JSON)
    application.add_middleware(ExceptionHandlerMiddleware)

    # 3. Request logging + trace ID injection
    application.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(voice_router)

    return application


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn)
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower(),
        access_log=False,  # We use our own middleware logging
    )
