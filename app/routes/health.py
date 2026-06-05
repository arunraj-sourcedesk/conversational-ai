"""Health check endpoint."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Returns service name and version. Use for liveness probes."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        service=settings.app_name,
    )
