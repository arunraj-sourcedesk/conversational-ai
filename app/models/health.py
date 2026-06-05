"""Models for health-check endpoints."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    version: str
    service: str
