from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.voice import router as voice_router

__all__ = ["chat_router", "health_router", "voice_router"]
