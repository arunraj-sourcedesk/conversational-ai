"""
Domain-specific exceptions for clean error propagation across layers.
"""


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LLMError(AppError):
    """Raised when the LLM provider returns an error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class STTError(AppError):
    """Raised when speech-to-text transcription fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class TTSError(AppError):
    """Raised when text-to-speech synthesis fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class ValidationError(AppError):
    """Raised on invalid input."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class TimeoutError(AppError):
    """Raised when an upstream request times out."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message, status_code=504)


class AudioProcessingError(AppError):
    """Raised on audio file issues."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)
