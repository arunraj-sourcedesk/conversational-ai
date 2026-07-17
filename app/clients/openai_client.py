"""
OpenAI async client abstraction layer.

Wraps the official `openai` Python SDK and provides:
- Async chat completion (non-streaming)
- Retry logic with exponential back-off
- Structured error mapping to domain exceptions
"""

import asyncio
from typing import Any

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from app.core.config import Settings
from app.core.exceptions import LLMError, TimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Thin async wrapper around the OpenAI SDK with retry logic."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout,
            max_retries=settings.openai_max_retries,
        )

    # ------------------------------------------------------------------
    # Chat Completion — non-streaming
    # ------------------------------------------------------------------

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> tuple[str, int]:
        """
        Perform a full (non-streaming) chat completion.

        Returns:
            (reply_text, total_tokens_used)
        """
        model = model or self._settings.openai_chat_model
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            logger.debug("chat_complete: model=%s tokens=%d", model, tokens)
            return content, tokens

        except APITimeoutError as exc:
            raise TimeoutError("OpenAI request timed out") from exc
        except RateLimitError as exc:
            raise LLMError("OpenAI rate limit exceeded — try again shortly") from exc
        except APIError as exc:
            raise LLMError(f"OpenAI API error: {exc.message}") from exc
        except Exception as exc:
            raise LLMError(f"Unexpected LLM error: {exc}") from exc

