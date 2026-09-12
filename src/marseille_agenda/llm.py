"""Model factory: every LLM call goes through OpenRouter."""

from __future__ import annotations

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from .config import Settings

DETERMINISTIC = ModelSettings(temperature=0.0, max_tokens=8000)


def make_model(settings: Settings, name: str) -> OpenRouterModel:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        app_url="https://github.com/tfrere/marseille-agenda",
        app_title="marseille-agenda",
    )
    return OpenRouterModel(name, provider=provider, settings=DETERMINISTIC)
