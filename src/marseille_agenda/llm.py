"""Model factory: every LLM call goes through OpenRouter."""

from __future__ import annotations

import logging

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from .config import Settings

log = logging.getLogger(__name__)

DETERMINISTIC = ModelSettings(temperature=0.0, max_tokens=8000)


def fetch_credits(settings: Settings) -> float | None:
    """Remaining OpenRouter balance in USD, or None when unavailable."""
    if not settings.openrouter_api_key:
        return None
    try:
        import httpx

        r = httpx.get(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()["data"]
        return round(float(d["total_credits"]) - float(d["total_usage"]), 4)
    except Exception as exc:  # noqa: BLE001 - purely informational
        log.warning("could not fetch OpenRouter credits: %s", exc)
        return None


def make_model(settings: Settings, name: str) -> OpenRouterModel:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        app_url="https://github.com/tfrere/marseille-agenda",
        app_title="marseille-agenda",
    )
    return OpenRouterModel(name, provider=provider, settings=DETERMINISTIC)
