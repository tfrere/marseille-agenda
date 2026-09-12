"""Runtime configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TZ = ZoneInfo("Europe/Paris")

# Reject events further in the future than this (guards against bad year inference).
MAX_HORIZON_DAYS = 18 * 30
# Keep an event that disappeared from its source for this many runs before dropping it.
MISSING_RUNS_BEFORE_DROP = 3
# If a source that had at least this many events suddenly yields none (or drops by more
# than VOLUME_DROP_RATIO), keep the last verified state and raise an alert.
VOLUME_GUARD_MIN_EVENTS = 5
VOLUME_DROP_RATIO = 0.5
# Hard cap on source text sent to the extractor (chars).
MAX_SOURCE_CHARS = 160_000


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    extractor_model: str
    """Grounded extraction (independent reader) and schema generation."""
    verifier_model: str
    """Adversarial verifier: a different model family on purpose."""
    discover_model: str
    """Tool-using discovery agent."""
    search_model: str
    """Cheap model used only as a carrier for OpenRouter's web-search plugin."""
    data_dir: Path
    venues_file: Path

    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key)


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        extractor_model=os.environ.get("EXTRACTOR_MODEL", "anthropic/claude-sonnet-5"),
        verifier_model=os.environ.get("VERIFIER_MODEL", "google/gemini-3.8-flash"),
        discover_model=os.environ.get("DISCOVER_MODEL", "anthropic/claude-sonnet-5"),
        search_model=os.environ.get("SEARCH_MODEL", "openai/gpt-5.4-mini"),
        data_dir=Path(os.environ.get("DATA_DIR", ROOT / "data")),
        venues_file=Path(os.environ.get("VENUES_FILE", ROOT / "venues.json")),
    )
