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
    vision_model: str
    """Cheap vision model reading social posts (caption + flyer images)."""
    vision_verifier_model: str
    """Adversarial verifier for social posts: vision-capable, different family from vision_model."""
    apify_token: str | None
    """Apify API token; enables Instagram / Facebook sources."""
    data_dir: Path
    venues_file: Path
    site_dir: Path = ROOT / "site"
    """Static site root; event visuals are cached under `site/img/`."""

    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_social(self) -> bool:
        return bool(self.apify_token)


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
        extractor_model=os.environ.get("EXTRACTOR_MODEL", "anthropic/claude-sonnet-5"),
        verifier_model=os.environ.get("VERIFIER_MODEL", "google/gemini-3.8-flash"),
        discover_model=os.environ.get("DISCOVER_MODEL", "anthropic/claude-sonnet-5"),
        search_model=os.environ.get("SEARCH_MODEL", "openai/gpt-5.4-mini"),
        # Benchmarked on tests/test_live.py: Qwen3-VL reads flyers deterministically and returns the
        # nested structure; DeepSeek V4.1 Flash drops nested lists as an extractor but is a perfect
        # verifier (flat output). Both are ~20x cheaper than Sonnet.
        vision_model=os.environ.get("VISION_MODEL", "qwen/qwen3-vl-32b-instruct"),
        vision_verifier_model=os.environ.get("VISION_VERIFIER_MODEL", "deepseek/deepseek-v4.1-flash"),
        apify_token=os.environ.get("APIFY_API_KEY") or os.environ.get("APIFY_TOKEN") or None,
        data_dir=Path(os.environ.get("DATA_DIR", ROOT / "data")),
        venues_file=Path(os.environ.get("VENUES_FILE", ROOT / "venues.json")),
        site_dir=Path(os.environ.get("SITE_DIR", ROOT / "site")),
    )
