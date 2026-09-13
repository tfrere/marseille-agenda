"""Pydantic models shared by the extractor, the verifier and the pipeline."""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

EventStatus = Literal["scheduled", "cancelled", "postponed", "sold_out"]
Verdict = Literal["accept", "reject", "uncertain"]


class ExtractedEvent(BaseModel):
    """One event as read from a single source document.

    Every factual field must be backed by `evidence`: verbatim excerpts of the source.
    The pipeline rejects events whose evidence, title or URL cannot be found in the
    source text, so the model must copy, never paraphrase, those fields.
    """

    title: str = Field(description="Event title, copied verbatim from the source (same casing).")
    start_date: date = Field(description="First day of the event (ISO date). Infer the year from context if missing.")
    start_time: time | None = Field(default=None, description="Start time if written on the page, else null.")
    end_date: date | None = Field(default=None, description="Last day for multi-day events (exhibitions, festivals), else null.")
    end_time: time | None = Field(default=None, description="End time if written, else null.")
    location_name: str | None = Field(
        default=None,
        description="Room or venue name exactly as written on the page (e.g. 'Le Dodu', 'Grand plateau'). Null if not written.",
    )
    url: str | None = Field(
        default=None,
        description="Detail or ticketing URL for this specific event, copied verbatim from a link in the source. Null if none.",
    )
    price: str | None = Field(default=None, description="Price as written (e.g. '10€ / 8€ prévente', 'Gratuit'). Null if not written.")
    summary: str | None = Field(default=None, description="One short sentence (<= 200 chars) describing the event. Paraphrase allowed.")
    status: EventStatus = Field(default="scheduled", description="'cancelled' / 'postponed' / 'sold_out' if the page says so.")
    evidence: list[str] = Field(
        min_length=1,
        max_length=3,
        description="1 to 3 verbatim excerpts (20-300 chars each) of the source text proving title, date and time.",
    )

    @field_validator("start_time", "end_time")
    @classmethod
    def _local_naive_time(cls, v: time | None) -> time | None:
        # Some models answer "19:00Z"; times are local (Europe/Paris) wall-clock values.
        return v.replace(tzinfo=None) if v is not None and v.tzinfo is not None else v


class ExtractionResult(BaseModel):
    events: list[ExtractedEvent] = Field(default_factory=list)
    notes: str | None = Field(default=None, description="Anything odd about the page (empty agenda, past events only, JS-only content).")


class VerifierVerdict(BaseModel):
    """Output of the adversarial verifier for one event."""

    verdict: Verdict
    reason: str = Field(description="One or two sentences. If rejecting, quote the exact contradiction.")
    date_ok: bool
    time_ok: bool
    is_real_public_event: bool = Field(description="False for opening hours, generic offers, past events, recurring schedule templates.")
    not_cancelled: bool
    url_ok: bool


SourceKind = Literal["html", "json", "instagram", "facebook"]


class Venue(BaseModel):
    """One entry of venues.json. Only `name` is required; everything else is optional."""

    name: str
    id: str = Field(default="", description="Stable slug; derived from the name when omitted.")
    city: str = "Marseille"
    category: str = "autre"
    map_name: str | None = Field(default=None, description="Name as used in the marseille-tes-lieux map (PLACES.n).")
    website: str | None = Field(default=None, description="Optional hint for the discovery agent.")
    instagram: str | None = Field(default=None, description="Instagram handle or profile URL. Found by discovery when omitted.")
    facebook: str | None = Field(default=None, description="Facebook page slug or URL. Found by discovery when omitted.")
    web: bool = Field(default=True, description="Set to false when the venue has no web agenda at all: social sources only.")
    social: bool = Field(default=True, description="Set to false to ignore Instagram/Facebook for this venue, even when discovery "
                                                   "finds handles (e.g. an account shared by several venues).")
    lat: float | None = None
    lng: float | None = None
    enabled: bool = True

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = slugify(self.name)
        if not self.map_name:
            self.map_name = self.name


def slugify(s: str) -> str:
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "venue"


class VenuesFile(BaseModel):
    venues: list[Venue]


class Event(BaseModel):
    """Published event (data/events.json)."""

    uid: str
    venue_id: str
    venue_name: str
    category: str
    title: str
    start_date: date
    start_time: time | None = None
    end_date: date | None = None
    end_time: time | None = None
    location_name: str | None = None
    url: str | None = None
    price: str | None = None
    summary: str | None = None
    status: EventStatus = "scheduled"
    event_type: str | None = None
    free: bool | None = None
    # Multi-day events (exhibitions, recurring visits "until Sept 30") are published once
    # with start/end dates; the calendar shows them as a range, not one entry per day.
    recurring: bool = False
    source_url: str
    source_kind: SourceKind
    evidence: list[str]
    verifier: Verdict | None = None
    verifier_reason: str | None = None
    first_seen: date
    last_seen: date
    lat: float | None = None
    lng: float | None = None


class SourceState(BaseModel):
    url: str
    content_hash: str | None = None
    last_fetched: date | None = None
    last_ok: date | None = None
    last_error: str | None = None
    event_count: int = 0
    # uids extracted the last time this source was successfully processed
    uids: list[str] = Field(default_factory=list)
    missing_runs: dict[str, int] = Field(default_factory=dict)


class State(BaseModel):
    events: dict[str, Event] = Field(default_factory=dict)
    sources: dict[str, SourceState] = Field(default_factory=dict)
    last_run: date | None = None


class Alert(BaseModel):
    level: Literal["warning", "error"]
    venue_id: str
    source_url: str
    message: str


class RunReport(BaseModel):
    run_date: date
    sources_total: int
    sources_ok: int
    sources_skipped_unchanged: int
    events_published: int
    events_new: int
    events_rejected_grounding: int
    events_rejected_verifier: int
    events_uncertain: int
    alerts: list[Alert] = Field(default_factory=list)
    llm_calls: int = 0
    posts_analyzed: int = Field(default=0, description="Social posts read by the vision model this run.")
    apify_runs: int = Field(default=0, description="Apify actor runs this run.")
    credits_remaining_usd: float | None = Field(default=None, description="OpenRouter balance after the run.")
    run_cost_usd: float | None = Field(default=None, description="Balance delta during the run.")
