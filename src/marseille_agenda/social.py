"""Instagram / Facebook sources: fetch through Apify, read new posts once with a vision
model, cache the verdicts per post, and rebuild the venue's social events every run.

State lives in data/social.json. A post is analysed exactly once (LLM cost is per new post,
not per run); the events it announced stay published until they are past, then the post is
pruned. Facebook's structured "upcoming hosted events" tab needs no model at all.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from . import apify
from .apify import ApifyError, FacebookEvent, SocialPost
from .apply import SchemaEvent
from .config import TZ, Settings
from .extraction_schema import SourceRecord
from .schema import Venue, Verdict
from .social_extract import PostDeps, PostEvent, download_images, extract_post, verify_post_event

log = logging.getLogger("marseille_agenda.social")

SocialKind = Literal["instagram", "facebook", "facebook-events"]

POST_RETENTION_DAYS = 180
"""Posts older than this cannot announce anything still upcoming; they are pruned."""
POSTS_PER_FETCH = 20
FETCH_WINDOW_MIN_DAYS = 7
FETCH_WINDOW_MAX_DAYS = 30
FB_EVENTS_INTERVAL_DAYS = 7
"""The structured events tab is the priciest actor; events are announced weeks ahead anyway."""
FB_EVENTS_PER_FETCH = 20
FAILURES_BEFORE_ALERT = 2


class StoredEvent(PostEvent):
    verifier: Verdict | None = None
    verifier_reason: str | None = None


class PostRecord(BaseModel):
    id: str
    url: str
    published: date
    analyzed: date
    is_announcement: bool = True
    events: list[StoredEvent] = Field(default_factory=list)
    notes: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    """Pictures of the post (the flyer the vision model read). The first one is the visual of
    the events it announced; CDN URLs expire within days, so it is cached at publication time."""


class SocialSource(BaseModel):
    kind: SocialKind
    handle: str
    url: str
    """Profile / page URL: the `source_url` of the published events."""
    last_fetched: date | None = None
    last_ok: date | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    posts: dict[str, PostRecord] = Field(default_factory=dict)
    fb_events: list[FacebookEvent] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.handle}"


class SocialFile(BaseModel):
    sources: dict[str, SocialSource] = Field(default_factory=dict)


def load_social(path: Path) -> SocialFile:
    if path.exists():
        return SocialFile.model_validate_json(path.read_text(encoding="utf-8"))
    return SocialFile()


def save_social(path: Path, data: SocialFile) -> None:
    path.write_text(data.model_dump_json(indent=1), encoding="utf-8")


# ------------------------------------------------------------------ which sources for a venue


def venue_social_sources(venue: Venue, record: SourceRecord | None, social: SocialFile) -> list[SocialSource]:
    """Configured handles win; otherwise handles seen by the discovery agent."""
    out: list[SocialSource] = []
    ig = venue.instagram or (record.source.instagram if record else None)
    fb = venue.facebook or (record.source.facebook if record else None)
    if ig:
        url = apify.instagram_url(ig)
        out.append(_get(social, "instagram", url.rstrip("/").rsplit("/", 1)[-1], url))
    if fb:
        url = apify.facebook_url(fb)
        slug = url.rstrip("/").split("facebook.com/", 1)[-1]
        out.append(_get(social, "facebook", slug, url))
        out.append(_get(social, "facebook-events", slug, url.rstrip("/") + "/upcoming_hosted_events"))
    return out


def _get(social: SocialFile, kind: SocialKind, handle: str, url: str) -> SocialSource:
    key = f"{kind}:{handle}"
    src = social.sources.get(key)
    if src is None:
        src = SocialSource(kind=kind, handle=handle, url=url)
        social.sources[key] = src
    return src


# ------------------------------------------------------------------ outcome


Published = tuple[SchemaEvent, Verdict | None, str | None]
"""An event ready to publish with the verifier verdict recorded when it was first seen."""
_DAYS_RE = re.compile(r"(\d+)\s*day")


@dataclass
class SocialOutcome:
    events: list[Published] = field(default_factory=list)
    error: str | None = None
    """Set when the fetch failed; cached events are still returned."""
    fetched: bool = False
    posts_analyzed: int = 0
    llm_calls: int = 0
    rejected_checks: int = 0
    rejected_verifier: int = 0
    uncertain: int = 0
    notes: list[str] = field(default_factory=list)


def due(src: SocialSource, today: date) -> bool:
    if src.last_fetched is None:
        return True
    interval = FB_EVENTS_INTERVAL_DAYS if src.kind == "facebook-events" else 1
    return (today - src.last_fetched).days >= interval


def fetch_window(src: SocialSource, today: date) -> date:
    """Only posts newer than this date are requested (each returned post is billed)."""
    anchor = src.last_ok or (today - timedelta(days=FETCH_WINDOW_MAX_DAYS))
    start = anchor - timedelta(days=2)
    start = min(start, today - timedelta(days=FETCH_WINDOW_MIN_DAYS))
    return max(start, today - timedelta(days=FETCH_WINDOW_MAX_DAYS))


# ------------------------------------------------------------------ processing


class SocialRunner:
    def __init__(self, settings: Settings, today: date, *, extractor, verifier, allow_llm: bool, verify: bool):
        self.settings = settings
        self.today = today
        self.extractor = extractor
        self.verifier = verifier
        self.allow_llm = allow_llm
        self.verify = verify

    async def process(self, venue: Venue, src: SocialSource, client: httpx.Client) -> SocialOutcome:
        out = SocialOutcome()
        new_posts: list[SocialPost] = []
        if due(src, self.today):
            try:
                new_posts = await asyncio.to_thread(self.fetch, src)
                out.fetched = True
                src.last_ok = self.today
                src.last_error = None
                src.consecutive_failures = 0
            except ApifyError as exc:
                src.consecutive_failures += 1
                src.last_error = str(exc)
                log.warning("[%s] %s fetch failed: %s", venue.id, src.key, exc)
                if src.consecutive_failures >= FAILURES_BEFORE_ALERT:
                    out.error = f"{src.kind} fetch failed {src.consecutive_failures}x: {exc}"
            src.last_fetched = self.today

        if src.kind == "facebook-events":
            out.events = [(e, None, None) for e in (fb_event_to_schema(x, self.today) for x in src.fb_events) if e is not None]
            return out

        await self.analyze_new_posts(venue, src, new_posts, client, out)
        self.prune(src)
        out.events = rebuild(src, self.today)
        return out

    # -- fetch (sync, runs in a thread); returns the posts not seen before

    def fetch(self, src: SocialSource) -> list[SocialPost]:
        token = self.settings.apify_token
        assert token
        if src.kind == "instagram":
            items = apify.run_actor(token, apify.IG_ACTOR,
                                    apify.instagram_input(src.url, limit=POSTS_PER_FETCH, newer_than=fetch_window(src, self.today)))
            posts = apify.instagram_posts(items)
        elif src.kind == "facebook":
            items = apify.run_actor(token, apify.FB_POSTS_ACTOR,
                                    apify.facebook_posts_input(src.url, limit=POSTS_PER_FETCH, newer_than=fetch_window(src, self.today)))
            posts = apify.facebook_posts(items)
        else:
            items = apify.run_actor(token, apify.FB_EVENTS_ACTOR,
                                    apify.facebook_events_input(src.url.replace("/upcoming_hosted_events", ""), limit=FB_EVENTS_PER_FETCH))
            src.fb_events = [e for e in apify.facebook_events(items) if not e.is_past]
            log.info("%s: %d upcoming structured events", src.key, len(src.fb_events))
            return []
        new = [p for p in posts if p.id not in src.posts]
        log.info("%s: %d posts returned, %d new", src.key, len(posts), len(new))
        return new

    # -- vision extraction of new posts

    async def analyze_new_posts(self, venue: Venue, src: SocialSource, pending: list[SocialPost], client: httpx.Client,
                                out: SocialOutcome) -> None:
        if not pending:
            return
        if not self.allow_llm:
            # Not recorded: they will come back as "new" on the next fetch window.
            out.notes.append(f"{len(pending)} new {src.kind} posts but LLM disabled")
            return
        sem = asyncio.Semaphore(3)

        async def one(post: SocialPost) -> PostRecord:
            async with sem:
                return await self.analyze_post(venue, src, post, client, out)

        for rec in await asyncio.gather(*[one(p) for p in pending], return_exceptions=True):
            if isinstance(rec, BaseException):
                log.error("[%s] %s post analysis failed: %s", venue.id, src.key, rec)
                out.notes.append(f"{src.kind} post analysis error: {type(rec).__name__}: {rec}")
                continue
            src.posts[rec.id] = rec

    async def analyze_post(self, venue: Venue, src: SocialSource, post: SocialPost, client: httpx.Client,
                           out: SocialOutcome) -> PostRecord:
        images = await asyncio.to_thread(download_images, client, post.image_urls)
        if not post.text.strip() and not images:
            return PostRecord(id=post.id, url=post.url, published=post.published_date, analyzed=self.today,
                              is_announcement=False, notes="no caption and no readable image", image_urls=post.image_urls)
        deps = PostDeps(post=post, today=self.today, venue_name=f"{venue.name} ({venue.city})", images=images)
        run = await extract_post(self.extractor, deps)
        out.llm_calls += run.usage.requests
        out.posts_analyzed += 1
        result = run.output
        out.rejected_checks += len(deps.rejected)
        for ev, problems in deps.rejected:
            if not any("past" in p for p in problems):
                out.notes.append(f"{src.kind} rejected {ev.title!r} {ev.start_date}: {'; '.join(problems)}")
        rec = PostRecord(id=post.id, url=post.url, published=post.published_date, analyzed=self.today,
                         is_announcement=result.is_announcement, notes=result.notes, image_urls=post.image_urls)
        log.info("[%s] %s post %s (%s): %d event(s) %s", venue.id, src.kind, post.id, post.published_date,
                 len(result.events), "" if result.is_announcement else "(not an announcement)")
        for ev in result.events:
            stored = StoredEvent(**ev.model_dump())
            if self.verify and self.verifier is not None:
                vres = await verify_post_event(self.verifier, ev, post, images, f"{venue.name} ({venue.city})", self.today)
                out.llm_calls += vres.usage.requests
                stored.verifier, stored.verifier_reason = vres.output.verdict, vres.output.reason
                if vres.output.verdict == "reject":
                    out.rejected_verifier += 1
                    out.notes.append(f"{src.kind} verifier rejected {ev.title!r} {ev.start_date}: {vres.output.reason}")
                elif vres.output.verdict == "uncertain":
                    out.uncertain += 1
                    out.notes.append(f"{src.kind} verifier uncertain {ev.title!r} {ev.start_date}: {vres.output.reason}")
            rec.events.append(stored)
        return rec

    def prune(self, src: SocialSource) -> None:
        limit = self.today - timedelta(days=POST_RETENTION_DAYS)
        for pid in [pid for pid, p in src.posts.items() if p.published < limit]:
            del src.posts[pid]


# ------------------------------------------------------------------ rebuild published events


def rebuild(src: SocialSource, today: date) -> list[Published]:
    """Every upcoming, accepted event announced by cached posts. Newest post wins on ties."""
    by_key: dict[tuple[str, date], Published] = {}
    for post in sorted(src.posts.values(), key=lambda p: p.published):
        for ev in post.events:
            if ev.verifier in ("reject", "uncertain"):
                continue
            if (ev.end_date or ev.start_date) < today:
                continue
            se = SchemaEvent(**ev.model_dump(exclude={"verifier", "verifier_reason", "from_image"}), grounded_text=False,
                             image=post.image_urls[0] if post.image_urls else None)
            se.url = se.url or post.url
            by_key[(se.title.strip().lower(), se.start_date)] = (se, ev.verifier, ev.verifier_reason)
    return list(by_key.values())


def fb_event_to_schema(e: FacebookEvent, today: date) -> SchemaEvent | None:
    if e.start is None or e.is_past:
        return None
    local = e.start.astimezone(TZ)
    end_date = None
    if e.duration:
        m = _DAYS_RE.search(e.duration)
        if m:
            end_date = local.date() + timedelta(days=int(m.group(1)) - 1)
    if (end_date or local.date()) < today:
        return None
    summary = " ".join(e.description.split())[:200] or None
    return SchemaEvent(
        title=e.name, start_date=local.date(), start_time=local.time().replace(second=0, microsecond=0),
        end_date=end_date, location_name=e.location_name, url=e.ticket_url or e.url, price=e.price, summary=summary,
        status="cancelled" if e.cancelled else "scheduled", image=e.image_url or None,
        evidence=[e.date_sentence or local.strftime("%Y-%m-%d %H:%M")], grounded_text=False,
    )
