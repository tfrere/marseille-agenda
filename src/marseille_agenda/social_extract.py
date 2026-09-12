"""Vision extraction of events from social posts (caption + flyer images).

A post is unstructured by nature: the date is often only on the flyer, the year is almost
never written, and venues repost other venues' events. So the guarantees are different
from the web path and enforced deterministically here:

- caption-grounded fields must be verbatim substrings of the caption;
- image-grounded fields are allowed only when the post has images;
- an announced event happens on or after the publication date and within a year of it
  (this pins the year and rejects recaps of past events);
- weekday named in the evidence must match the date.

Everything that passes is then challenged by a second vision model (different family).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext
from pydantic_ai.models import Model

from .apify import SocialPost
from .schema import ExtractedEvent, VerifierVerdict
from .validate import FRENCH_WEEKDAYS, _WEEKDAY_RE, normalize, strip_accents

log = logging.getLogger("marseille_agenda.social")

MAX_IMAGE_BYTES = 6_000_000
# An announcement precedes its event by at most this many days. Under 365 on purpose: a
# wrong year inferred by the model is always at least 365 days off and gets caught.
MAX_ANNOUNCE_LEAD_DAYS = 300
_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.I)


class PostEvent(ExtractedEvent):
    from_image: bool = Field(
        default=False,
        description="True when the title or the date was read from an image (flyer) rather than from the caption.",
    )


class PostExtraction(BaseModel):
    events: list[PostEvent] = Field(default_factory=list)
    is_announcement: bool = Field(description="True when the post announces one or more upcoming events at the venue.")
    notes: str | None = Field(default=None, description="Anything odd: recap of a past event, event elsewhere, unreadable flyer.")


INSTRUCTIONS = """\
You read one social media post (caption + attached images) published by a cultural venue in
Marseille, France, and list the upcoming public events it announces AT THAT VENUE.

Rules:
- Only events that take place at the venue (or are organised by it on site). If the post shares
  an event happening somewhere else, or a recap / thanks / photos of a past event, return no
  events and say so in `notes`.
- Read the flyer images carefully: they usually carry the date, time and title. Set
  `from_image=true` on an event whose title or date you read from an image.
- Year: it is almost never written. The post was published on the given date; an announced
  event happens on or after that date, normally within the next three months. Pick the year
  accordingly and check the weekday if one is written.
- Caption-grounded fields (when `from_image=false`) must be copied VERBATIM from the caption:
  title, evidence, url. Never paraphrase them. For image-grounded events, `evidence` is your
  transcription of the decisive words on the flyer (date, time, title).
- Times: "19h", "19h30", "20:00" -> start_time. A door-opening time is not the start time when a
  separate show time is written.
- One event per distinct date. A programme with several dates gives several events. A single
  event spanning several days (exhibition, festival) gives one event with start_date/end_date.
- `url`: a ticketing/detail link written in the caption, verbatim; else null. Do not invent.
- `location_name`: the room or the venue name as written; null if not written.
Output the structured result only.
"""


@dataclass
class PostDeps:
    post: SocialPost
    today: date
    venue_name: str
    images: list[BinaryContent] = field(default_factory=list)
    rejected: list[tuple[PostEvent, list[str]]] = field(default_factory=list)


# ------------------------------------------------------------------ deterministic checks


def check_post_event(ev: PostEvent, post: SocialPost, today: date, *, has_images: bool) -> list[str]:
    problems: list[str] = []
    caption = normalize(post.text)
    caption_noacc = strip_accents(caption)

    def in_caption(s: str) -> bool:
        n = normalize(s)
        return bool(n) and (n in caption or strip_accents(n) in caption_noacc)

    if ev.from_image and not has_images:
        problems.append("from_image is true but the post has no readable image")
    if not ev.from_image:
        for q in ev.evidence:
            if len(normalize(q)) < 8:
                problems.append(f"evidence too short: {q!r}")
            elif not in_caption(q):
                problems.append(f"evidence not found verbatim in caption: {q[:80]!r}")
        if not in_caption(ev.title):
            problems.append(f"title not found in caption: {ev.title!r}")

    if ev.url:
        known = set(_URL_RE.findall(post.text))
        u = ev.url.strip()
        if not ({u, u.rstrip("/")} & (known | {k.rstrip("/") for k in known})):
            # A URL transcribed from a flyer cannot be checked: drop it rather than the event.
            ev.url = None

    pub = post.published_date
    last_day = ev.end_date or ev.start_date
    if last_day < today:
        problems.append(f"event is in the past ({ev.start_date} .. {last_day}, today {today})")
    if ev.start_date < pub:
        problems.append(f"start_date {ev.start_date} is before the post publication date {pub} (recap or wrong year)")
    if ev.start_date > pub + timedelta(days=MAX_ANNOUNCE_LEAD_DAYS):
        problems.append(f"start_date {ev.start_date} is {(ev.start_date - pub).days} days after the post ({pub}): wrong year")
    if ev.end_date and ev.end_date < ev.start_date:
        problems.append(f"end_date {ev.end_date} before start_date {ev.start_date}")

    weekdays = {m.lower() for m in _WEEKDAY_RE.findall(" ".join(ev.evidence))}
    if len(weekdays) == 1:
        named = weekdays.pop()
        actual = FRENCH_WEEKDAYS[ev.start_date.weekday()]
        if named != actual:
            problems.append(f"weekday mismatch: evidence says {named} but {ev.start_date} is a {actual}")
    return problems


# ------------------------------------------------------------------ images


def download_images(client: httpx.Client, urls: list[str]) -> list[BinaryContent]:
    out: list[BinaryContent] = []
    for u in urls:
        try:
            r = client.get(u, timeout=30, follow_redirects=True)
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if r.status_code != 200 or not ctype.startswith("image/") or len(r.content) > MAX_IMAGE_BYTES:
                log.debug("skip image %s: status=%s type=%s size=%d", u[:80], r.status_code, ctype, len(r.content))
                continue
            out.append(BinaryContent(data=r.content, media_type=ctype))
        except httpx.HTTPError as e:
            log.debug("image download failed %s: %s", u[:80], e)
    return out


# ------------------------------------------------------------------ extractor


def build_post_extractor(model: Model | str) -> Agent[PostDeps, PostExtraction]:
    agent: Agent[PostDeps, PostExtraction] = Agent(
        model, deps_type=PostDeps, output_type=PostExtraction, instructions=INSTRUCTIONS, retries=2, name="post_extractor",
    )

    @agent.output_validator
    async def grounded(ctx: RunContext[PostDeps], output: PostExtraction) -> PostExtraction:
        deps = ctx.deps
        good: list[PostEvent] = []
        bad: list[tuple[PostEvent, list[str]]] = []
        seen: set[tuple[str, date]] = set()
        for ev in output.events:
            problems = check_post_event(ev, deps.post, deps.today, has_images=bool(deps.images))
            key = (normalize(ev.title), ev.start_date)
            if key in seen:
                problems.append("duplicate of another event in this answer (same title and date)")
            seen.add(key)
            if problems:
                bad.append((ev, problems))
            else:
                good.append(ev)
        if bad and ctx.retry < ctx.max_retries:
            lines = [f"- {ev.title!r} ({ev.start_date}): " + "; ".join(p) for ev, p in bad]
            raise ModelRetry(
                "Some events failed deterministic checks. Fix them (copy caption text verbatim, re-check the "
                "year against the publication date and the weekday) or drop them, then return the full list:\n"
                + "\n".join(lines)
            )
        deps.rejected = bad
        return PostExtraction(events=good, is_announcement=output.is_announcement, notes=output.notes)

    return agent


def post_prompt(deps: PostDeps) -> list[str | BinaryContent]:
    p = deps.post
    weekday = FRENCH_WEEKDAYS[deps.today.weekday()]
    pub_weekday = FRENCH_WEEKDAYS[p.published_date.weekday()]
    text = (
        f"Today is {weekday} {deps.today.isoformat()}. Venue: {deps.venue_name}.\n"
        f"Post published on {pub_weekday} {p.published_date.isoformat()} on {p.platform}: {p.url}\n"
        f"Images attached: {len(deps.images)}\n\n"
        f"=== CAPTION START ===\n{p.text or '(no caption)'}\n=== CAPTION END ==="
        + (f"\n\nTEXT THE PLATFORM READ ON THE IMAGES (hint only, may be wrong; trust the images):\n{p.image_text}" if p.image_text else "")
    )
    return [text, *deps.images]


async def extract_post(agent: Agent[PostDeps, PostExtraction], deps: PostDeps):
    return await agent.run(post_prompt(deps), deps=deps)


# ------------------------------------------------------------------ verifier

VERIFIER_INSTRUCTIONS = """\
You are an adversarial fact-checker for a public events calendar in Marseille, France.
Another system extracted ONE event from a social media post (caption + images). Try to prove it
wrong using only the post. Be sceptical but fair.

Reject when any of these holds:
- The post is a recap, thanks or photos of an event that already happened, not an announcement.
- The event takes place at another venue than the one given (the venue merely shares it).
- The date, time or weekday written in the caption or on the flyer contradicts the extracted
  values (the publication date is given: the event must be on or after it, and the year must
  make the written weekday match).
- It is not a dated public event: opening hours, a call for applications, merchandise, a
  workshop series with no concrete date.
- The post says the event is cancelled / postponed / sold out but the extracted status disagrees.

Return "uncertain" when the flyer is unreadable or the date is genuinely ambiguous. Return
"accept" only when every check passes. `reason` must quote the decisive words.
"""


@dataclass
class PostVerifyDeps:
    today: date


def build_post_verifier(model: Model | str) -> Agent[PostVerifyDeps, VerifierVerdict]:
    return Agent(model, deps_type=PostVerifyDeps, output_type=VerifierVerdict, instructions=VERIFIER_INSTRUCTIONS,
                 retries=1, name="post_verifier")


def verifier_prompt(ev: PostEvent, post: SocialPost, images: list[BinaryContent], venue_name: str,
                    today: date) -> list[str | BinaryContent]:
    ev_weekday = FRENCH_WEEKDAYS[ev.start_date.weekday()]
    text = (
        f"Today is {FRENCH_WEEKDAYS[today.weekday()]} {today.isoformat()}. Venue: {venue_name}.\n"
        f"Post published on {FRENCH_WEEKDAYS[post.published_date.weekday()]} {post.published_date.isoformat()} "
        f"on {post.platform}: {post.url}\n\n"
        f"EXTRACTED EVENT:\n{ev.model_dump_json(indent=1, exclude={'evidence'})}\n"
        f"(start_date {ev.start_date} is a {ev_weekday})\n\n"
        f"EVIDENCE GIVEN BY THE EXTRACTOR:\n" + "\n".join(f"- {q}" for q in ev.evidence) + "\n\n"
        f"=== CAPTION START ===\n{post.text or '(no caption)'}\n=== CAPTION END ===\n"
        f"Images attached: {len(images)}"
    )
    return [text, *images]


async def verify_post_event(agent: Agent[PostVerifyDeps, VerifierVerdict], ev: PostEvent, post: SocialPost,
                            images: list[BinaryContent], venue_name: str, today: date):
    return await agent.run(verifier_prompt(ev, post, images, venue_name, today), deps=PostVerifyDeps(today=today))
