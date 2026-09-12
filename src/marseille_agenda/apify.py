"""Apify client and normalizers for the Instagram / Facebook actors.

Meta's public endpoints reject anonymous clients (401/400 even from residential IPs), so
social feeds are read through Apify's maintained actors, which emulate the mobile apps
behind rotating residential proxies. This module keeps the actor-specific shapes out of
the rest of the pipeline: everything downstream sees `SocialPost` and `FacebookEvent`.

Actors (pay per result, no platform fee):
  apify/instagram-api-scraper   posts of a public profile      ~1.4 USD / 1000 posts
  apify/facebook-posts-scraper  posts of a public page         ~2.0 USD / 1000 posts
  apify/facebook-events-scraper upcoming events hosted by page ~13  USD / 1000 events
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from .config import TZ

log = logging.getLogger("marseille_agenda.apify")

API = "https://api.apify.com/v2"
IG_ACTOR = "apify~instagram-api-scraper"
FB_POSTS_ACTOR = "apify~facebook-posts-scraper"
FB_EVENTS_ACTOR = "apify~facebook-events-scraper"
# run-sync endpoint: the actor must finish within this many seconds (Apify caps at 300).
RUN_TIMEOUT_S = 240
MAX_IMAGES_PER_POST = 4

Platform = Literal["instagram", "facebook"]


class SocialPost(BaseModel):
    platform: Platform
    id: str
    url: str
    published: datetime
    text: str = ""
    image_urls: list[str] = Field(default_factory=list)
    image_text: str = ""
    """Text the platform itself read on the images (Facebook `ocrText`, Instagram `alt`). A hint
    for the model, never a grounding source: only the caption and the images themselves are."""

    @property
    def published_date(self) -> date:
        return self.published.astimezone(TZ).date()


class FacebookEvent(BaseModel):
    """A structured event from a page's `upcoming_hosted_events` tab (no LLM needed)."""

    id: str
    url: str
    name: str
    start: datetime | None
    date_sentence: str
    description: str = ""
    location_name: str | None = None
    cancelled: bool = False
    is_past: bool = False
    duration: str | None = None
    price: str | None = None
    ticket_url: str | None = None
    image_url: str | None = None


class ApifyError(RuntimeError):
    pass


# ------------------------------------------------------------------ handles


def instagram_url(handle_or_url: str) -> str:
    h = handle_or_url.strip()
    m = re.search(r"instagram\.com/([A-Za-z0-9_.]+)", h)
    handle = m.group(1) if m else h.lstrip("@").strip("/")
    return f"https://www.instagram.com/{handle}/"


def facebook_url(slug_or_url: str) -> str:
    s = slug_or_url.strip()
    m = re.search(r"facebook\.com/([^?#\s]+)", s)
    slug = (m.group(1) if m else s).strip("/")
    slug = re.sub(r"/(events|upcoming_hosted_events|past_hosted_events|posts|about)$", "", slug)
    return f"https://www.facebook.com/{slug}/"


# ------------------------------------------------------------------ client


def run_actor(token: str, actor: str, run_input: dict[str, Any], *, client: httpx.Client | None = None,
              timeout_s: int = RUN_TIMEOUT_S) -> list[dict[str, Any]]:
    """Run an actor synchronously and return its dataset items.

    Raises ApifyError on HTTP errors, actor failure or timeout. Never retries on its own:
    every run costs money, the caller decides.
    """
    url = f"{API}/acts/{actor}/run-sync-get-dataset-items"
    params = {"token": token, "timeout": timeout_s, "format": "json", "clean": "true"}
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(timeout_s + 30, connect=20))
    try:
        r = client.post(url, params=params, json=run_input)
    except httpx.HTTPError as e:
        raise ApifyError(f"{actor}: {e.__class__.__name__}: {e}") from e
    finally:
        if own:
            client.close()
    if r.status_code == 408:
        raise ApifyError(f"{actor}: actor run timed out after {timeout_s}s")
    if r.status_code >= 400:
        raise ApifyError(f"{actor}: HTTP {r.status_code}: {r.text[:300]}")
    try:
        items = r.json()
    except ValueError as e:
        raise ApifyError(f"{actor}: non-JSON response") from e
    if not isinstance(items, list):
        raise ApifyError(f"{actor}: unexpected response shape {type(items).__name__}")
    # Actors report per-item problems as items with an `error` key instead of failing.
    errors = [str(it.get("error")) for it in items if isinstance(it, dict) and it.get("error")]
    items = [it for it in items if isinstance(it, dict) and not it.get("error")]
    if errors and not items:
        if all(e == "no_results" for e in errors):
            return []  # e.g. a page with no upcoming hosted event: a legitimate empty answer
        raise ApifyError(f"{actor}: {errors[0]}")
    for e in errors:
        log.warning("%s: item error: %s", actor, e)
    return items


# ------------------------------------------------------------------ inputs


def instagram_input(profile_url: str, *, limit: int, newer_than: date) -> dict[str, Any]:
    return {
        "directUrls": [profile_url],
        "resultsType": "posts",
        "resultsLimit": limit,
        "onlyPostsNewerThan": newer_than.isoformat(),
        "addParentData": False,
    }


def facebook_posts_input(page_url: str, *, limit: int, newer_than: date) -> dict[str, Any]:
    return {
        "startUrls": [{"url": page_url}],
        "resultsLimit": limit,
        "captionText": False,
        "onlyPostsNewerThan": newer_than.isoformat(),
    }


def facebook_events_input(page_url: str, *, limit: int) -> dict[str, Any]:
    return {"startUrls": [page_url.rstrip("/") + "/upcoming_hosted_events"], "maxEvents": limit}


# ------------------------------------------------------------------ normalizers


def _dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _dedupe(urls: list[str | None], limit: int = MAX_IMAGES_PER_POST) -> list[str]:
    out: list[str] = []
    for u in urls:
        if u and isinstance(u, str) and u.startswith("http") and u not in out:
            out.append(u)
        if len(out) >= limit:
            break
    return out


def instagram_posts(items: list[dict[str, Any]]) -> list[SocialPost]:
    posts: list[SocialPost] = []
    for it in items:
        pid = str(it.get("id") or it.get("shortCode") or "")
        published = _dt(it.get("timestamp"))
        if not pid or published is None:
            continue
        images = [it.get("displayUrl")] + list(it.get("images") or [])
        for child in it.get("childPosts") or []:
            if isinstance(child, dict):
                images.append(child.get("displayUrl"))
        url = it.get("url") or (f"https://www.instagram.com/p/{it['shortCode']}/" if it.get("shortCode") else "")
        text = str(it.get("caption") or "").strip()
        alts = [str(it.get("alt") or "")] + [str(c.get("alt") or "") for c in it.get("childPosts") or [] if isinstance(c, dict)]
        posts.append(SocialPost(platform="instagram", id=pid, url=url or "https://www.instagram.com/",
                                published=published, text=text, image_urls=_dedupe(images),
                                image_text="\n".join(dict.fromkeys(a.strip() for a in alts if a.strip()))))
    return posts


def facebook_posts(items: list[dict[str, Any]]) -> list[SocialPost]:
    posts: list[SocialPost] = []
    for it in items:
        pid = str(it.get("postId") or it.get("id") or "")
        published = _dt(it.get("time") or it.get("timestamp"))
        if not pid or published is None:
            continue
        images: list[str | None] = []
        ocr: list[str] = []
        for m in it.get("media") or []:
            if not isinstance(m, dict):
                continue
            # Photos carry `thumbnail` / `photo_image.uri`; videos carry a poster under `thumbnail`.
            photo = m.get("photo_image") if isinstance(m.get("photo_image"), dict) else {}
            images.append(photo.get("uri") or m.get("thumbnail") or m.get("image") or None)
            if m.get("ocrText"):
                ocr.append(str(m["ocrText"]).strip())
        text = str(it.get("text") or "").strip()
        posts.append(SocialPost(platform="facebook", id=pid, url=str(it.get("url") or it.get("topLevelUrl") or "https://www.facebook.com/"),
                                published=published, text=text, image_urls=_dedupe(images),
                                image_text="\n".join(dict.fromkeys(ocr))))
    return posts


def facebook_events(items: list[dict[str, Any]]) -> list[FacebookEvent]:
    events: list[FacebookEvent] = []
    for it in items:
        eid = str(it.get("id") or "")
        name = str(it.get("name") or "").strip()
        if not eid or not name:
            continue
        loc = it.get("location") if isinstance(it.get("location"), dict) else {}
        tickets = it.get("ticketsInfo") if isinstance(it.get("ticketsInfo"), dict) else {}
        events.append(FacebookEvent(
            id=eid, url=str(it.get("url") or f"https://www.facebook.com/events/{eid}/"), name=name,
            start=_dt(it.get("utcStartDate")), date_sentence=str(it.get("dateTimeSentence") or it.get("startTime") or ""),
            description=str(it.get("description") or ""), location_name=(loc.get("name") or None),
            cancelled=bool(it.get("isCanceled")), is_past=bool(it.get("isPast")), duration=it.get("duration") or None,
            price=(tickets.get("price") or None), ticket_url=(tickets.get("buyUrl") or None),
            image_url=it.get("imageUrl") or None,
        ))
    return events
