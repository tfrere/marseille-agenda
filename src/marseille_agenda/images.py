"""Download the visual of every published event once and cache it as a small WebP.

Deterministic, no model. Runs at publication time: `site/img/<uid>.webp` for each published
event whose `image_source` could be fetched and read as a picture, `image = None` otherwise
(the site shows an honest placeholder). Failures are remembered in the state so a dead URL is
not retried every day; files of events no longer published are removed.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from .schema import Event, State

log = logging.getLogger("marseille_agenda.images")

MAX_SIDE_PX = 640
WEBP_QUALITY = 78
MIN_SIDE_PX = 80
"""Anything smaller is an icon or a tracking pixel, not a poster."""
MAX_BYTES = 10 * 1024 * 1024
TIMEOUT_S = 15.0
RETRY_FAILED_AFTER_DAYS = 14
IMG_SUBDIR = "img"
# Hotlink protection usually keys on the User-Agent and the Referer: look like a browser
# coming from the venue's page.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


class ImageError(Exception):
    """The URL did not yield a usable picture (HTTP error, not an image, too small, too big)."""


@dataclass
class ImageStats:
    cached: int = 0
    """Published events whose file already existed."""
    downloaded: int = 0
    failed: int = 0
    skipped_recent_failure: int = 0
    without_source: int = 0
    removed: int = 0
    """Stale files deleted from the image directory."""
    failures: Counter = field(default_factory=Counter)
    """Failure reason counts, keyed by (host, reason)."""

    def describe(self) -> str:
        return (f"{self.cached} cached, {self.downloaded} downloaded, {self.failed} failed, "
                f"{self.skipped_recent_failure} skipped (recent failure), {self.without_source} without source, "
                f"{self.removed} stale file(s) removed")


def make_image_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(TIMEOUT_S),
                        headers={"User-Agent": BROWSER_USER_AGENT, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})


def sync_images(events: list[Event], state: State, site_dir: Path, today: date, client: httpx.Client | None = None) -> ImageStats:
    """Give every published event its cached visual, downloading what is missing.

    Sets `event.image` to `img/<uid>.webp` (relative to the site) or None. `events` are the
    published events; every other file in `site_dir/img` is an orphan and is removed.
    """
    img_dir = site_dir / IMG_SUBDIR
    img_dir.mkdir(parents=True, exist_ok=True)
    stats = ImageStats()
    own = client is None
    client = client or make_image_client()
    try:
        for e in events:
            _sync_one(e, state, img_dir, today, client, stats)
    finally:
        if own:
            client.close()
    published = {e.uid for e in events}
    for f in img_dir.glob("*.webp"):
        if f.stem not in published:
            f.unlink()
            stats.removed += 1
    # Forget failures old enough to be retried anyway: keeps the state small.
    limit = today - timedelta(days=RETRY_FAILED_AFTER_DAYS)
    for url in [u for u, d in state.image_failures.items() if d < limit]:
        del state.image_failures[url]
    return stats


def _sync_one(e: Event, state: State, img_dir: Path, today: date, client: httpx.Client, stats: ImageStats) -> None:
    if not e.image_source:
        e.image = None
        stats.without_source += 1
        return
    target = img_dir / f"{e.uid}.webp"
    rel = f"{IMG_SUBDIR}/{e.uid}.webp"
    if target.exists() and target.stat().st_size > 0:
        e.image = rel
        stats.cached += 1
        return
    e.image = None
    failed_on = state.image_failures.get(e.image_source)
    if failed_on is not None and (today - failed_on).days < RETRY_FAILED_AFTER_DAYS:
        stats.skipped_recent_failure += 1
        return
    try:
        data = download_image(client, e.image_source, referer=e.source_url if e.source_url.startswith("http") else e.url)
        save_thumbnail(data, target)
    except ImageError as exc:
        host = urlsplit(e.image_source).hostname or "?"
        log.debug("image for %r (%s) skipped: %s", e.title, e.image_source, exc)
        state.image_failures[e.image_source] = today
        stats.failed += 1
        stats.failures[(host, str(exc).split(":")[0])] += 1
        return
    state.image_failures.pop(e.image_source, None)
    e.image = rel
    stats.downloaded += 1


def download_image(client: httpx.Client, url: str, referer: str | None = None) -> bytes:
    headers = {"Referer": referer} if referer else {}
    try:
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code >= 400:
                raise ImageError(f"HTTP {resp.status_code}")
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype and not ctype.startswith("image/") and ctype != "application/octet-stream":
                raise ImageError(f"not an image: content-type {ctype}")
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                raise ImageError(f"too big: {declared} bytes declared")
            buf = BytesIO()
            for chunk in resp.iter_bytes():
                buf.write(chunk)
                if buf.tell() > MAX_BYTES:
                    raise ImageError(f"too big: more than {MAX_BYTES} bytes")
    except httpx.HTTPError as exc:
        raise ImageError(f"request failed: {type(exc).__name__}: {exc}") from exc
    return buf.getvalue()


def save_thumbnail(data: bytes, target: Path) -> tuple[int, int]:
    """Decode, drop transparency onto white, fit in MAX_SIDE_PX and write a WebP. Returns the size."""
    try:
        im = Image.open(BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError(f"not an image: {type(exc).__name__}") from exc
    im = ImageOps.exif_transpose(im) or im
    if min(im.size) < MIN_SIDE_PX:
        raise ImageError(f"too small: {im.size[0]}x{im.size[1]}")
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(background, rgba)
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    im.save(tmp, "WEBP", quality=WEBP_QUALITY, method=4)
    tmp.replace(target)
    return im.size
