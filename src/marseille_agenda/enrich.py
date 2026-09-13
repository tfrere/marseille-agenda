"""Complete titles a listing cut short, from the event's own page. Deterministic, grounded.

A home page or card grid often truncates titles ("Le Grand Voyage aux Fertiles..."). The
detail page linked by the card carries the full one. It is accepted only when, after
whitespace/case/accent normalization, it starts with the listing prefix and is longer: the
published title is then verbatim on the linked page and verified to extend what the listing
showed. Nothing is ever invented; when no candidate matches, the prefix stays.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

from .schema import Event, State
from .validate import normalize, strip_accents

log = logging.getLogger("marseille_agenda.enrich")

MAX_FETCHES_PER_VENUE = 20
TIMEOUT_S = 15.0
_SITE_SUFFIX = re.compile(r"\s+[-|\u2013\u2014\u00b7]\s+([^-|\u2013\u2014\u00b7]{1,60})$")
"""Trailing ' - Site name' / ' | Site name' of a <title>; group 1 is the site name."""


@dataclass
class EnrichStats:
    completed: int = 0
    unchanged: int = 0
    """Detail page consulted, no candidate extending the prefix."""
    cached: int = 0
    """Answered from the previous run's state, no fetch."""
    errors: int = 0
    skipped: int = 0
    """No usable URL (missing or on another host) or fetch budget exhausted."""

    def describe(self) -> str:
        return (f"{self.completed} completed, {self.unchanged} unchanged, {self.cached} from cache, "
                f"{self.errors} fetch errors, {self.skipped} skipped")


def _key(s: str) -> str:
    return strip_accents(normalize(s))


def extends_prefix(candidate: str, prefix: str) -> bool:
    kc, kp = _key(candidate), _key(prefix)
    return bool(kp) and kc.startswith(kp) and len(kc) > len(kp)


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def _meta(soup: BeautifulSoup, prop: str) -> str:
    tag = soup.find("meta", attrs={"property": prop})
    return _clean(tag.get("content")) if tag is not None and isinstance(tag.get("content"), str) else ""


def site_name(soup: BeautifulSoup, candidates: list[str]) -> str | None:
    """The site name CMSs append to titles ("X - La Fabulerie"): `og:site_name` when declared,
    else the last segment of <title> when another candidate confirms it (ends with the same
    segment, or equals the title without it), or when <title> is the only candidate. A segment
    seen once in a lone og:title/h1 is part of the title and stays."""
    declared = _meta(soup, "og:site_name")
    if declared:
        return declared
    if soup.title is None:
        return None
    full = _clean(soup.title.get_text(" ", strip=True))
    m = _SITE_SUFFIX.search(full)
    if not m:
        return None
    tail, bare = m.group(1).lower(), full[: m.start()].lower()
    others = [c.lower() for c in candidates if c != full]
    if not others or any(c.endswith(tail) or c == bare for c in others):
        return m.group(1)
    return None


def _strip_site(text: str, site: str | None) -> str:
    if not site:
        return text
    return re.sub(r"\s+[-|\u2013\u2014\u00b7]\s+" + re.escape(site) + r"$", "", text, flags=re.IGNORECASE).strip()


def title_candidates(html: str) -> list[str]:
    """Full-title candidates of a detail page, best first: og:title, h1, <title>, each without
    the trailing ' - Site name' / ' | Site name' segment (the site name would otherwise extend
    the prefix too and end up published as part of the title)."""
    soup = BeautifulSoup(html, "lxml")
    raw: list[str] = []
    for text in [_meta(soup, "og:title"), *(h1.get_text(" ", strip=True) for h1 in soup.find_all("h1")[:3]),
                 soup.title.get_text(" ", strip=True) if soup.title is not None else ""]:
        t = _clean(text)
        if t and t not in raw:
            raw.append(t)
    site = site_name(soup, raw)
    out: list[str] = []
    for t in raw:
        t = _strip_site(t, site)
        if t and t not in out:
            out.append(t)
    return out


def full_title(html: str, prefix: str) -> str | None:
    return next((c for c in title_candidates(html) if extends_prefix(c, prefix)), None)


def complete_truncated_titles(events: list[Event], state: State, client: httpx.Client,
                              *, max_fetches: int = MAX_FETCHES_PER_VENUE) -> EnrichStats:
    """Complete the truncated titles of one venue's freshly extracted events.

    The previous run's state answers first (a detail page is fetched once per uid: the
    outcome, completed or not, is remembered in `title_completed_from`); then at most
    `max_fetches` detail pages on the source's own host are read.
    """
    stats = EnrichStats()
    fetches = 0
    pages: dict[str, str | None] = {}
    """Detail pages read this run (a weekly workshop links every session to the same page)."""
    for e in events:
        if not e.title_truncated:
            continue
        if not e.url or urlsplit(e.url).hostname != urlsplit(e.source_url).hostname:
            stats.skipped += 1
            continue
        prev = state.events.get(e.uid)
        if prev is not None and prev.title_completed_from == e.url and _key(prev.title).startswith(_key(e.title)):
            e.title, e.title_completed_from = prev.title, prev.title_completed_from
            stats.cached += 1
            continue
        if e.url not in pages:
            if fetches >= max_fetches:
                stats.skipped += 1
                continue
            fetches += 1
            try:
                resp = client.get(e.url, timeout=TIMEOUT_S)
                resp.raise_for_status()
                pages[e.url] = resp.text
            except httpx.HTTPError as exc:
                log.debug("title lookup for %r failed: %s", e.title, exc)
                pages[e.url] = None
        html = pages[e.url]
        if html is None:
            stats.errors += 1
            continue
        found = full_title(html, e.title)
        e.title_completed_from = e.url
        if found is None:
            log.debug("title lookup for %r: no candidate extends the prefix on %s", e.title, e.url)
            stats.unchanged += 1
            continue
        log.info("title completed: %r -> %r", e.title, found)
        e.title = found
        stats.completed += 1
    return stats
