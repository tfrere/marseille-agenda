"""Source discovery agent: from a venue name, find the URL that lists its upcoming events.

Prefers machine-readable sources (JSON APIs, including ones hidden behind JS front-ends)
over server-rendered HTML listings, and verifies by fetching that the candidate actually
contains upcoming events before answering.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from .config import Settings
from .extraction_schema import DiscoveredSource
from .fetch import USER_AGENT, html_to_text, summarize_json

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
You locate the best source of UPCOMING public events for one venue or organisation in Marseille.
You have tools to search the web, fetch pages, inspect JSON, and dig up hidden APIs.

Strategy:
1. If a website is given, start there; otherwise `web_search` the venue name + "Marseille" and
   words like agenda, programme, événements. Ignore aggregators (Facebook, allevents, sortiraparis...)
   unless the venue has no site of its own.
2. Look for the agenda / programme / événements page. `fetch_page` reports signals: JSON-LD events,
   .ics links, Nuxt/Next payloads, WordPress REST hints, count of date-like strings.
3. If the listing page has few dates in its text but a JS payload signal, the events are loaded
   client-side: call `find_api_routes` on it and `fetch_json` the candidates (try query params
   seen in the code, e.g. upcoming=1, per_page=100). `probe_common_endpoints` tries standard
   WordPress/Events-Calendar routes.
4. Prefer, in order: JSON endpoint listing upcoming events > server-rendered HTML listing page.
   ICS/RSS feeds are NOT supported: do not return them.
5. VERIFY before answering: the returned URL, fetched now, must contain several upcoming events
   (dates on or after today) with titles. Put up to 5 of those titles in `sample_titles`.
6. Do not exceed ~20 tool calls. If nothing works, return the best HTML page you found with a low
   confidence and explain in `reasoning`.
"""

_AGENDA_WORDS = re.compile(r"agenda|program|event|évén|evene|calend|spectacle|concert|saison|billet|rendez|rencontre|soir", re.I)
_DATE_HINT = re.compile(
    r"\b\d{1,2}\s+(janv|févr|fevr|mars|avr|mai|juin|juil|août|aout|sept|oct|nov|déc|dec)\w*\.?(\s+\d{4})?|\b20\d{2}-\d{2}-\d{2}",
    re.I,
)
_ROUTE_RE = re.compile(r"""["'`](/(?:api|wp-json|graphql|rest|feed|data)[^"'`\s<>]{2,160})["'`]""")


@dataclass
class DiscoverDeps:
    client: httpx.AsyncClient
    settings: Settings
    today: date
    venue_name: str
    website: str | None


def _signals(html: str, text: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    jsonld_events = 0
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except json.JSONDecodeError:
            continue
        blob = json.dumps(data)
        jsonld_events += blob.count('"Event"') + blob.count('"MusicEvent"') + blob.count('"TheaterEvent"')
    ics = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True) if ".ics" in a["href"] or "ical=1" in a["href"]]
    return {
        "jsonld_events": jsonld_events,
        "ics_links": ics[:3],
        "js_payload": bool(soup.find("script", id="__NUXT_DATA__") or soup.find("script", id="__NEXT_DATA__") or "window.__NUXT__" in html),
        "wordpress": "wp-content" in html or "wp-json" in html,
        "events_calendar_plugin": "tribe-events" in html or "tribe_events" in html,
        "date_like_strings_in_text": len(_DATE_HINT.findall(text)),
        "text_chars": len(text),
    }


def build_discoverer(model: Model | str) -> Agent[DiscoverDeps, DiscoveredSource]:
    agent: Agent[DiscoverDeps, DiscoveredSource] = Agent(
        model, deps_type=DiscoverDeps, output_type=DiscoveredSource, instructions=INSTRUCTIONS, retries=2, name="discoverer",
    )

    @agent.tool
    async def web_search(ctx: RunContext[DiscoverDeps], query: str) -> str:
        """Search the web. Returns up to 8 results as 'title | url | snippet' lines."""
        return await _web_search(ctx.deps, query)

    @agent.tool
    async def fetch_page(ctx: RunContext[DiscoverDeps], url: str) -> str:
        """Fetch a URL. Returns status, content type, structural signals, a text excerpt and agenda-related links.
        If the response is JSON, returns a structural summary instead."""
        try:
            r = await ctx.deps.client.get(url)
        except httpx.HTTPError as exc:
            return f"ERROR fetching {url}: {exc}"
        ctype = r.headers.get("content-type", "")
        head = f"status={r.status_code} final_url={r.url} content_type={ctype} bytes={len(r.content)}"
        if r.status_code >= 400:
            return head
        if "json" in ctype or r.text.lstrip().startswith(("{", "[")):
            return head + "\nJSON SUMMARY:\n" + summarize_json(r.text, max_chars=6000)
        text, links = html_to_text(r.text, str(r.url))
        sig = _signals(r.text, text, str(r.url))
        soup = BeautifulSoup(r.text, "lxml")
        cands: dict[str, str] = {}
        for a in soup.find_all("a", href=True):
            href = urljoin(str(r.url), a["href"])
            label = a.get_text(" ", strip=True)[:60]
            if href.startswith("http") and (_AGENDA_WORDS.search(href) or _AGENDA_WORDS.search(label)):
                cands.setdefault(href, label)
        link_lines = "\n".join(f"- {label!r} -> {href}" for href, label in list(cands.items())[:40])
        return (
            f"{head}\nSIGNALS: {json.dumps(sig, ensure_ascii=False)}\n"
            f"TEXT EXCERPT (first 3500 chars):\n{text[:3500]}\n\nAGENDA-LIKE LINKS:\n{link_lines or '(none)'}"
        )

    @agent.tool
    async def fetch_json(ctx: RunContext[DiscoverDeps], url: str) -> str:
        """Fetch a JSON endpoint and return its structure with sample items (keys, list sizes, sample values)."""
        try:
            r = await ctx.deps.client.get(url, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            return f"ERROR fetching {url}: {exc}"
        head = f"status={r.status_code} content_type={r.headers.get('content-type', '')} bytes={len(r.content)}"
        try:
            json.loads(r.text)
        except json.JSONDecodeError:
            return head + "\nNOT JSON. First 500 chars:\n" + r.text[:500]
        return head + "\n" + summarize_json(r.text, max_chars=8000, sample_items=2)

    @agent.tool
    async def find_api_routes(ctx: RunContext[DiscoverDeps], url: str) -> str:
        """Scan a page's inline and same-origin JavaScript for API routes (/api/..., /wp-json/..., graphql).
        Returns each route with surrounding code so query parameters can be inferred."""
        try:
            r = await ctx.deps.client.get(url)
        except httpx.HTTPError as exc:
            return f"ERROR fetching {url}: {exc}"
        soup = BeautifulSoup(r.text, "lxml")
        origin = f"{urlparse(str(r.url)).scheme}://{urlparse(str(r.url)).netloc}"
        sources: list[tuple[str, str]] = [("inline", " ".join(s.string or "" for s in soup.find_all("script") if not s.get("src")))]
        srcs = [urljoin(str(r.url), s["src"]) for s in soup.find_all("script", src=True)]
        srcs = [s for s in srcs if s.startswith(origin)][:12]
        for s in srcs:
            try:
                js = (await ctx.deps.client.get(s)).text
                sources.append((s, js))
                # Follow one level of chunk imports for Nuxt/Vite bundles.
                for chunk in sorted(set(re.findall(r"""["']\./([\w-]+\.js)["']""", js)))[:60]:
                    curl = urljoin(s, chunk)
                    try:
                        cjs = (await ctx.deps.client.get(curl)).text
                    except httpx.HTTPError:
                        continue
                    if _ROUTE_RE.search(cjs) or "evenement" in cjs or "event" in cjs:
                        sources.append((curl, cjs))
            except httpx.HTTPError:
                continue
        found: dict[str, str] = {}
        for name, code in sources:
            for m in _ROUTE_RE.finditer(code):
                route = m.group(1)
                if route in found:
                    continue
                a, b = max(0, m.start() - 220), min(len(code), m.end() + 220)
                found[route] = f"[{name.rsplit('/', 1)[-1]}] ...{code[a:b]}..."
                if len(found) >= 40:
                    break
        payload = soup.find("script", id="__NUXT_DATA__")
        extra = ""
        if payload and payload.string:
            api_strings = sorted({s for s in re.findall(r'"(/api/[^"]{2,120})"', payload.string)})[:20]
            extra = "\n__NUXT_DATA__ api strings: " + ", ".join(api_strings)
        if not found:
            return f"No API routes found in {len(sources)} scripts of {r.url}." + extra
        lines = [f"- {route}\n    {ctx_.replace(chr(10), ' ')}" for route, ctx_ in found.items()]
        return f"Routes found (origin {origin}). Nuxt fronts usually proxy /wp-json/<ns>/v1/... as /api/<name>/...:\n" + "\n".join(lines) + extra

    @agent.tool
    async def probe_common_endpoints(ctx: RunContext[DiscoverDeps], base_url: str) -> str:
        """Try standard event endpoints on a site (WordPress REST, The Events Calendar, common agenda paths).
        Returns status/type/size and event signals for each."""
        o = urlparse(base_url)
        base = f"{o.scheme}://{o.netloc}"
        paths = [
            "/wp-json/tribe/events/v1/events?per_page=50",
            "/wp-json/wp/v2/types",
            "/wp-json/wp/v2/posts?per_page=5",
            "/agenda", "/agenda/", "/programme", "/programmation", "/evenements", "/events", "/calendrier", "/actualites",
        ]
        out = []
        for p in paths:
            u = base + p
            try:
                r = await ctx.deps.client.get(u)
            except httpx.HTTPError as exc:
                out.append(f"- {u}: ERROR {type(exc).__name__}")
                continue
            ctype = r.headers.get("content-type", "")
            line = f"- {u}: {r.status_code} {ctype.split(';')[0]} {len(r.content)}b"
            if r.status_code < 400:
                if "json" in ctype:
                    try:
                        j = json.loads(r.text)
                        if isinstance(j, dict) and "events" in j:
                            line += f" tribe_events={len(j['events'])} total={j.get('total')}"
                        elif isinstance(j, dict):
                            line += f" keys={list(j)[:12]}"
                        elif isinstance(j, list):
                            line += f" list[{len(j)}]"
                    except json.JSONDecodeError:
                        pass
                else:
                    text, _ = html_to_text(r.text, u)
                    line += f" date_like={len(_DATE_HINT.findall(text))}"
            out.append(line)
        return "\n".join(out)

    return agent


async def _web_search(deps: DiscoverDeps, query: str) -> str:
    """Web search through OpenRouter's web plugin (no extra API key), DuckDuckGo HTML as fallback."""
    try:
        r = await deps.client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {deps.settings.openrouter_api_key}"},
            json={
                "model": deps.settings.search_model,
                "plugins": [{"id": "web", "max_results": 8}],
                "messages": [{"role": "user", "content": f"Web search for: {query}\nList the results."}],
                "max_tokens": 400,
            },
            timeout=60,
        )
        if r.status_code == 200:
            msg = r.json()["choices"][0]["message"]
            lines = []
            for a in msg.get("annotations") or []:
                c = a.get("url_citation") or {}
                if c.get("url"):
                    lines.append(f"{(c.get('title') or '')[:80]} | {c['url']} | {(c.get('content') or '')[:160].replace(chr(10), ' ')}")
            if lines:
                return "\n".join(dict.fromkeys(lines))
            log.warning("web plugin returned no annotations; falling back to DuckDuckGo")
        else:
            log.warning("web plugin failed %s: %s", r.status_code, r.text[:200])
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("web plugin error: %s", exc)
    return await _ddg(deps.client, query)


async def _ddg(client: httpx.AsyncClient, query: str) -> str:
    r = await client.get("https://html.duckduckgo.com/html/", params={"q": query}, headers={"User-Agent": USER_AGENT})
    soup = BeautifulSoup(r.text, "lxml")
    lines = []
    for res in soup.select(".result")[:8]:
        a = res.select_one("a.result__a")
        snip = res.select_one(".result__snippet")
        if a and a.get("href"):
            href = a["href"]
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                from urllib.parse import unquote

                href = unquote(m.group(1))
            lines.append(f"{a.get_text(' ', strip=True)[:80]} | {href} | {(snip.get_text(' ', strip=True) if snip else '')[:160]}")
    return "\n".join(lines) or "(no results)"


async def discover_source(
    agent: Agent[DiscoverDeps, DiscoveredSource], settings: Settings, venue_name: str, website: str | None, today: date,
):
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"}, follow_redirects=True, timeout=40,
    ) as client:
        deps = DiscoverDeps(client=client, settings=settings, today=today, venue_name=venue_name, website=website)
        prompt = f"Today is {today.isoformat()}. Venue: {venue_name} (Marseille, France)."
        if website:
            prompt += f" Known website: {website}"
        return await agent.run(prompt, deps=deps, usage_limits=UsageLimits(request_limit=30))
