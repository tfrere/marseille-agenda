"""Source discovery agent: from a venue name, find the URL that lists its upcoming events.

Design for cost control: the heavy lifting (crawling the site, collecting agenda-like links,
scanning JavaScript for API routes, probing standard endpoints) is done by ONE deterministic
coarse-grained tool, `scan_site`, which returns a compact digest. The LLM only has to pick a
candidate and verify it, within a hard request budget. Every tool output is capped and every
tool call is logged with its size, so a runaway run is visible and bounded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from .config import Settings
from .dates import DateParseError, parse_date_text
from .extraction_schema import DiscoveredSource
from .fetch import USER_AGENT, html_to_text, summarize_json

log = logging.getLogger(__name__)

# Hard cap on model requests per discovery, and the step after which tools refuse to work so the
# model has to answer with what it has. Each request re-sends the whole context, so cost grows
# roughly quadratically with the number of steps: keep them few.
MAX_REQUESTS = 12
SOFT_STOP_STEP = 8
MAX_DIGEST_CHARS = 14_000

INSTRUCTIONS = """\
You locate the best source of UPCOMING public events for one venue or organisation in Marseille.

Tools:
- `scan_site(url)`: ONE call crawls a site: home page, agenda-like pages with their date counts and
  excerpts, hidden API routes found in its JavaScript, standard endpoints. Always prefer it over
  many small calls. When a website is known, its scan is already in the prompt (PRE-SCAN).
- `web_search(query)`: only when no website is known or the site has no agenda. Ignore aggregators
  (Facebook, allevents, sortiraparis...) unless the venue has no site of its own; local
  independent agendas (e.g. mars-infos.org, radar.squat.net) are acceptable for venues without a site.
- `fetch_page`, `fetch_json`, `find_api_routes`, `probe_common_endpoints`: targeted follow-ups.

Rules:
1. Prefer, in order: JSON endpoint listing upcoming events > server-rendered HTML listing page.
   ICS/RSS feeds are NOT supported: do not return them.
2. A page whose text has few date-like strings but a JS payload signal loads events client-side:
   use the API routes from the scan (try query params seen in the code, e.g. upcoming=1, perPage=100)
   and `fetch_json` the best one. Return the URL with the LARGEST page size the API accepts
   (e.g. perPage=100, per_page=100) and no `page` parameter, so one fetch covers the whole agenda.
3. VERIFY before answering: the returned URL must have been fetched (by a scan or by you) and show
   several upcoming events (dates on or after today) with titles. Put up to 5 titles in `sample_titles`.
4. Budget: at most 6 tool calls, one at a time (no parallel batches). If a tool answers "BUDGET
   EXHAUSTED", stop and answer immediately with your best candidate and a lower confidence.
5. Social accounts: fill `instagram` (handle) and `facebook` (page slug) when the scan's SOCIAL LINKS
   or a search result shows the venue's own accounts. Never spend a tool call just for them; leave
   them null when unsure. They feed a separate flyer-reading pipeline, not the source URL.
"""

_AGENDA_WORDS = re.compile(r"agenda|program|event|évén|evene|calend|spectacle|concert|saison|billet|rendez|rencontre|soir|expo", re.I)
_DATE_HINT = re.compile(
    r"\b\d{1,2}\s+(janv|févr|fevr|mars|avr|mai|juin|juil|août|aout|sept|oct|nov|déc|dec)\w*\.?(\s+\d{4})?|\b20\d{2}-\d{2}-\d{2}",
    re.I,
)
_ROUTE_RE = re.compile(r"""["'`](/?(?:api|wp-json|graphql|rest|feed|data)/[^"'`\s<>]{1,160})["'`]""")
_CHUNK_RE = re.compile(r"""["']\./([\w-]+\.js)["']""")
MAX_CHUNKS = 200


@dataclass
class DiscoverDeps:
    client: httpx.AsyncClient
    settings: Settings
    today: date
    venue_name: str
    website: str | None


# --------------------------------------------------------------------------- plain helpers


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


def _agenda_links(soup: BeautifulSoup, base: str, same_origin: bool, limit: int) -> dict[str, str]:
    """Agenda-looking links, best first (keyword in both label and href scores higher)."""
    origin = urlparse(base).netloc
    scored: dict[str, tuple[int, str]] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(base, a["href"]).split("#")[0]
        if not href.startswith("http") or (same_origin and urlparse(href).netloc != origin):
            continue
        label = a.get_text(" ", strip=True)[:60]
        score = bool(_AGENDA_WORDS.search(href)) + bool(_AGENDA_WORDS.search(label))
        if score and (href not in scored or scored[href][0] < score):
            scored[href] = (score, label)
    best = sorted(scored.items(), key=lambda kv: -kv[1][0])[:limit]
    return {href: label for href, (_, label) in best}


def _date_snippets(text: str, today: date, *, limit: int = 10, window: int = 140) -> str:
    """Text windows around date-like strings, each tagged upcoming/past, so the agent can verify
    a listing without reading the whole page."""
    out: list[str] = []
    last_end = -1
    for m in _DATE_HINT.finditer(text):
        if m.start() < last_end:
            continue
        a, b = max(0, m.start() - window), min(len(text), m.end() + window)
        last_end = b
        try:
            parsed = parse_date_text(text[max(0, m.start() - 12):m.end() + 30], today)
            tag = "upcoming" if (parsed.end or parsed.start) >= today else "past"
            tag += f" {parsed.start.isoformat()}"
        except DateParseError:
            tag = "unparsed"
        out.append(f"[{tag}] ...{' '.join(text[a:b].split())}...")
        if len(out) >= limit:
            break
    return "\n".join(out) or "(no date-like strings)"


async def _page_report(client: httpx.AsyncClient, url: str, today: date, *, excerpt: int, max_links: int) -> str:
    try:
        r = await client.get(url)
    except httpx.HTTPError as exc:
        return f"ERROR fetching {url}: {exc}"
    ctype = r.headers.get("content-type", "")
    head = f"status={r.status_code} final_url={r.url} content_type={ctype.split(';')[0]} bytes={len(r.content)}"
    if r.status_code >= 400:
        return head
    if "json" in ctype or r.text.lstrip().startswith(("{", "[")):
        return head + "\nJSON SUMMARY:\n" + summarize_json(r.text, max_chars=4000)
    text, _ = html_to_text(r.text, str(r.url))
    sig = _signals(r.text, text, str(r.url))
    links = _agenda_links(BeautifulSoup(r.text, "lxml"), str(r.url), same_origin=False, limit=max_links)
    link_lines = "\n".join(f"- {label!r} -> {href}" for href, label in links.items())
    return (
        f"{head}\nSIGNALS: {json.dumps(sig, ensure_ascii=False)}\n"
        f"DATE SNIPPETS:\n{_date_snippets(text, today)}\n\n"
        f"TEXT EXCERPT (first {excerpt} chars):\n{text[:excerpt]}\n\nAGENDA-LIKE LINKS:\n{link_lines or '(none)'}"
    )


async def _json_report(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, headers={"Accept": "application/json"})
    except httpx.HTTPError as exc:
        return f"ERROR fetching {url}: {exc}"
    head = f"status={r.status_code} content_type={r.headers.get('content-type', '').split(';')[0]} bytes={len(r.content)}"
    try:
        json.loads(r.text)
    except json.JSONDecodeError:
        return head + "\nNOT JSON. First 400 chars:\n" + r.text[:400]
    return head + "\n" + summarize_json(r.text, max_chars=5000, sample_items=2)


_EVENT_ROUTE = re.compile(r"even|event|agenda|program|calend|spectacle|session|manifestation", re.I)


async def _collect_routes(client: httpx.AsyncClient, url: str, *, ctx_chars: int = 120) -> tuple[dict[str, str], str]:
    """API routes referenced by a page's JavaScript -> {route: code context}, plus Nuxt payload api strings."""
    try:
        r = await client.get(url)
    except httpx.HTTPError as exc:
        return {}, f"ERROR fetching {url}: {exc}"
    soup = BeautifulSoup(r.text, "lxml")
    origin = f"{urlparse(str(r.url)).scheme}://{urlparse(str(r.url)).netloc}"
    sources: list[tuple[str, str]] = [("inline", " ".join(s.string or "" for s in soup.find_all("script") if not s.get("src")))]
    srcs = [urljoin(str(r.url), s["src"]) for s in soup.find_all("script", src=True)]
    srcs += [urljoin(str(r.url), l["href"]) for l in soup.find_all("link", rel="modulepreload", href=True)]
    srcs = list(dict.fromkeys(s for s in srcs if s.startswith(origin)))[:20]
    sem = asyncio.Semaphore(8)

    async def get_js(u: str) -> tuple[str, str]:
        async with sem:
            try:
                return u, (await client.get(u)).text
            except httpx.HTTPError:
                return u, ""

    entries = await asyncio.gather(*(get_js(s) for s in srcs))
    sources += [e for e in entries if e[1]]
    # Follow one level of chunk imports for Nuxt/Vite bundles (the events API is usually referenced
    # by the agenda page component, a lazily imported chunk).
    chunks: dict[str, None] = {}
    for s, js in entries:
        for chunk in _CHUNK_RE.findall(js):
            chunks.setdefault(urljoin(s, chunk))
    seen = {s for s, _ in entries}
    todo = [c for c in chunks if c not in seen][:MAX_CHUNKS]
    sources += [e for e in await asyncio.gather(*(get_js(c) for c in todo)) if e[1] and _ROUTE_RE.search(e[1])]
    found: dict[str, str] = {}
    for name, code in sources:
        for m in _ROUTE_RE.finditer(code):
            route = m.group(1)
            if route in found:
                continue
            a, b = max(0, m.start() - ctx_chars), min(len(code), m.end() + ctx_chars)
            found[route] = f"[{name.rsplit('/', 1)[-1]}] ...{code[a:b]}...".replace("\n", " ")
    payload = soup.find("script", id="__NUXT_DATA__")
    extra = ""
    if payload and payload.string:
        api_strings = sorted({s for s in re.findall(r'"(/api/[^"]{2,120})"', payload.string)})[:20]
        if api_strings:
            extra = "__NUXT_DATA__ api strings: " + ", ".join(api_strings)
    return found, extra


def _render_routes(origin: str, found: dict[str, str], extra: str, *, max_routes: int = 30) -> str:
    """Event-looking routes first and with code context; the rest as bare paths."""
    if not found:
        return "No API routes found." + (f"\n{extra}" if extra else "")
    eventish = [r for r in found if _EVENT_ROUTE.search(r)]
    others = [r for r in found if r not in eventish]
    lines = [f"- {r}\n    {found[r]}" for r in eventish[:max_routes]]
    rest = others[: max(0, max_routes - len(lines))]
    if rest:
        lines.append("other routes (no context): " + ", ".join(rest))
    return (
        f"Routes found (origin {origin}). Nuxt fronts usually proxy /wp-json/<ns>/v1/... as /api/<name>/...:\n"
        + "\n".join(lines) + (f"\n{extra}" if extra else "")
    )


_WP_PREFIX = re.compile(r"^/?wp-json/[\w-]+/v\d+")
_BASE_ROUTE = re.compile(r"^/api/[\w-]+/?$")
MAX_ROUTE_PROBES = 12


def _event_route_candidates(found: dict[str, str]) -> list[str]:
    """Concrete (no template placeholder) listing routes that look event-related, plus the same
    routes rewritten through every base prefix the front-end defines (Nuxt/Next proxies usually
    map /wp-json/<ns>/vN/... to /api/<name>/...)."""
    bases = sorted({r.rstrip("/") for r in found if _BASE_ROUTE.match(r)})
    out: list[str] = []
    for route, ctx in found.items():
        if "${" in route or "{" in route or route.endswith(".ics") or "?id=" in route:
            continue
        if not _EVENT_ROUTE.search(route) or re.search(r"presse|press|terms/|/page/", route):
            continue
        params = ""
        if re.search(r"upcoming", ctx):
            params = "?upcoming=1&perPage=100"
        norm = "/" + route.lstrip("/")
        variants = [norm]
        if _WP_PREFIX.match(norm):
            rest = _WP_PREFIX.sub("", norm)
            variants += [b + rest for b in bases]
        for v in variants:
            u = v + params
            if u not in out:
                out.append(u)
    # Variants that carry the query parameters seen in the code are the most promising.
    out.sort(key=lambda u: "?" not in u)
    return out[:MAX_ROUTE_PROBES]


async def _probe_routes(client: httpx.AsyncClient, origin: str, found: dict[str, str]) -> str:
    """Fetch each candidate event route and report what comes back (status, JSON shape)."""
    cands = _event_route_candidates(found)
    if not cands:
        return ""
    sem = asyncio.Semaphore(4)

    async def one(path: str) -> str:
        url = origin + path
        async with sem:
            try:
                r = await client.get(url, headers={"Accept": "application/json"}, timeout=30)
            except httpx.HTTPError as exc:
                return f"- {url}: ERROR {type(exc).__name__}"
        line = f"- {url}: {r.status_code} {r.headers.get('content-type', '').split(';')[0]} {len(r.content)}b"
        if r.status_code < 400:
            try:
                j = json.loads(r.text)
            except json.JSONDecodeError:
                return line + " (not JSON)"
            if isinstance(j, list):
                line += f" -> JSON list[{len(j)}]"
            elif isinstance(j, dict):
                lists = {k: len(v) for k, v in j.items() if isinstance(v, list)}
                line += f" -> JSON keys={list(j)[:10]} lists={lists}"
        return line

    lines = await asyncio.gather(*(one(p) for p in cands))
    return "EVENT ROUTE PROBES (fetched now, use a 200 JSON one with a non-empty list):\n" + "\n".join(lines)


async def _api_routes(client: httpx.AsyncClient, url: str) -> str:
    found, extra = await _collect_routes(client, url)
    o = urlparse(url)
    origin = f"{o.scheme}://{o.netloc}"
    probes = await _probe_routes(client, origin, found)
    return _render_routes(origin, found, extra) + (f"\n\n{probes}" if probes else "")


_PROBE_PATHS = [
    "/wp-json/tribe/events/v1/events?per_page=50",
    "/wp-json/wp/v2/types",
    "/agenda", "/programme", "/programmation", "/evenements", "/events", "/calendrier", "/actualites",
]


async def _probe(client: httpx.AsyncClient, base_url: str) -> tuple[str, list[str]]:
    """Report on standard endpoints, plus the HTML pages among them that answered 200."""
    o = urlparse(base_url)
    base = f"{o.scheme}://{o.netloc}"
    out = []
    html_ok: list[str] = []
    for p in _PROBE_PATHS:
        u = base + p
        try:
            r = await client.get(u)
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
                html_ok.append(str(r.url))
        out.append(line)
    return "\n".join(out), html_ok


_SOCIAL_RE = re.compile(
    r"https?://(?:www\.)?(instagram\.com|facebook\.com)/([A-Za-z0-9_.\-]+)/?(?:\?[^\"'\s]*)?$", re.I
)
_SOCIAL_SKIP = {"sharer", "sharer.php", "share", "p", "reel", "explore", "events", "hashtag", "groups", "pages", "profile.php", "intent"}


def _social_links(soup: BeautifulSoup) -> dict[str, str]:
    """Instagram handle / Facebook page slug linked from a page (typically the footer)."""
    out: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        m = _SOCIAL_RE.match(a["href"].strip())
        if not m:
            continue
        site, slug = m.group(1).lower(), m.group(2)
        if slug.lower() in _SOCIAL_SKIP:
            continue
        key = "instagram" if site.startswith("instagram") else "facebook"
        out.setdefault(key, slug)
    return out


async def scan_site(client: httpx.AsyncClient, url: str, today: date) -> str:
    """One-shot deterministic crawl of a venue site, rendered as a compact digest."""
    try:
        r = await client.get(url)
    except httpx.HTTPError as exc:
        return f"ERROR fetching {url}: {exc}"
    if r.status_code >= 400:
        return f"status={r.status_code} for {url}"
    home_text, _ = html_to_text(r.text, str(r.url))
    home_sig = _signals(r.text, home_text, str(r.url))
    soup = BeautifulSoup(r.text, "lxml")
    links = _agenda_links(soup, str(r.url), same_origin=True, limit=5)
    social = _social_links(soup)

    parts = [
        f"HOME {r.url}\nSIGNALS: {json.dumps(home_sig, ensure_ascii=False)}\n"
        f"SOCIAL LINKS: {json.dumps(social, ensure_ascii=False) if social else 'none'}\n"
        f"DATE SNIPPETS:\n{_date_snippets(home_text, today, limit=6)}\nEXCERPT: {home_text[:800]}"
    ]
    js_pages: list[str] = [str(r.url)] if home_sig["js_payload"] else []
    for href, label in links.items():
        try:
            pr = await client.get(href)
        except httpx.HTTPError as exc:
            parts.append(f"PAGE {label!r} {href}: ERROR {type(exc).__name__}")
            continue
        if pr.status_code >= 400 or "html" not in pr.headers.get("content-type", ""):
            parts.append(f"PAGE {label!r} {href}: status={pr.status_code}")
            continue
        text, _ = html_to_text(pr.text, str(pr.url))
        sig = _signals(pr.text, text, str(pr.url))
        if sig["js_payload"]:
            js_pages.append(str(pr.url))
        sub = _agenda_links(BeautifulSoup(pr.text, "lxml"), str(pr.url), same_origin=True, limit=8)
        sub_lines = "\n".join(f"  - {lb!r} -> {h}" for h, lb in sub.items() if h not in links)
        parts.append(
            f"PAGE {label!r} {pr.url}\nSIGNALS: {json.dumps(sig, ensure_ascii=False)}\n"
            f"DATE SNIPPETS:\n{_date_snippets(text, today, limit=6)}\nEXCERPT: {text[:500]}\n"
            + (f"MORE LINKS:\n{sub_lines}" if sub_lines else "")
        )
    probe_report, probe_html = await _probe(client, str(r.url))
    parts.append("STANDARD ENDPOINTS:\n" + probe_report)
    if js_pages:
        # Client-side rendering: the events API is referenced by the JS chunks of the agenda-like
        # pages, so scan the home page plus the standard pages that answered 200.
        found: dict[str, str] = {}
        extra = ""
        for page in dict.fromkeys(js_pages[:2] + probe_html[:3]):
            f, e = await _collect_routes(client, page)
            found.update({k: v for k, v in f.items() if k not in found})
            extra = extra or e
        o = urlparse(str(r.url))
        origin = f"{o.scheme}://{o.netloc}"
        parts.append("API ROUTES (client-side rendering detected):\n" + _render_routes(origin, found, extra))
        probes = await _probe_routes(client, origin, found)
        if probes:
            parts.append(probes)
    digest = "\n\n".join(parts)
    if len(digest) > MAX_DIGEST_CHARS:
        digest = digest[:MAX_DIGEST_CHARS] + "\n... [digest truncated]"
    return digest


# --------------------------------------------------------------------------- agent


def _budget(ctx: RunContext[DiscoverDeps], tool: str, arg: str) -> str | None:
    log.info("discover[%s] step %d %s(%s)", ctx.deps.venue_name, ctx.run_step, tool, arg[:100])
    if ctx.run_step > SOFT_STOP_STEP:
        return "BUDGET EXHAUSTED: stop exploring and return your final answer now (lower the confidence if unverified)."
    return None


def _done(ctx: RunContext[DiscoverDeps], tool: str, out: str) -> str:
    log.info("discover[%s] step %d %s -> %d chars", ctx.deps.venue_name, ctx.run_step, tool, len(out))
    return out


def build_discoverer(model: Model | str) -> Agent[DiscoverDeps, DiscoveredSource]:
    agent: Agent[DiscoverDeps, DiscoveredSource] = Agent(
        model, deps_type=DiscoverDeps, output_type=DiscoveredSource, instructions=INSTRUCTIONS, retries=2, name="discoverer",
    )

    @agent.tool
    async def web_search(ctx: RunContext[DiscoverDeps], query: str) -> str:
        """Search the web. Returns up to 8 results as 'title | url | snippet' lines."""
        return _budget(ctx, "web_search", query) or _done(ctx, "web_search", await _web_search(ctx.deps, query))

    @agent.tool(name="scan_site")
    async def scan_site_tool(ctx: RunContext[DiscoverDeps], url: str) -> str:
        """Crawl a site in one call: home page signals, agenda-like pages (date counts, excerpts, links),
        API routes found in JavaScript when rendering is client-side, and standard endpoint probes."""
        return _budget(ctx, "scan_site", url) or _done(ctx, "scan_site", await scan_site(ctx.deps.client, url, ctx.deps.today))

    @agent.tool
    async def fetch_page(ctx: RunContext[DiscoverDeps], url: str) -> str:
        """Fetch one URL: status, structural signals, text excerpt and agenda-related links (JSON summary if JSON)."""
        return _budget(ctx, "fetch_page", url) or _done(
            ctx, "fetch_page", await _page_report(ctx.deps.client, url, ctx.deps.today, excerpt=1500, max_links=25)
        )

    @agent.tool
    async def fetch_json(ctx: RunContext[DiscoverDeps], url: str) -> str:
        """Fetch a JSON endpoint and return its structure with sample items (keys, list sizes, sample values)."""
        return _budget(ctx, "fetch_json", url) or _done(ctx, "fetch_json", await _json_report(ctx.deps.client, url))

    @agent.tool
    async def find_api_routes(ctx: RunContext[DiscoverDeps], url: str) -> str:
        """Scan a page's inline and same-origin JavaScript for API routes (/api/..., /wp-json/..., graphql),
        each with a little surrounding code so query parameters can be inferred."""
        return _budget(ctx, "find_api_routes", url) or _done(ctx, "find_api_routes", await _api_routes(ctx.deps.client, url))

    @agent.tool
    async def probe_common_endpoints(ctx: RunContext[DiscoverDeps], base_url: str) -> str:
        """Try standard event endpoints on a site (WordPress REST, The Events Calendar, common agenda paths)."""
        return _budget(ctx, "probe_common_endpoints", base_url) or _done(ctx, "probe", (await _probe(ctx.deps.client, base_url))[0])

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
                href = unquote(m.group(1))
            lines.append(f"{a.get_text(' ', strip=True)[:80]} | {href} | {(snip.get_text(' ', strip=True) if snip else '')[:160]}")
    return "\n".join(lines) or "(no results)"


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"}, follow_redirects=True, timeout=40,
    )


async def discover_source(
    agent: Agent[DiscoverDeps, DiscoveredSource], settings: Settings, venue_name: str, website: str | None, today: date,
):
    async with make_client() as client:
        deps = DiscoverDeps(client=client, settings=settings, today=today, venue_name=venue_name, website=website)
        prompt = f"Today is {today.isoformat()}. Venue: {venue_name} (Marseille, France)."
        if website:
            digest = await scan_site(client, website, today)
            log.info("discover[%s] pre-scan of %s -> %d chars", venue_name, website, len(digest))
            prompt += f" Known website: {website}\n\n=== PRE-SCAN of {website} ===\n{digest}\n=== END PRE-SCAN ==="
        else:
            prompt += " No website is known: search the web first."
        run = await agent.run(prompt, deps=deps, usage_limits=UsageLimits(request_limit=MAX_REQUESTS))
        u = run.usage
        log.info("discover[%s] done: %d requests, %d tool calls, %d input tokens, cost %s",
                 venue_name, u.requests, u.tool_calls, u.input_tokens, u.cost)
        return run
