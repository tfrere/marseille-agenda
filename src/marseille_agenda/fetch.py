"""Fetch source documents and turn them into plain text the LLM can be grounded against."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

USER_AGENT = "Mozilla/5.0 (compatible; marseille-agenda/0.1; +https://github.com/tfrere/marseille-agenda)"

# Elements whose text never describes an event and only adds noise for the model.
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_DROP_TAGS = {"script", "style", "noscript", "svg", "iframe", "select", "option", "form", "button", "template"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "nav", "aside", "main",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "table", "tr", "td", "th",
    "br", "hr", "blockquote", "pre", "figure", "figcaption", "dd", "dt", "dl",
}


@dataclass
class SourceDocument:
    url: str
    kind: str
    text: str
    """Plain text derived from the source; the grounding checks run against this."""
    raw: str
    """Raw body as fetched (HTML or JSON)."""
    links: list[str] = field(default_factory=list)
    """Absolute URLs found in the document (for URL grounding)."""
    cache: dict = field(default_factory=dict, repr=False, compare=False)
    """Per-document memo (normalized text etc.)."""

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def html_to_text(html: str, base_url: str) -> tuple[str, list[str]]:
    """Convert HTML to readable text, keeping link targets inline as `text (url)`.

    Keeping the URLs inside the text is what lets the pipeline demand that any URL the
    model outputs literally appears in the source.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()

    links: list[str] = []
    parts: list[str] = []

    # Explicit stack: real-world pages nest deeper than Python's recursion limit.
    stack: list[tuple[object, str | None]] = [(soup.body or soup, None)]
    while stack:
        node, closing = stack.pop()
        if closing is not None:
            parts.append(closing)
            continue
        if isinstance(node, NavigableString):
            if node.parent and node.parent.name in ("[document]", "html", "head"):
                continue
            if type(node).__name__ in ("Comment", "Doctype", "Declaration", "ProcessingInstruction"):
                continue
            parts.append(str(node))
            continue
        if not isinstance(node, Tag):
            continue
        name = node.name.lower()
        after: str | None = None
        if name in _BLOCK_TAGS:
            parts.append("\n")
            after = "\n"
        if name == "a" and node.get("href"):
            href = urljoin(base_url, str(node["href"]).strip())
            if href.startswith("http"):
                links.append(href)
                after = f" ({href})"
        elif name == "time" and node.get("datetime"):
            parts.append(f" [{node['datetime']}] ")
        elif isinstance(dd := node.get("data-date"), str) and _ISO_DAY.search(dd):
            # Agenda cards grouped by day often carry the machine date here while their text
            # only shows a time: surface it so readers, schemas and the verifier all see it.
            parts.append(f" [{dd.strip()}] ")
        if after is not None:
            stack.append((node, after))
        stack.extend((child, None) for child in reversed(list(node.children)))
    text = "".join(parts)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), links


_KEEP_ATTRS = {"class", "id", "href", "datetime", "content", "itemprop", "itemtype", "data-date", "data-start", "data-time", "title", "rel", "type"}


def condense_html(html: str, max_chars: int = 70_000) -> str:
    """Strip an HTML page down to its structural skeleton for the schema generator.

    Keeps tags, classes/ids/hrefs/datetime attributes and text; drops scripts, styles,
    inline SVG, comments, images and every other attribute. Long text nodes are clipped.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_DROP_TAGS | {"img", "picture", "source", "video", "audio", "link", "meta"})):
        tag.decompose()
    for node in soup.find_all(string=True):
        if type(node).__name__ in ("Comment", "Doctype", "Declaration", "ProcessingInstruction"):
            node.extract()
        elif len(node) > 400:
            node.replace_with(node[:400] + " [...]")
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr not in _KEEP_ATTRS:
                del tag.attrs[attr]
    out = str(soup.body or soup)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"> <", "><", out)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n<!-- [truncated] -->"
    return out


def _clip_json(data, sample_items: int, deep_items: int, markers: bool, max_str: int = 160):
    def clip(v, depth: int):
        if isinstance(v, dict):
            return {k: clip(x, depth + 1) for k, x in list(v.items())[:60]}
        if isinstance(v, list):
            n = sample_items if depth <= 2 else deep_items
            out = [clip(x, depth + 1) for x in v[:n]]
            if markers and len(v) > n:
                out.append(f"... ({len(v) - n} more items)")
            return out
        if isinstance(v, str) and len(v) > max_str:
            return v[:max_str] + (f"... [{len(v)} chars]" if markers else "")
        return v

    return clip(data, 0)


def summarize_json(raw: str, max_chars: int = 40_000, sample_items: int = 2) -> str:
    """Compact, structure-revealing rendering of a JSON document for the LLM (not valid JSON if truncated)."""
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:max_chars]
    text = json.dumps(_clip_json(data, sample_items, 3, markers=True), ensure_ascii=False, indent=1)
    return text[:max_chars] + ("\n... [truncated]" if len(text) > max_chars else "")


def trim_json(raw: str, max_chars: int = 100_000, sample_items: int = 8) -> str:
    """Valid, smaller JSON with the same structure: lists shortened, long strings clipped.

    Shrinks the number of kept items until the result fits `max_chars`, so both the LLM
    reader and the schema engine can consume exactly the same document.
    """
    import json

    data = json.loads(raw)
    for n in (sample_items, 6, 4, 3, 2, 1):
        text = json.dumps(_clip_json(data, n, max(3, n), markers=False, max_str=200), ensure_ascii=False)
        if len(text) <= max_chars:
            return text
    return text


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"},
        follow_redirects=True,
        timeout=httpx.Timeout(40.0),
    )


def fetch_document(url: str, kind: str, client: httpx.Client | None = None) -> SourceDocument:
    own = client is None
    client = client or make_client()
    try:
        resp = client.get(url)
        resp.raise_for_status()
        body = resp.text
    finally:
        if own:
            client.close()
    return document_from_body(url, kind, body)


def document_from_body(url: str, kind: str, body: str) -> SourceDocument:
    if kind == "html":
        text, links = html_to_text(body, url)
        return SourceDocument(url=url, kind=kind, text=text, raw=body, links=links)
    # Structured kinds: text is the raw body; adapters do their own parsing.
    return SourceDocument(url=url, kind=kind, text=body, raw=body, links=[])
