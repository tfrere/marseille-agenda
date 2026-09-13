"""Deterministic schema induction for HTML listings: no model involved.

An agenda page is a list of repeated cards, each carrying a date, a title and a link. Finding
the card is a counting problem, not a language problem: take every element whose short text
(or `data-date` / `datetime` attribute) is an upcoming date, walk up its ancestors, and look for
the ancestor selector shared by the most of them without nesting. Inside the cards, the title is
the heading or link whose text differs from card to card, the URL is the title's link, the date
is the machine attribute when there is one or the dated element otherwise.

The result is an ordinary `ExtractionSchema`, executed and judged with the same engine and the
same grounding checks as a model-written one (`apply_schema`, `check_event`). When the page is
not a listing (prose, a single event, a client-rendered shell) the induction returns None and
the pipeline falls back to the generator agent.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from bs4 import BeautifulSoup, Tag

from .apply import ApplyResult, apply_schema, first_item_image, group_runs
from .dates import DateParseError, parse_date_text
from .extraction_schema import ExtractionSchema, FieldSpec, HtmlRule
from .fetch import SourceDocument
from .validate import check_event

log = logging.getLogger(__name__)

MIN_ITEMS = 3
MAX_ANCESTORS = 8
MAX_DATE_TEXT = 120
HORIZON_DAYS = 400
_CLASS_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_TIME_RE = re.compile(r"\b\d{1,2}\s*(?:h|:)\s*\d{0,2}\b", re.I)
_TITLE_CLASS = re.compile(r"title|titre|name|nom|heading|headline", re.I)
_LOCATION_CLASS = re.compile(r"lieu|location|place|adresse|address|venue", re.I)
_DROP = {"script", "style", "noscript", "svg", "template", "iframe", "select", "option", "form", "button", "head"}
_DATE_ATTRS = ("data-date", "datetime", "data-start", "data-start-date")


@dataclass
class Induction:
    schema: ExtractionSchema
    item_selector: str
    items: int
    events: int
    """Upcoming events that pass the deterministic grounding checks."""
    coverage: float
    """Share of the upcoming dates found on the page that the schema turns into events."""
    quality: float
    machine_dates: bool
    sample_titles: list[str]

    def describe(self) -> str:
        return (f"{self.items} cards matched by {self.item_selector!r}, {self.events} upcoming events, "
                f"coverage {self.coverage:.2f}, quality {self.quality:.2f}"
                + (", machine-readable dates" if self.machine_dates else ""))


def _clean_soup(doc: SourceDocument) -> BeautifulSoup:
    soup = BeautifulSoup(doc.raw, "lxml")
    for tag in soup.find_all(_DROP):
        tag.decompose()
    return soup


def upcoming_date_count(doc: SourceDocument, today: date) -> int:
    """How many elements of the page carry an upcoming date (text or machine attribute).

    Zero on a page that claims to be an agenda means the cards are injected by JavaScript
    (a client-rendered shell), not that the venue has nothing announced.
    """
    if doc.kind != "html":
        return 0
    return len(_date_nodes(_clean_soup(doc), today)[0])


def induce_schema(doc: SourceDocument, today: date) -> Induction | None:
    if doc.kind != "html":
        return None
    soup = _clean_soup(doc)
    class_count = Counter(c for t in soup.find_all(True) for c in t.get("class", []) if _CLASS_OK.match(c))

    date_nodes, dated_nodes = _date_nodes(soup, today)
    if len(date_nodes) < MIN_ITEMS:
        return None

    candidates = _item_candidates(soup, date_nodes, dated_nodes, class_count)
    best: Induction | None = None
    for sel, dated in candidates[:8]:
        items = soup.select(sel)
        schema = _build_schema(sel, items, date_nodes, class_count)
        if schema is None:
            continue
        ind = _evaluate(schema, sel, doc, today, len(date_nodes), len(items))
        if ind is None:
            continue
        log.debug("induction candidate %r: %s", sel, ind.describe())
        if best is None or (ind.quality, ind.events) > (best.quality, best.events):
            best = ind
    if best is None:
        best = _induce_runs(soup, doc, today, date_nodes, class_count)
    return best


# --------------------------------------------------------------------------- flat listings


def _induce_runs(soup: BeautifulSoup, doc: SourceDocument, today: date, date_nodes: list[Tag], class_count: Counter) -> Induction | None:
    """Flat listings without a per-event wrapper (SPIP, old sites): `<div class="date">` then
    the title block, then the next date. Upcoming date nodes sharing a parent and a selector
    start the items; each item runs until the next start. Fields are derived from the synthetic
    run wrappers exactly like from cards, so the schema is an ordinary one in `run` mode.
    """
    groups: dict[tuple[int, str], list[Tag]] = defaultdict(list)
    for dn in date_nodes:
        parent = dn.parent
        if not isinstance(parent, Tag):
            continue
        sel = _rel_selector(dn, class_count)
        if sel and sel not in ("div", "span", "p", "li"):  # a bare tag would start a run at every paragraph
            groups[(id(parent), sel)].append(dn)
        elif sel in ("h1", "h2", "h3", "h4", "h5", "h6", "dt"):
            groups[(id(parent), sel)].append(dn)
    best: Induction | None = None
    for (_, sel), starts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(starts) < MIN_ITEMS:
            break
        if sum(1 for s in starts if _titled_before_next_start(s)) < 0.8 * len(starts):
            continue
        # Runs are built on a private copy: `group_runs` restructures the tree it works on.
        work = BeautifulSoup(str(soup), "lxml")
        try:
            items = group_runs(work, sel)
        except Exception:  # noqa: BLE001 - selector the parser dislikes
            continue
        work_dates = _date_nodes(work, today)[0]
        schema = _build_schema(sel, items, work_dates, class_count, item_mode="run")
        if schema is None:
            continue
        ind = _evaluate(schema, sel, doc, today, len(date_nodes), len(items))
        if ind is None:
            continue
        log.debug("induction run candidate %r: %s", sel, ind.describe())
        if best is None or (ind.quality, ind.events) > (best.quality, best.events):
            best = ind
    return best


def _titled_before_next_start(start: Tag) -> bool:
    """A heading, link or title-classed element among the siblings following the date node,
    before the next date node: the run has something to call a title."""
    for sib in start.next_siblings:
        if not isinstance(sib, Tag):
            continue
        if sib.name == start.name and sib.get("class") == start.get("class"):
            return False  # next start: no title in this run
        for node in [sib, *sib.find_all(True)]:
            text = node.get_text(" ", strip=True)
            if not (3 <= len(text) <= 160) or _is_dated(text) or _looks_like_time(text):
                continue
            if node.name in _HEADINGS or node.name == "a" or any(_TITLE_CLASS.search(c) for c in node.get("class", [])):
                return True
    return False


# --------------------------------------------------------------------------- date nodes


def _date_nodes(soup: BeautifulSoup, today: date) -> tuple[list[Tag], list[Tag]]:
    """Deepest elements whose own short text, or machine attribute, is a date.

    Returns (upcoming, all): the upcoming ones drive the search, all dated elements (past
    events listed on the same page included) tell how many cards of a candidate are dated.
    """
    dated: list[Tag] = []
    upcoming_ids: set[int] = set()
    horizon = today + timedelta(days=HORIZON_DAYS)
    for node in soup.find_all(True):
        iso = _iso_attr(node)
        if iso:
            dated.append(node)
            if today - timedelta(days=1) <= date.fromisoformat(_ISO_DAY.search(iso).group(0)) <= horizon:
                upcoming_ids.add(id(node))
            continue
        text = node.get_text(" ", strip=True)
        if not text or len(text) > MAX_DATE_TEXT or not re.search(r"\d", text):
            continue
        try:
            parsed = parse_date_text(text, today)
        except (DateParseError, ValueError):
            continue
        dated.append(node)
        if today - timedelta(days=1) <= parsed.start <= horizon:
            upcoming_ids.add(id(node))
    # Drop ancestors that only repeat a descendant's date.
    ids = {id(n) for n in dated}
    deepest = [n for n in dated if not any(id(d) in ids for d in n.find_all(True))]
    return [n for n in deepest if id(n) in upcoming_ids], deepest


def _iso_attr(node: Tag) -> str | None:
    for a in _DATE_ATTRS:
        v = node.get(a)
        if isinstance(v, str) and _ISO_DAY.search(v):
            return v
    return None


# --------------------------------------------------------------------------- item selector


def _selector(node: Tag, class_count: Counter, *, shared: bool = True) -> str | None:
    """`tag.class1.class2`. Card selectors keep only classes shared by MIN_ITEMS elements (a
    `post-1234` class would match one card); container selectors keep every class."""
    classes = sorted(c for c in node.get("class", []) if _CLASS_OK.match(c) and (not shared or class_count[c] >= MIN_ITEMS))
    if classes:
        return node.name + "".join(f".{c}" for c in classes)
    return None


def _path_selector(node: Tag, class_count: Counter) -> str | None:
    """Selector for a node without a class of its own: bare tags up to the nearest classed
    ancestor (`div.css-events-list > div > a`), at most 3 levels."""
    sel = _selector(node, class_count)
    if sel:
        return sel
    path = [node.name]
    anc = node.parent
    for _ in range(3):
        if not isinstance(anc, Tag) or anc.name in ("body", "html", "[document]"):
            return None
        top = _selector(anc, class_count, shared=False)
        if top:
            return " > ".join([top, *reversed(path)])
        path.append(anc.name)
        anc = anc.parent
    return None


def _node_selectors(node: Tag, class_count: Counter) -> list[str]:
    """Every selector worth tallying for a node: each shared class alone (cards differ by
    modifier classes such as `is-youth`), all of them together, or the structural path."""
    shared = sorted(c for c in node.get("class", []) if _CLASS_OK.match(c) and class_count[c] >= MIN_ITEMS)
    if not shared:
        path = _path_selector(node, class_count)
        return [path] if path else []
    out = [f"{node.name}.{c}" for c in shared]
    if len(shared) > 1:
        out.append(node.name + "".join(f".{c}" for c in shared))
    return out


def _item_candidates(soup: BeautifulSoup, upcoming: list[Tag], dated: list[Tag], class_count: Counter) -> list[tuple[str, int]]:
    """Ancestor selectors ranked by the number of distinct upcoming cards they match."""
    tally: dict[str, set[int]] = defaultdict(set)
    for dn in upcoming:
        # A card carrying its own data-date is itself the item; a dated text node never is.
        node, depth = (dn if _iso_attr(dn) else dn.parent), 0
        while isinstance(node, Tag) and node.name not in ("body", "html", "[document]") and depth < MAX_ANCESTORS:
            for sel in _node_selectors(node, class_count):
                tally[sel].add(id(node))
            node, depth = node.parent, depth + 1
    dated_ancestors: set[int] = set()
    for dn in dated:
        dated_ancestors.add(id(dn))  # a card with its own data-date counts as dated
        for p in dn.parents:
            dated_ancestors.add(id(p))

    ranked: list[tuple[str, int]] = []
    for sel, hits in tally.items():
        if len(hits) < MIN_ITEMS:
            continue
        try:
            items = soup.select(sel)
        except Exception:  # noqa: BLE001 - selector the parser dislikes
            continue
        if not items or sum(1 for i in items if id(i) in dated_ancestors) / len(items) < 0.5:
            continue  # most cards must carry a date (past ones count)
        dated = hits
        item_ids = {id(i) for i in items}
        if any(id(p) in item_ids for i in items for p in i.parents):
            continue  # nested matches: too coarse or too fine
        ranked.append((sel, len(dated)))
    # Most dated cards first; among equals prefer the tighter selector.
    ranked.sort(key=lambda x: (-x[1], -x[0].count("."), x[0]))
    return ranked


# --------------------------------------------------------------------------- fields


def _build_schema(sel: str, items: list[Tag], date_nodes: list[Tag], class_count: Counter,
                  item_mode: str = "wrapper") -> ExtractionSchema | None:
    # Grid fillers and empty placeholder cards tell nothing about the layout.
    sample = [i for i in items if i.get_text(strip=True)][:40]
    if len(sample) < MIN_ITEMS:
        return None
    title = _title_field(sample, class_count)
    if title is None:
        return None
    fields: dict[str, FieldSpec] = {"title": title}
    url = _url_field(sample, title)
    if url:
        fields["url"] = url
    fields["date"] = _date_field(sample, date_nodes, class_count)
    if fields["date"].selector == "time" and sum(1 for i in sample if len(i.select("time[datetime]")) == 2) >= 0.5 * len(sample):
        # Two <time> per card is a run (exhibition from X to Y), not two events.
        fields["date"] = FieldSpec(selector="time:first-of-type", attr="datetime")
        fields["end_date"] = FieldSpec(selector="time:last-of-type", attr="datetime")
    time_field = _time_field(sample, class_count, fields["date"])
    if time_field:
        fields["time"] = time_field
    image = _image_field(sample, title)
    if image:
        fields["image"] = image
    location = _location_field(sample, class_count)
    if location:
        fields["location"] = location
    notes = ("Induced deterministically from the repeated dated cards of the page." if item_mode == "wrapper"
             else "Induced deterministically from a flat listing: each item runs from a date node to the next.")
    rule = HtmlRule(item_selector=sel, item_mode=item_mode, fields=fields, notes=notes)
    return ExtractionSchema(rules=[rule], notes="Deterministic induction (no model).")


def _rel_selector(node: Tag, class_count: Counter) -> str:
    classes = sorted(c for c in node.get("class", []) if _CLASS_OK.match(c) and class_count[c] >= MIN_ITEMS)
    return node.name + "".join(f".{c}" for c in classes)


_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _looks_like_time(text: str) -> bool:
    """`18h`, `19:30`, `20h30 - 22h`: a time slot, never a title."""
    return bool(re.fullmatch(r"[\d\s:hH.\-–/etàa]+", text)) and bool(re.search(r"\d", text))


def _is_dated(text: str) -> bool:
    if not re.search(r"\d", text):
        return False
    try:
        parse_date_text(text, date(2000, 1, 1))
        return True
    except DateParseError:
        return False


def _title_field(items: list[Tag], class_count: Counter) -> FieldSpec | None:
    """The heading/link whose text is present in most cards and differs between cards."""
    scores: dict[str, list[str]] = defaultdict(list)
    prior: dict[str, int] = {}
    for item in items:
        seen: set[str] = set()
        for node in item.find_all(True):
            text = node.get_text(" ", strip=True)
            if not (3 <= len(text) <= 160) or _is_dated(text) or _looks_like_time(text):
                continue
            is_heading = node.name in _HEADINGS
            titled = any(_TITLE_CLASS.search(c) for c in node.get("class", []))
            if not (is_heading or titled or node.name in ("a", "strong", "b")):
                continue
            parent = node.parent
            in_heading = node.name == "a" and isinstance(parent, Tag) and parent is not item and parent.name in _HEADINGS
            sel = f"{_rel_selector(parent, class_count)} a" if in_heading else _rel_selector(node, class_count)
            if sel in seen or len(item.select(sel)) != len([n for n in item.select(sel) if n.get_text(strip=True)]):
                continue
            seen.add(sel)
            scores[sel].append(text)
            # A link inside a heading is the canonical title pattern; bare links/bold are weak.
            prior[sel] = max(prior.get(sel, 0), 4 if in_heading else 3 if is_heading or titled else 1)
    best, best_score = None, 0.0
    for sel, texts in scores.items():
        presence = len(texts) / len(items)
        if presence < 0.8:
            continue
        distinct = len(set(texts)) / len(texts)
        score = presence + distinct + 0.2 * prior[sel]
        if score > best_score:
            best, best_score = sel, score
    return FieldSpec(selector=best) if best else None


def _url_field(items: list[Tag], title: FieldSpec) -> FieldSpec | None:
    for sel in (title.selector, f"{title.selector} a", "a"):
        hrefs = []
        for item in items:
            n = item.select_one(sel)
            if n is None:
                continue
            a = n if n.name == "a" else n.find("a", href=True) if sel != "a" else n
            if a is not None and a.has_attr("href") and not str(a["href"]).startswith(("#", "javascript:")):
                hrefs.append(str(a["href"]))
        # Distinct links should roughly track distinct titles (an exhibition listed on
        # every day of its run legitimately repeats its URL).
        titles = {i.select_one(title.selector).get_text(" ", strip=True) for i in items if i.select_one(title.selector)}
        if len(hrefs) >= 0.8 * len(items) and len(set(hrefs)) >= 0.5 * max(1, len(titles)):
            return FieldSpec(selector=sel if sel == "a" or sel.endswith(" a") else sel, attr="href") if sel == title.selector and all(
                (item.select_one(sel) or item).name == "a" for item in items if item.select_one(sel)
            ) else FieldSpec(selector=sel if sel.endswith(" a") or sel == "a" else f"{sel} a", attr="href")
    # The card itself is the link (`<a class="card" href=...>`): no descendant to select.
    if sum(1 for i in items if i.name == "a" and i.has_attr("href")) >= 0.8 * len(items):
        return FieldSpec(selector="", attr="href")
    return None


def _date_field(items: list[Tag], date_nodes: list[Tag], class_count: Counter) -> FieldSpec:
    n = len(items)
    if sum(1 for i in items if _iso_attr(i)) >= 0.8 * n:
        attr_counts = Counter(a for i in items for a in _DATE_ATTRS if i.get(a))
        attr = attr_counts.most_common(1)[0][0]
        return FieldSpec(selector="", attr=attr, regex=r"(\d{4}-\d{2}-\d{2})")
    if sum(1 for i in items if i.select_one("time[datetime]")) >= 0.8 * n:
        return FieldSpec(selector="time", attr="datetime")  # end-of-run companion added by the caller
    # The dated element itself, when it sits at a stable place in the cards. A listing may mix
    # events with plain articles: the selector must cover the dated cards, not every card.
    ids = {id(d) for d in date_nodes}
    counts: Counter = Counter()
    dated_items = 0
    for item in items:
        sels = {_rel_selector(d, class_count) for d in item.find_all(True) if id(d) in ids}
        dated_items += bool(sels)
        counts.update(sels)
    for sel, c in counts.most_common(3):
        if c >= 0.5 * n and c >= 0.9 * dated_items and all(len(i.select(sel)) <= 2 for i in items):
            return FieldSpec(selector=sel)
    return FieldSpec(selector="")  # whole card text; the engine picks the date out of it


def _image_field(items: list[Tag], title: FieldSpec) -> FieldSpec | None:
    """`img[src]` when the cards carry their own visual: most cards have a usable picture and
    the pictures are about as varied as the titles (an exhibition listed on every day of its run
    legitimately repeats its poster; a venue logo on every card does not count as a visual).
    Without the field the engine still falls back to the first picture of each card."""
    urls = [u for u in (first_item_image(i, "https://induction.invalid/") for i in items) if u]
    if len(urls) < 0.8 * len(items):
        return None
    titles = {i.select_one(title.selector).get_text(" ", strip=True) for i in items if i.select_one(title.selector)}
    if len(set(urls)) < 0.5 * max(1, len(titles)):
        return None
    return FieldSpec(selector="img", attr="src")


def _location_field(items: list[Tag], class_count: Counter) -> FieldSpec | None:
    """An element whose class names a place (`lieu`, `location`, `venue`, `adresse`...) with a
    short text in at least 60% of the cards. Aggregators list where each event happens."""
    counts: Counter = Counter()
    for item in items:
        seen: set[str] = set()
        for node in item.find_all(True):
            if not any(_LOCATION_CLASS.search(c) for c in node.get("class", [])):
                continue
            text = node.get_text(" ", strip=True)
            if not (2 <= len(text) <= 160) or _is_dated(text) or _looks_like_time(text):
                continue
            sel = _rel_selector(node, class_count)
            if sel not in seen and len(item.select(sel)) == 1:
                seen.add(sel)
                counts[sel] += 1
    for sel, c in counts.most_common(1):
        if c >= 0.6 * len(items):
            return FieldSpec(selector=sel)
    return None


def _time_field(items: list[Tag], class_count: Counter, date_field: FieldSpec) -> FieldSpec | None:
    if date_field.selector == "" and date_field.attr is None:
        return None  # the card text already carries the time when there is one
    counts: Counter = Counter()
    for item in items:
        for node in item.find_all(True):
            text = node.get_text(" ", strip=True)
            if 2 <= len(text) <= 40 and _TIME_RE.search(text) and not _is_dated(text) and not any(
                _TIME_RE.search(c.get_text(" ", strip=True)) for c in node.find_all(True)
            ):
                counts[_rel_selector(node, class_count)] += 1
                break
    for sel, c in counts.most_common(1):
        if c >= 0.6 * len(items):
            return FieldSpec(selector=sel)
    return None


# --------------------------------------------------------------------------- judgement


def _evaluate(schema: ExtractionSchema, sel: str, doc: SourceDocument, today: date, n_dates: int, n_items: int) -> Induction | None:
    try:
        res: ApplyResult = apply_schema(schema, doc, today)
    except Exception as exc:  # noqa: BLE001
        log.debug("induction: schema %r failed to execute: %s", sel, exc)
        return None
    if res.items_seen < MIN_ITEMS:
        return None
    valid = [ev for ev in res.events if not check_event(ev, doc, today, check_text=ev.grounded_text)]
    if len(valid) < MIN_ITEMS:
        return None
    pairs = {(ev.title.strip().lower(), ev.start_date) for ev in valid}
    distinct = len(pairs) / len(valid)
    coverage = min(1.0, len(valid) / max(n_dates, 1))
    failure_ratio = res.failure_ratio
    if failure_ratio > 0.3 or distinct < 0.8 or coverage < 0.5:
        return None
    rule = schema.rules[0]
    machine = isinstance(rule, HtmlRule) and rule.fields["date"].attr is not None
    quality = round(0.5 * coverage + 0.3 * distinct + 0.2 * (1 - failure_ratio), 3)
    titles: list[str] = []
    for ev in valid:
        if ev.title not in titles:
            titles.append(ev.title)
    return Induction(schema=schema, item_selector=sel, items=n_items, events=len(valid), coverage=coverage,
                     quality=quality, machine_dates=machine, sample_titles=titles[:5])
