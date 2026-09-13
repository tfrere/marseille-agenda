"""Execute a declarative ExtractionSchema against a fetched document. No LLM involved."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from .dates import DateParseError, parse_date_text, parse_times
from .extraction_schema import ExtractionSchema, FieldSpec, HtmlRule, JsonRule
from .fetch import SourceDocument, html_to_text
from .schema import EventStatus, ExtractedEvent


class SchemaEvent(ExtractedEvent):
    event_type: str | None = None
    free: bool | None = None
    grounded_text: bool = True
    """False for JSON sources: evidence is a rendering of the item, not a page excerpt."""


@dataclass
class ApplyResult:
    events: list[SchemaEvent] = field(default_factory=list)
    items_seen: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def failure_ratio(self) -> float:
        return len(self.failures) / self.items_seen if self.items_seen else 0.0


def apply_schema(schema: ExtractionSchema, doc: SourceDocument, today: date) -> ApplyResult:
    result = ApplyResult()
    for rule in schema.rules:
        if isinstance(rule, HtmlRule):
            _apply_html(rule, doc, today, result)
        else:
            _apply_json(rule, doc, today, result)
    # De-duplicate on (title, start_date), keeping the first occurrence.
    seen: set[tuple[str, date]] = set()
    unique: list[SchemaEvent] = []
    for ev in result.events:
        key = (ev.title.strip().lower(), ev.start_date)
        if key not in seen:
            seen.add(key)
            unique.append(ev)
    result.events = unique
    return result


# --------------------------------------------------------------------------- HTML


def _html_values(item: Tag, spec: FieldSpec, base_url: str) -> list[str]:
    if spec.constant is not None:
        return [spec.constant]
    nodes: list[Tag] = item.select(spec.selector) if spec.selector else [item]
    values: list[str] = []
    for n in nodes:
        if spec.attr:
            v = n.get(spec.attr)
            if isinstance(v, list):
                v = " ".join(v)
            if v is None:
                continue
            v = str(v).strip()
            if spec.attr in ("href", "src") and v:
                v = urljoin(base_url, v)
        else:
            v = n.get_text(" ", strip=True)
        v = _apply_regex(v, spec.regex)
        if v:
            values.append(v)
    if spec.join is not None and values:
        return [spec.join.join(values)]
    return values


_DATE_ATTRS = ("data-date", "datetime", "data-start", "data-start-date", "data-day", "content")
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _iso_attribute_dates(item: Tag) -> list[str]:
    """ISO dates carried by attributes of the item, its descendants, then its close ancestors.

    Agenda pages often group events under a day and put the machine date on the group or the
    card (`data-date="2026-09-13"`) while the card text only shows the time.
    """
    out: list[str] = []

    def scan(node: Tag) -> None:
        for a in _DATE_ATTRS:
            v = node.get(a)
            if isinstance(v, str) and (m := _ISO_DAY.search(v)):
                out.append(m.group(0))

    for node in [item, *item.find_all(True)]:
        scan(node)
    parent, depth = item.parent, 0
    while isinstance(parent, Tag) and depth < 4 and not out:
        scan(parent)
        parent, depth = parent.parent, depth + 1
    return out


def _date_fallback(item: Tag, field_text: str, today: date, date_format: str | None):
    """The selected date text has no date in it (typically just a time): look further.

    A selector that only catches part of the date line is a very common generator mistake.
    The whole item text is the next best evidence, then ISO attributes around the item; the
    agreement check at generation time still judges the outcome.
    """
    try:
        return parse_date_text(item.get_text(" ", strip=True), today, date_format)
    except DateParseError:
        pass
    for iso in _iso_attribute_dates(item):
        try:
            return parse_date_text(f"{iso} {field_text}", today)
        except DateParseError:
            continue
    raise DateParseError(f"no date in the selected text {field_text!r}, the item text or data-date/datetime attributes")


def _apply_html(rule: HtmlRule, doc: SourceDocument, today: date, result: ApplyResult) -> None:
    soup = BeautifulSoup(doc.raw, "lxml")
    for sel in rule.exclude_selectors:
        for n in soup.select(sel):
            n.decompose()
    container: Tag = soup
    if rule.container_selector:
        found = soup.select_one(rule.container_selector)
        if found is None:
            result.failures.append(f"container not found: {rule.container_selector}")
            return
        container = found
    items = container.select(rule.item_selector)
    result.items_seen += len(items)
    if not items:
        result.failures.append(f"no items for selector: {rule.item_selector}")
        return

    for item in items:
        try:
            titles = _html_values(item, rule.fields["title"], doc.url)
            if not titles:
                raise ValueError("empty title")
            title = titles[0]
            date_values = _html_values(item, rule.fields["date"], doc.url) if "date" in rule.fields else [item.get_text(" ", strip=True)]
            evidence_text, _ = html_to_text(str(item), doc.url)
            # Plain truncation (no ellipsis) so the quote stays a verbatim substring of the page.
            evidence = [re.sub(r"\s+", " ", evidence_text).strip()[:300]]
            time_text = _first(_html_values(item, rule.fields["time"], doc.url)) if "time" in rule.fields else None
            end_date_text = _first(_html_values(item, rule.fields["end_date"], doc.url)) if "end_date" in rule.fields else None
            common = _common_fields(rule, item, doc.url, lambda spec: _html_values(item, spec, doc.url))
            parsed_dates = []
            for dv in date_values:
                try:
                    parsed_dates.append(parse_date_text(dv, today, rule.date_format))
                except DateParseError:
                    continue
            if not parsed_dates:
                # Prefer the value that looks like a time ("14h-19h") as the companion of the fallback date.
                time_like = next((dv for dv in date_values if re.search(r"\d{1,2}\s*[h:]\s*\d{0,2}", dv)), time_text or "")
                parsed_dates = [_date_fallback(item, time_like, today, rule.date_format)]
            for parsed in parsed_dates:
                start_time, end_time = parsed.start_time, parsed.end_time
                if time_text:
                    # A time field holds no date ("11h", "20:30 - 22:00"): read the times only.
                    t_start, t_end = parse_times(time_text)
                    if t_start:
                        start_time, end_time = t_start, t_end
                end_date = parsed.end
                if end_date_text:
                    try:
                        end_date = parse_date_text(end_date_text, today, rule.date_format).start
                    except DateParseError:
                        pass
                result.events.append(
                    SchemaEvent(
                        title=title, start_date=parsed.start, end_date=end_date,
                        start_time=start_time, end_time=end_time, evidence=evidence, **common,
                    )
                )
        except (DateParseError, ValueError, KeyError) as exc:
            result.failures.append(f"{_clip(item.get_text(' ', strip=True), 80)!r}: {exc}")


# --------------------------------------------------------------------------- JSON


def _resolve(obj: Any, path: str) -> Any:
    if path in ("", "$"):
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            if part.isdigit():
                idx = int(part)
                cur = cur[idx] if idx < len(cur) else None
            else:
                cur = [_resolve(x, part) for x in cur]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _json_values(ctx: dict[str, Any], spec: FieldSpec) -> list[str]:
    if spec.constant is not None:
        return [spec.constant]
    path = spec.selector
    scope = ctx["sub"]
    if path.startswith("item."):
        scope, path = ctx["item"], path[5:]
    elif path.startswith("sub."):
        path = path[4:]
    v = _resolve(scope, path)
    if v is None:
        return []
    if isinstance(v, list):
        vals = [x for x in (_scalar(x) for x in v) if x]
        if spec.join is not None:
            vals = [spec.join.join(dict.fromkeys(vals))] if vals else []
    else:
        s = _scalar(v)
        vals = [s] if s else []
    return [x for x in (_apply_regex(x, spec.regex) for x in vals) if x]


def _scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def _apply_json(rule: JsonRule, doc: SourceDocument, today: date, result: ApplyResult) -> None:
    try:
        data = json.loads(doc.raw)
    except json.JSONDecodeError as exc:
        result.failures.append(f"invalid JSON: {exc}")
        return
    items = _resolve(data, rule.items_path)
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        result.failures.append(f"items_path {rule.items_path!r} did not resolve to a list")
        return

    for item in items:
        subs: list[Any]
        if rule.expand_path:
            expanded = _resolve(item, rule.expand_path)
            subs = expanded if isinstance(expanded, list) else []
        else:
            subs = [item]
        for sub in subs:
            result.items_seen += 1
            ctx = {"item": item, "sub": sub}
            try:
                title = _first(_json_values(ctx, rule.fields["title"]))
                if not title:
                    raise ValueError("empty title")
                title = _strip_html(title)
                date_text = _first(_json_values(ctx, rule.fields["date"]))
                if not date_text:
                    raise ValueError("empty date")
                parsed = parse_date_text(date_text, today, rule.date_format)
                start_time, end_time = parsed.start_time, parsed.end_time
                if "time" in rule.fields:
                    start_time = _parse_time(_first(_json_values(ctx, rule.fields["time"])), rule.time_format) or start_time
                if "end_time" in rule.fields:
                    end_time = _parse_time(_first(_json_values(ctx, rule.fields["end_time"])), rule.time_format) or end_time
                end_date = parsed.end
                if "end_date" in rule.fields:
                    ed = _first(_json_values(ctx, rule.fields["end_date"]))
                    if ed:
                        end_date = parse_date_text(ed, today, rule.date_format).start
                common = _common_fields(rule, None, doc.url, lambda spec: _json_values(ctx, spec))
                evidence = [_clip(f"{title} | {date_text}" + (f" | {start_time:%H:%M}" if start_time else ""), 300)]
                result.events.append(
                    SchemaEvent(
                        title=title, start_date=parsed.start, end_date=end_date,
                        start_time=start_time, end_time=end_time, evidence=evidence,
                        grounded_text=False, **common,
                    )
                )
            except (DateParseError, ValueError, KeyError, TypeError) as exc:
                result.failures.append(f"item {_clip(_scalar(sub), 80)!r}: {exc}")


# --------------------------------------------------------------------------- shared


def _common_fields(rule: HtmlRule | JsonRule, item: Tag | None, base_url: str, read) -> dict[str, Any]:
    out: dict[str, Any] = {}
    f = rule.fields
    if "url" in f:
        u = _first(read(f["url"]))
        if u and u.startswith("http"):
            out["url"] = u
        elif u and u.startswith("/"):
            out["url"] = urljoin(base_url, u)
    if "location" in f:
        out["location_name"] = _strip_html(_first(read(f["location"])) or "") or None
    if "price" in f:
        out["price"] = _strip_html(_first(read(f["price"])) or "") or None
    if "summary" in f:
        s = _strip_html(_first(read(f["summary"])) or "")
        out["summary"] = _clip(s, 240) or None
    if "event_type" in f:
        out["event_type"] = _strip_html(_first(read(f["event_type"])) or "") or None
    if "free" in f:
        v = (_first(read(f["free"])) or "").lower()
        if v in ("true", "1", "gratuit", "free", "oui", "yes", "entrée libre"):
            out["free"] = True
        elif v in ("false", "0", "non", "no", "payant"):
            out["free"] = False
        else:
            out["free"] = None
        if out["free"] and not out.get("price"):
            out["price"] = "Gratuit"
    status: EventStatus = "scheduled"
    st = (_first(read(f["status"])) or "").lower() if "status" in f else ""
    if "annul" in st or "cancel" in st:
        status = "cancelled"
    elif "report" in st or "postpon" in st:
        status = "postponed"
    elif "complet" in st or "sold" in st:
        status = "sold_out"
    out["status"] = status
    return out


def _parse_time(v: str | None, fmt: str | None) -> time | None:
    if not v:
        return None
    if fmt:
        try:
            return datetime.strptime(v.strip(), fmt).time()
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})\s*(?:h|:)\s*(\d{2})?", v)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    return time(h, mi) if 0 <= h <= 23 and 0 <= mi <= 59 else None


def _apply_regex(v: str, pattern: str | None) -> str:
    if not pattern or not v:
        return v
    m = re.search(pattern, v, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return (m.group(1) if m.groups() else m.group(0)).strip()


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&#038;", "&").replace("&amp;", "&").replace("&nbsp;", " ").replace("&#8217;", "'")
    return re.sub(r"\s+", " ", s).strip()


def _first(vals: list[str]) -> str | None:
    return vals[0] if vals else None


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."
