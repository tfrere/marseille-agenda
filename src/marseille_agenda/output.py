"""Write the published artifacts: events.json, events.ics, report.json."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from icalendar import Calendar
from icalendar import Event as IcsEvent

from .config import TZ
from .schema import Event, RunReport, State, Venue
from .validate import normalize, strip_accents

# Preferred source when the same event is announced in several places.
_SOURCE_RANK = {"json": 0, "html": 1, "facebook": 2, "instagram": 3}
_STOPWORDS = {"de", "du", "des", "la", "le", "les", "l", "d", "a", "au", "aux", "et", "en", "un", "une", "the", "of", "avec", "par", "sur"}


def _tokens(title: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", strip_accents(normalize(title))) if len(t) > 1 and t not in _STOPWORDS}


def _same_event(a: Event, b: Event) -> bool:
    """Same venue and day; same clock time when both have one; titles overlapping enough."""
    if a.venue_id != b.venue_id or a.start_date != b.start_date:
        return False
    if a.start_time and b.start_time and a.start_time != b.start_time:
        return False
    ta, tb = _tokens(a.title), _tokens(b.title)
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / min(len(ta), len(tb))
    # A shared clock time is strong evidence: a looser title match is enough then.
    return overlap >= (0.5 if a.start_time and b.start_time else 0.75)


def published_events(state: State) -> list[Event]:
    """Events to publish: one entry per real-world event across website / Facebook / Instagram.

    Cross-source duplicates carry different titles ("REBECCA, Alfred Hitchcock, 1940" on Instagram
    vs "HANTEES ciné-club : REBECCA d'Hitchcock" on Facebook), so exact uids cannot catch them.
    The state keeps every copy; only the output is folded, preferring the most structured source.
    """
    ranked = sorted(state.events.values(), key=lambda e: (_SOURCE_RANK.get(e.source_kind, 9), _sort_key(e)))
    kept: list[Event] = []
    for e in ranked:
        if not any(_same_event(e, k) for k in kept if k.start_date == e.start_date):
            kept.append(e)
    return sorted(kept, key=_sort_key)


def _sort_key(e: Event):
    return (e.start_date, e.start_time or datetime.min.time(), e.venue_name, e.title)


def write_events_json(path: Path, state: State, venues: list[Venue], today: date) -> None:
    events = published_events(state)
    payload = {
        "generated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "today": today.isoformat(),
        "venues": [
            {
                "id": v.id,
                "name": v.name,
                "map_name": v.map_name,
                "category": v.category,
                "lat": v.lat,
                "lng": v.lng,
                "website": v.website,
                "event_count": sum(1 for e in events if e.venue_id == v.id),
            }
            for v in venues
        ],
        "events": [json.loads(e.model_dump_json()) for e in events],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def write_ics(path: Path, state: State) -> None:
    cal = Calendar()
    cal.add("prodid", "-//marseille-agenda//tfrere//FR")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", "Marseille - agenda des lieux")
    cal.add("x-wr-timezone", "Europe/Paris")
    for e in published_events(state):
        ics = IcsEvent()
        ics.add("uid", f"{e.uid}@marseille-agenda")
        ics.add("summary", e.title)
        if e.start_time and not e.recurring:
            start = datetime.combine(e.start_date, e.start_time, tzinfo=TZ)
            ics.add("dtstart", start)
            if e.end_time:
                end_day = e.end_date or e.start_date
                ics.add("dtend", datetime.combine(end_day, e.end_time, tzinfo=TZ))
            else:
                ics.add("dtend", start + timedelta(hours=2))
        else:
            ics.add("dtstart", e.start_date)
            ics.add("dtend", (e.end_date or e.start_date) + timedelta(days=1))
        loc = e.venue_name + (f" - {e.location_name}" if e.location_name else "")
        ics.add("location", loc)
        desc_parts = [p for p in [e.summary, f"Prix : {e.price}" if e.price else None, e.url] if p]
        ics.add("description", "\n".join(desc_parts))
        if e.url:
            ics.add("url", e.url)
        if e.status == "cancelled":
            ics.add("status", "CANCELLED")
        ics.add("categories", [e.category])
        cal.add_component(ics)
    path.write_bytes(cal.to_ical())


def write_report(path: Path, report: RunReport) -> None:
    path.write_text(report.model_dump_json(indent=1), encoding="utf-8")


def load_state(path: Path) -> State:
    if path.exists():
        return State.model_validate_json(path.read_text(encoding="utf-8"))
    return State()


def save_state(path: Path, state: State) -> None:
    path.write_text(state.model_dump_json(indent=1), encoding="utf-8")
