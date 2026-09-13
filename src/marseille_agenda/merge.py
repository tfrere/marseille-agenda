"""Stable identifiers, state merging, volume guard and expiry."""

from __future__ import annotations

import hashlib
from datetime import date, time, timedelta
from typing import TypeVar

from pydantic import BaseModel

from .config import MISSING_RUNS_BEFORE_DROP, VOLUME_DROP_RATIO, VOLUME_GUARD_MIN_EVENTS
from .schema import Alert, Event, SourceState, State
from .validate import normalize, strip_accents


def make_uid(venue_id: str, title: str, start: date) -> str:
    key = f"{venue_id}|{strip_accents(normalize(title))}|{start.isoformat()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def merge_source(
    state: State,
    source_url: str,
    venue_id: str,
    new_events: list[Event] | None,
    today: date,
    content_hash: str | None,
    error: str | None = None,
) -> list[Alert]:
    """Merge one source's freshly extracted events into `state`.

    `new_events=None` means the source could not be processed this run (fetch/LLM error):
    previous events are kept untouched and an alert is raised.
    """
    alerts: list[Alert] = []
    src = state.sources.get(source_url) or SourceState(url=source_url)
    state.sources[source_url] = src
    src.last_fetched = today

    if new_events is None:
        src.last_error = error or "unknown error"
        alerts.append(Alert(level="error", venue_id=venue_id, source_url=source_url,
                            message=f"source failed, last verified state kept: {src.last_error}"))
        return alerts

    src.last_error = None
    src.last_ok = today
    src.content_hash = content_hash
    prev_uids = set(src.uids)
    new_uids = {e.uid for e in new_events}

    # Volume guard: a sudden collapse is more likely a broken page/extraction than reality.
    prev_live = [u for u in prev_uids if u in state.events and _still_upcoming(state.events[u], today)]
    collapsed = len(prev_live) >= VOLUME_GUARD_MIN_EVENTS and (
        len(new_uids) == 0 or len(new_uids) < len(prev_live) * (1 - VOLUME_DROP_RATIO)
    )
    if collapsed:
        alerts.append(Alert(level="warning", venue_id=venue_id, source_url=source_url,
                            message=f"event count dropped from {len(prev_live)} to {len(new_uids)}; previous events kept"))

    for e in new_events:
        old = state.events.get(e.uid)
        if old:
            e.first_seen = old.first_seen
        e.last_seen = today
        state.events[e.uid] = e
        src.missing_runs.pop(e.uid, None)

    for uid in prev_uids - new_uids:
        if uid not in state.events:
            continue
        if collapsed:
            continue  # keep everything while the guard is active
        n = src.missing_runs.get(uid, 0) + 1
        src.missing_runs[uid] = n
        if n >= MISSING_RUNS_BEFORE_DROP:
            del state.events[uid]
            src.missing_runs.pop(uid, None)

    src.uids = sorted(uid for uid, e in state.events.items() if e.source_url == source_url)
    src.event_count = len(new_uids)
    return alerts


MIN_DAILY_RUN = 3
MAX_CLOSED_DAYS = 2  # weekly closing days (Monday/Tuesday) do not break a run
E = TypeVar("E", bound=BaseModel)


def collapse_daily_runs(events: list[E]) -> list[E]:
    """Fold the same event repeated on consecutive days into one entry with a date range.

    Agendas listed day by day (Friche la Belle de Mai, cinemas) repeat every running exhibition
    under each date; the calendar wants one entry "from ... to ...", not one per day. Runs shorter
    than MIN_DAILY_RUN days are left alone (a two-night concert stays two entries); gaps of up to
    MAX_CLOSED_DAYS days (weekly closing) do not break a run. Works on the
    events of ONE source (schema events or published events: title, dates, url, evidence).
    """
    groups: dict[tuple[str, str], list[E]] = {}
    for e in events:
        groups.setdefault((strip_accents(normalize(e.title)), e.url or ""), []).append(e)
    out: list[E] = []
    for group in groups.values():
        group.sort(key=lambda e: (e.start_date, e.start_time or _MIDNIGHT))
        run: list[E] = []
        for e in group:
            if run and e.start_date <= (run[-1].end_date or run[-1].start_date) + timedelta(days=MAX_CLOSED_DAYS + 1):
                run.append(e)
                continue
            out.extend(_fold(run))
            run = [e]
        out.extend(_fold(run))
    out.sort(key=lambda e: (e.start_date, e.start_time or _MIDNIGHT, e.title))
    return out


def _fold(run: list[E]) -> list[E]:
    if len(run) < MIN_DAILY_RUN:
        return run
    first = run[0].model_copy(deep=True)
    first.end_date = max((e.end_date or e.start_date) for e in run)
    # Not a page quote: tells the verifier (and readers of events.json) where the range comes from.
    first.evidence = [*first.evidence, f"same entry listed on every day from {first.start_date} to {first.end_date}"]
    return [first]


_MIDNIGHT = time(0, 0)


def expire_past(state: State, today: date) -> int:
    gone = [uid for uid, e in state.events.items() if not _still_upcoming(e, today)]
    for uid in gone:
        del state.events[uid]
        for src in state.sources.values():
            src.missing_runs.pop(uid, None)
            if uid in src.uids:
                src.uids.remove(uid)
    return len(gone)


def _still_upcoming(e: Event, today: date) -> bool:
    return (e.end_date or e.start_date) >= today
