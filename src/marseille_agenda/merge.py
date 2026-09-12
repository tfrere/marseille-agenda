"""Stable identifiers, state merging, volume guard and expiry."""

from __future__ import annotations

import hashlib
from datetime import date

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
