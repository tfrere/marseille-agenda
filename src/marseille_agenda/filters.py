"""Category-level publication rules, applied identically to every source of a venue.

Cinemas list every screening of the week; the calendar only wants the sessions that are an
occasion (debate, guest, premiere, cine-club, cine-concert, festival night). The rule is a
vocabulary test over everything the source said about the event, not a per-venue adapter.
"""

from __future__ import annotations

import re
from typing import Protocol

from .schema import Venue
from .validate import normalize, strip_accents

_SPECIAL_SCREENING = re.compile(
    r"\b(?:"
    r"debat|discussion|rencontre|echange|en presence|presente par|suivie? d|precedee? d|anime par|"
    r"avant[- ]premiere|premiere|preview|"
    r"cine[- ]?club|cine[- ]?concert|cine[- ]?bistro|cine[- ]?goute|cine[- ]?quiz|quiz|"
    r"seance speciale|seance unique|soiree speciale|soiree|nuit du|marathon|"
    r"festival|retrospective|cycle|carte blanche|hommage|masterclass|conference|lecon de cinema|"
    r"atelier|workshop|braderie|anniversaire|fete|vernissage|exposition|"
    r"projection[- ]debat|projection[- ]rencontre|ouverture|cloture|inauguration|"
    r"invite|realisateur|realisatrice|equipe du film|interprete|"
    r"gratuit|entree libre"
    r")")


class _Eventish(Protocol):
    title: str
    summary: str | None
    location_name: str | None
    evidence: list[str]


def is_special_screening(ev: _Eventish) -> bool:
    """True when the source describes the session as more than a regular screening."""
    parts = [ev.title, ev.summary or "", ev.location_name or "", *ev.evidence, getattr(ev, "event_type", None) or ""]
    text = strip_accents(normalize(" ".join(parts)))
    return bool(_SPECIAL_SCREENING.search(text))


def publishable(venue: Venue, events: list) -> tuple[list, int]:
    """Apply the venue's category rule. Returns (kept events, number dropped)."""
    if venue.category != "cinema" or venue.all_screenings:
        return events, 0
    kept = [e for e in events if is_special_screening(e)]
    return kept, len(events) - len(kept)


def at_venue(venue: Venue, location: str | None) -> bool:
    """`Venue.location_filter` test: an aggregator (Mille Bâbords lists the whole militant
    scene) may be restricted to the events held at its own address. The regex is matched
    case- and accent-insensitively; an event without a location is kept."""
    if not venue.location_filter or not location:
        return True
    # Only accents are stripped from the pattern: lowercasing it would turn \S into \s.
    pattern = strip_accents(venue.location_filter)
    return re.search(pattern, strip_accents(normalize(location)), re.IGNORECASE) is not None


def held_here(venue: Venue, events: list) -> tuple[list, int]:
    """Apply `Venue.location_filter`. Returns (kept events, number dropped)."""
    if not venue.location_filter:
        return events, 0
    kept = [e for e in events if at_venue(venue, e.location_name)]
    return kept, len(events) - len(kept)
