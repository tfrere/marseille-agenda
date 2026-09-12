"""Deterministic grounding and sanity checks applied to every LLM-extracted event.

These run before the adversarial verifier and cost nothing. They catch outright
inventions (quotes or URLs absent from the source) and impossible dates.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta

from .config import MAX_HORIZON_DAYS
from .fetch import SourceDocument
from .schema import ExtractedEvent

FRENCH_WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_WEEKDAY_RE = re.compile(r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b", re.IGNORECASE)


def normalize(s: str) -> str:
    """Whitespace/quote/case-insensitive form used for substring matching."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00a0", " ").replace("\u202f", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalized_source(doc: SourceDocument) -> tuple[str, str]:
    """Normalizing a 600 KB document per event is the hot spot; cache it on the document."""
    cached = doc.cache.get("normalized")
    if cached is None:
        norm = normalize(doc.text)
        cached = (norm, strip_accents(norm))
        doc.cache["normalized"] = cached
    return cached


def check_event(ev: ExtractedEvent, doc: SourceDocument, today: date, *, check_text: bool = True) -> list[str]:
    """Return a list of human-readable problems; empty list means the event passes.

    `check_text=False` skips the verbatim checks for events built from structured data
    (JSON), where evidence is a rendering of the record rather than a page excerpt.
    """
    problems: list[str] = []
    norm_source, norm_source_noacc = _normalized_source(doc)

    if check_text:
        # 1. Evidence quotes must be verbatim substrings of the source text.
        for q in ev.evidence:
            nq = normalize(q)
            if len(nq) < 12:
                problems.append(f"evidence too short to be meaningful: {q!r}")
            elif nq not in norm_source and strip_accents(nq) not in norm_source_noacc:
                problems.append(f"evidence not found verbatim in source: {q[:80]!r}")

        # 2. Title must appear in the source (copied, not paraphrased).
        nt = normalize(ev.title)
        if nt not in norm_source and strip_accents(nt) not in norm_source_noacc:
            problems.append(f"title not found in source: {ev.title!r}")

    # 3. URL must be a link present in the document (or the page itself).
    if ev.url and check_text:
        u = ev.url.strip()
        candidates = {u, u.rstrip("/"), u + "/"}
        known = set(doc.links) | {doc.url, doc.url.rstrip("/")}
        if not (candidates & known) and u not in doc.raw:
            problems.append(f"url not present in source: {u}")

    # 4. Date sanity.
    last_day = ev.end_date or ev.start_date
    if last_day < today:
        problems.append(f"event is in the past ({ev.start_date} .. {last_day}, today {today})")
    if ev.start_date > today + timedelta(days=MAX_HORIZON_DAYS):
        problems.append(f"start_date {ev.start_date} is implausibly far in the future")
    if ev.end_date and ev.end_date < ev.start_date:
        problems.append(f"end_date {ev.end_date} before start_date {ev.start_date}")

    # 5. Weekday consistency: if the evidence names exactly one weekday, the date must match.
    weekdays = {m.lower() for m in _WEEKDAY_RE.findall(" ".join(ev.evidence))}
    if len(weekdays) == 1:
        named = weekdays.pop()
        actual = FRENCH_WEEKDAYS[ev.start_date.weekday()]
        if named != actual:
            problems.append(f"weekday mismatch: evidence says {named} but {ev.start_date} is a {actual}")

    return problems


def evidence_context(ev: ExtractedEvent, doc: SourceDocument, radius: int = 700) -> str:
    """Source excerpts around each evidence quote, for the verifier."""
    text = doc.text
    norm_text = normalize(text)
    windows: list[tuple[int, int]] = []
    for q in ev.evidence:
        nq = normalize(q)
        idx = norm_text.find(nq)
        if idx < 0:
            continue
        # normalize() collapses whitespace so offsets drift; map roughly by ratio.
        ratio = len(text) / max(len(norm_text), 1)
        center = int(idx * ratio)
        windows.append((max(0, center - radius), min(len(text), center + len(q) + radius)))
    if not windows:
        return text[: 2 * radius]
    windows.sort()
    merged: list[list[int]] = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return "\n[...]\n".join(text[a:b] for a, b in merged)
