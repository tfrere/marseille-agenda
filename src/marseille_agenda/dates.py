"""Deterministic parsing of French (and a few machine) date/time expressions.

Handles the wording found on venue pages: "Le mardi 15 septembre à 18 h 30",
"du 6 au 7 juin", "12 Sep 2026 @ 21:00", "19 Sept.", "2026-09-25T18:30", "20260925",
"15/09/2026". When the year is missing it is inferred as the nearest occurrence on or
after `today`, constrained by the weekday name when one is written ("mardi 15 septembre"
must be a Tuesday, otherwise the candidate year is rejected).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time

FRENCH_WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

_MONTHS: dict[str, int] = {}
for _i, _names in enumerate(
    [
        ("janvier", "janv", "jan", "january"),
        ("fevrier", "fev", "feb", "february"),
        ("mars", "mar", "march"),
        ("avril", "avr", "apr", "april"),
        ("mai", "may"),
        ("juin", "jun", "june"),
        ("juillet", "juil", "jul", "july"),
        ("aout", "aug", "august"),
        ("septembre", "sept", "sep", "september"),
        ("octobre", "oct", "october"),
        ("novembre", "nov", "november"),
        ("decembre", "dec", "december"),
    ],
    start=1,
):
    for _n in _names:
        _MONTHS[_n] = _i

_MONTH_RE = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_WD_RE = "(" + "|".join(FRENCH_WEEKDAYS) + r")\.?"
_DAY = r"(\d{1,2})(?:er)?"

# "du 12 septembre au 3 octobre 2026" / "du 6 au 7 juin"
_RANGE_FULL = re.compile(
    rf"\bdu\s+{_DAY}\s+(?:{_MONTH_RE}\s+)?(?:(\d{{4}})\s+)?au\s+{_DAY}\s+{_MONTH_RE}(?:\s+(\d{{4}}))?"
)
# "6 & 7 juin", "6 et 7 juin", "12-13 septembre"
_RANGE_SHORT = re.compile(rf"\b{_DAY}\s*(?:&|et|-|–|/)\s*{_DAY}\s+{_MONTH_RE}(?:\s+(\d{{4}}))?")
# "mardi 15 septembre 2026", "15 sept.", "12 sep 2026"
_SINGLE = re.compile(rf"(?:\b{_WD_RE}\s+)?\b{_DAY}\s+{_MONTH_RE}(?:\s+(\d{{4}}))?")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[tT ](\d{1,2}):(\d{2}))?")
_COMPACT = re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
_TIME = re.compile(r"\b(\d{1,2})\s*(?:h|:|heures?)\s*(\d{2})?(?!\d)")
_WEEKDAY_WORD = re.compile(rf"\b{_WD_RE}")


class DateParseError(ValueError):
    pass


@dataclass
class ParsedDate:
    start: date
    end: date | None
    start_time: time | None
    end_time: time | None
    year_inferred: bool
    weekday_named: str | None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _times(text: str) -> tuple[time | None, time | None]:
    found: list[time] = []
    for m in _TIME.finditer(text):
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if 0 <= h <= 23 and 0 <= mi <= 59:
            found.append(time(h, mi))
    if not found:
        return None, None
    return found[0], (found[1] if len(found) > 1 else None)


def _infer_year(month: int, day: int, today: date, weekday: str | None) -> tuple[date, bool]:
    """Pick the year for a day/month with no year written."""
    candidates = [today.year - 1, today.year, today.year + 1]
    dated = [d for y in candidates if (d := _safe_date(y, month, day))]
    if not dated:
        raise DateParseError(f"{day:02d}/{month:02d} is not a valid date in {candidates}")
    if weekday:
        dated = [d for d in dated if FRENCH_WEEKDAYS[d.weekday()] == weekday]
        if not dated:
            raise DateParseError(f"no year in {candidates} makes {day:02d}/{month:02d} a {weekday}")
    upcoming = [d for d in dated if d >= today]
    if upcoming:
        return upcoming[0], True
    return max(dated), True


def parse_date_text(text: str, today: date, fmt: str | None = None) -> ParsedDate:
    """Parse a date (and optional time / end) out of free text.

    `fmt` is a strptime format for machine-formatted values (e.g. "%Y%m%d").
    Raises DateParseError when nothing usable is found.
    """
    raw = text.strip()
    if not raw:
        raise DateParseError("empty date text")

    if fmt:
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError as exc:
            raise DateParseError(f"{raw!r} does not match {fmt!r}") from exc
        has_time = any(t in fmt for t in ("%H", "%I", "%M"))
        return ParsedDate(dt.date(), None, dt.time() if has_time else None, None, False, None)

    t = _norm(raw)
    wd_match = _WEEKDAY_WORD.search(t)
    weekday = wd_match.group(1) if wd_match else None
    start_time, end_time = _times(t)

    m = _ISO.search(t)
    if m:
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            if m.group(4):
                st = time(int(m.group(4)), int(m.group(5)))
                others = [x for x in (start_time, end_time) if x and x != st]
                return ParsedDate(d, None, st, others[0] if others else None, False, weekday)
            return ParsedDate(d, None, start_time, end_time, False, weekday)

    m = _COMPACT.search(t)
    if m:
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return ParsedDate(d, None, start_time, end_time, False, weekday)

    m = _NUMERIC.search(t)
    if m:
        y = int(m.group(3))
        y = y + 2000 if y < 100 else y
        d = _safe_date(y, int(m.group(2)), int(m.group(1)))
        if d:
            return ParsedDate(d, None, start_time, end_time, False, weekday)

    m = _RANGE_FULL.search(t)
    if m:
        d1, mon1, y1, d2, mon2, y2 = m.groups()
        month2 = _MONTHS[mon2]
        month1 = _MONTHS[mon1] if mon1 else month2
        year = int(y2 or y1) if (y2 or y1) else None
        if year:
            end = _safe_date(year, month2, int(d2))
            start = _safe_date(year if month1 <= month2 else year - 1, month1, int(d1))
            if not start or not end:
                raise DateParseError(f"invalid range in {raw!r}")
            return ParsedDate(start, end, start_time, end_time, False, weekday)
        end, _ = _infer_year(month2, int(d2), today, None)
        start = _safe_date(end.year if month1 <= month2 else end.year - 1, month1, int(d1))
        if not start:
            raise DateParseError(f"invalid range in {raw!r}")
        return ParsedDate(start, end, start_time, end_time, True, weekday)

    m = _RANGE_SHORT.search(t)
    if m:
        d1, d2, mon, y = m.groups()
        month = _MONTHS[mon]
        if y:
            start, end = _safe_date(int(y), month, int(d1)), _safe_date(int(y), month, int(d2))
            if not start or not end:
                raise DateParseError(f"invalid range in {raw!r}")
            return ParsedDate(start, end, start_time, end_time, False, weekday)
        end, _ = _infer_year(month, int(d2), today, None)
        start = _safe_date(end.year, month, int(d1))
        if not start:
            raise DateParseError(f"invalid range in {raw!r}")
        return ParsedDate(start, end, start_time, end_time, True, weekday)

    m = _SINGLE.search(t)
    if m:
        wd, day, mon, y = m.groups()
        month = _MONTHS[mon]
        if y:
            d = _safe_date(int(y), month, int(day))
            if not d:
                raise DateParseError(f"invalid date in {raw!r}")
            if wd and FRENCH_WEEKDAYS[d.weekday()] != wd:
                raise DateParseError(f"{d} is a {FRENCH_WEEKDAYS[d.weekday()]}, text says {wd}")
            return ParsedDate(d, None, start_time, end_time, False, wd)
        d, inferred = _infer_year(month, int(day), today, wd or weekday)
        return ParsedDate(d, None, start_time, end_time, inferred, wd or weekday)

    raise DateParseError(f"no date found in {raw!r}")
