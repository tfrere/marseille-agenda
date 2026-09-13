from datetime import date, time

import pytest

from marseille_agenda.dates import DateParseError, parse_date_text

TODAY = date(2026, 9, 12)  # a Saturday


@pytest.mark.parametrize(
    "text, start, end, st, et",
    [
        ("Le mardi 15 septembre à 18 h 30, au Grand Bar du Chapitre", date(2026, 9, 15), None, time(18, 30), None),
        ("Le mercredi 28 octobre, à 19 heures, à la librairie Transit", date(2026, 10, 28), None, time(19, 0), None),
        ("Le 6 & 7 juin de 11h-19h, à la Brasserie communale", date(2027, 6, 6), date(2027, 6, 7), time(11, 0), time(19, 0)),
        ("12 Sep 2026 @ Le Molotov / 21:00", date(2026, 9, 12), None, time(21, 0), None),
        ("19 Sept. Clean Port", date(2026, 9, 19), None, None, None),
        ("du 12 septembre au 3 octobre 2026", date(2026, 9, 12), date(2026, 10, 3), None, None),
        ("Exposition du 20 décembre au 5 janvier", date(2026, 12, 20), date(2027, 1, 5), None, None),
        ("2026-09-25T18:30", date(2026, 9, 25), None, time(18, 30), None),
        ("20260925", date(2026, 9, 25), None, None, None),
        ("15/09/2026 20h", date(2026, 9, 15), None, time(20, 0), None),
        ("1er octobre 2026 à 20h30", date(2026, 10, 1), None, time(20, 30), None),
        ("samedi 3 octobre", date(2026, 10, 3), None, None, None),
    ],
)
def test_parse_examples(text, start, end, st, et):
    p = parse_date_text(text, TODAY)
    assert (p.start, p.end, p.start_time, p.end_time) == (start, end, st, et)


def test_past_date_without_year_stays_in_current_year_when_weekday_matches():
    # "mardi 21 juillet": 2026-07-21 is a Tuesday and already past. The parser must NOT jump
    # to 2027 (where July 21 is a Wednesday); the downstream date check rejects it as past.
    p = parse_date_text("Le mardi 21 juillet à 18 h 30", TODAY)
    assert p.start == date(2026, 7, 21)


def test_weekday_mismatch_raises():
    # Sept 16, 2026 is a Wednesday; 2025 -> Tuesday (past), 2027 -> Thursday. "mardi" fits only 2025.
    p = parse_date_text("mardi 16 septembre", TODAY)
    assert p.start == date(2025, 9, 16)  # honest past date, rejected later, never invented
    with pytest.raises(DateParseError):
        parse_date_text("lundi 16 septembre", TODAY)  # no candidate year is a Monday


def test_explicit_year_with_wrong_weekday_raises():
    with pytest.raises(DateParseError):
        parse_date_text("mardi 16 septembre 2026", TODAY)


def test_strptime_format():
    p = parse_date_text("20261009", TODAY, "%Y%m%d")
    assert p.start == date(2026, 10, 9) and p.start_time is None
    p = parse_date_text("2026-10-09 18:30:00", TODAY, "%Y-%m-%d %H:%M:%S")
    assert p.start_time == time(18, 30)


def test_no_date():
    with pytest.raises(DateParseError):
        parse_date_text("Horaires : tous les jours", TODAY)


def test_iso_midnight_means_no_start_time():
    from datetime import date, time
    from marseille_agenda.dates import parse_date_text

    p = parse_date_text("2026-09-18T00:00:00+00:00 vendredi 18 septembre 2026", date(2026, 9, 13))
    assert (p.start, p.start_time) == (date(2026, 9, 18), None)
    p = parse_date_text("2026-09-18T20:30:00+02:00", date(2026, 9, 13))
    assert (p.start, p.start_time) == (date(2026, 9, 18), time(20, 30))
    # A midnight machine date next to a human time keeps the human time.
    p = parse_date_text("2026-09-18T00:00:00 de 16h00 à 17h15", date(2026, 9, 13))
    assert p.start_time == time(16, 0)
