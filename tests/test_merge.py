from datetime import date, time

from marseille_agenda.merge import collapse_daily_runs, make_uid
from marseille_agenda.schema import Event


def _event(title: str, day: date, start: time | None = time(14, 0), url: str | None = "https://x.test/e") -> Event:
    return Event(
        uid=make_uid("friche", title, day), venue_id="friche", venue_name="Friche", category="culture",
        title=title, start_date=day, start_time=start, end_time=time(19, 0), url=url,
        source_url="https://x.test/agenda/", source_kind="html", evidence=[f"{title} {day}"],
        first_seen=day, last_seen=day,
    )


def test_collapse_daily_runs_folds_exhibitions_repeated_every_day():
    events = [_event("Expo-pause", date(2026, 9, d)) for d in range(13, 28) if d not in (14, 15, 21, 22)]  # closed Mon/Tue
    events += [_event("Marché paysan", date(2026, 9, d), time(16, 0)) for d in (14, 21, 28)]  # weekly
    events += [_event("Concert", date(2026, 9, 19), time(20, 0)), _event("Concert", date(2026, 9, 20), time(20, 0))]
    out = collapse_daily_runs(events)
    by_title = {}
    for e in out:
        by_title.setdefault(e.title, []).append(e)
    expo = by_title["Expo-pause"]
    assert len(expo) == 1 and (expo[0].start_date, expo[0].end_date) == (date(2026, 9, 13), date(2026, 9, 27))
    assert expo[0].uid == make_uid("friche", "Expo-pause", date(2026, 9, 13))
    assert expo[0].start_time == time(14, 0)
    assert len(by_title["Marché paysan"]) == 3, "weekly repeats are not consecutive days: untouched"
    assert len(by_title["Concert"]) == 2, "a two-night run stays two entries"


def test_collapse_daily_runs_splits_on_gaps_and_urls():
    a = [_event("Atelier", date(2026, 10, d)) for d in (1, 2, 3)]
    b = [_event("Atelier", date(2026, 10, d)) for d in (7, 8, 9)]  # 3 closed days: a new run
    other = [_event("Atelier", date(2026, 10, d), url="https://x.test/other") for d in (1, 2, 3)]
    out = collapse_daily_runs(a + b + other)
    ranges = sorted((e.url, e.start_date, e.end_date) for e in out)
    assert ranges == [
        ("https://x.test/e", date(2026, 10, 1), date(2026, 10, 3)),
        ("https://x.test/e", date(2026, 10, 7), date(2026, 10, 9)),
        ("https://x.test/other", date(2026, 10, 1), date(2026, 10, 3)),
    ]


def test_published_events_folds_cross_source_duplicates():
    from marseille_agenda.output import published_events
    from marseille_agenda.schema import State

    def ev(title, kind, start=time(19, 45), day=date(2026, 9, 15), venue="les-varietes"):
        e = _event(title, day, start, url=None)
        e.venue_id, e.source_kind, e.source_url = venue, kind, f"https://{kind}.test/"
        e.uid = make_uid(venue, title, day)
        return e

    st = State()
    for e in [
        ev("HANTEES ciné-club : REBECCA d'Hitchcock", "facebook"),
        ev("REBECCA, Alfred Hitchcock, 1940", "instagram"),
        ev("Rebecca", "html"),
        ev("SEANCE UNIQUE Projection", "instagram", time(19, 45), date(2026, 9, 17)),  # different day: kept
        ev("Rebecca", "html", time(19, 45), date(2026, 9, 15), venue="le-gyptis"),  # other venue: kept
        ev("Concert Rebecca Trio", "instagram", time(21, 0)),  # same day, other time: kept
    ]:
        st.events[e.uid + e.source_kind] = e
    out = published_events(st)
    assert [(e.title, e.source_kind) for e in out if e.start_date == date(2026, 9, 15) and e.venue_id == "les-varietes"] == [
        ("Rebecca", "html"),
        ("Concert Rebecca Trio", "instagram"),
    ]
    assert len(out) == 4


def test_published_events_folds_daily_copies_into_the_matching_range():
    from marseille_agenda.output import published_events
    from marseille_agenda.schema import State

    def ev(title, day, end=None, start=time(16, 0), venue="mucem"):
        e = _event(title, day, start)
        e.venue_id, e.source_kind, e.end_date = venue, "json", end
        e.uid = make_uid(venue, title, day)
        return e

    st = State()
    for e in [
        ev("Un objet, une histoire - En Ribambelle !", date(2026, 10, 24), end=date(2026, 10, 30)),  # the range
        ev("Un objet, une histoire - En Ribambelle !", date(2026, 10, 25)),  # daily copies inside the range: dropped
        ev("Un objet, une histoire - En Ribambelle !", date(2026, 10, 28)),
        ev("Un objet, une histoire - En Ribambelle !", date(2026, 10, 30)),  # last day of the range: dropped too
        ev("Un objet, une histoire - En Ribambelle !", date(2026, 10, 18)),  # before the range: kept
        ev("Un objet, une histoire - En Ribambelle !", date(2026, 11, 2)),  # after the range: kept
        ev("Un objet, une histoire - En Ribambelle !", date(2026, 10, 26), venue="friche"),  # other venue: kept
        ev("Visite flash", date(2026, 10, 26)),  # unrelated title on a covered day: kept
        ev("Liteul Pipol", date(2026, 10, 24), end=date(2026, 10, 26)),  # another range: ranges never fold each other
        ev("Visites Flash", date(2026, 9, 3), end=date(2026, 10, 11), start=time(14, 0)),  # weeks-long range...
        ev("Visites guidées en LSF de Mossi", date(2026, 9, 26), start=time(14, 0)),  # ...one shared word + same time: kept
    ]:
        st.events[e.uid] = e
    out = published_events(st)
    assert [(e.title, e.venue_id, e.start_date, e.end_date) for e in out] == [
        ("Visites Flash", "mucem", date(2026, 9, 3), date(2026, 10, 11)),
        ("Visites guidées en LSF de Mossi", "mucem", date(2026, 9, 26), None),
        ("Un objet, une histoire - En Ribambelle !", "mucem", date(2026, 10, 18), None),
        ("Liteul Pipol", "mucem", date(2026, 10, 24), date(2026, 10, 26)),
        ("Un objet, une histoire - En Ribambelle !", "mucem", date(2026, 10, 24), date(2026, 10, 30)),
        ("Un objet, une histoire - En Ribambelle !", "friche", date(2026, 10, 26), None),
        ("Visite flash", "mucem", date(2026, 10, 26), None),
        ("Un objet, une histoire - En Ribambelle !", "mucem", date(2026, 11, 2), None),
    ]
