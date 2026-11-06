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


def test_published_events_never_folds_two_cards_of_the_same_html_listing():
    from marseille_agenda.output import published_events
    from marseille_agenda.schema import State

    def build(kind: str) -> State:
        st = State()
        for title in ("L’Appart du Futur s’installe au Pop-up de La Fabulerie", "Le Pop-up La Fabulerie au Tiers lieu des Temps de l’enfant"):
            e = _event(title, date(2026, 9, 21), time(9, 0), url=None)
            e.venue_id, e.source_url, e.source_kind = "la-fabulerie", f"https://{kind}.test/", kind
            e.uid = make_uid("la-fabulerie", title, e.start_date)
            st.events[e.uid] = e
        return st

    assert len(published_events(build("html"))) == 2, "two cards of one listing are two events"
    assert len(published_events(build("facebook"))) == 1, "two posts about the same event still fold"


def test_published_events_folding_keeps_the_only_available_image():
    from marseille_agenda.output import published_events
    from marseille_agenda.schema import State

    def ev(title, kind, image_source=None, image=None):
        e = _event(title, date(2026, 9, 15), time(19, 45), url=None)
        e.source_kind, e.source_url, e.image_source, e.image = kind, f"https://{kind}.test/", image_source, image
        e.uid = make_uid("friche", title, e.start_date) + kind
        return e

    st = State()
    for e in [ev("Rebecca", "html"), ev("REBECCA, Alfred Hitchcock, 1940", "instagram", "https://cdn.test/flyer.jpg", "img/abc.webp")]:
        st.events[e.uid] = e
    out = published_events(st)
    assert len(out) == 1 and out[0].source_kind == "html"
    assert (out[0].image_source, out[0].image) == ("https://cdn.test/flyer.jpg", "img/abc.webp")

    st = State()
    for e in [ev("Rebecca", "html", "https://site.test/poster.jpg"), ev("REBECCA, Alfred Hitchcock, 1940", "instagram", "https://cdn.test/flyer.jpg")]:
        st.events[e.uid] = e
    assert published_events(st)[0].image_source == "https://site.test/poster.jpg", "the winner's own visual stays"


def test_merge_source_keeps_the_cached_image_while_the_source_url_is_unchanged():
    from marseille_agenda.merge import merge_source
    from marseille_agenda.schema import State

    st = State()
    day = date(2026, 9, 20)
    old = _event("Concert", day)
    old.image_source, old.image = "https://site.test/a.jpg", f"img/{old.uid}.webp"
    st.events[old.uid] = old

    same = _event("Concert", day)
    same.image_source = "https://site.test/a.jpg"
    merge_source(st, "https://x.test/agenda/", "friche", [same], day, "h1")
    assert st.events[old.uid].image == f"img/{old.uid}.webp"

    silent = _event("Concert", day)  # the page stopped showing a picture: keep what we had
    merge_source(st, "https://x.test/agenda/", "friche", [silent], day, "h2")
    assert (st.events[old.uid].image_source, st.events[old.uid].image) == ("https://site.test/a.jpg", f"img/{old.uid}.webp")

    changed = _event("Concert", day)  # a new poster: the cache is stale, download again
    changed.image_source = "https://site.test/b.jpg"
    merge_source(st, "https://x.test/agenda/", "friche", [changed], day, "h3")
    assert (st.events[old.uid].image_source, st.events[old.uid].image) == ("https://site.test/b.jpg", None)


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
        ev("Liteul Pipol", date(2026, 10, 24), end=date(2026, 10, 26)),  # another range with another title: kept
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


def test_continue_known_runs_keeps_the_first_day_of_a_run_the_listing_no_longer_shows():
    from marseille_agenda.merge import continue_known_runs

    def rng(title, start, end, url="https://x.test/e"):
        e = _event(title, start, url=url)
        e.end_date = end
        return e

    known = [rng("Unworlding", date(2026, 11, 4), date(2026, 11, 15)),
             rng("Unworlding", date(2026, 11, 5), date(2026, 11, 15)),  # a copy from the next run: the earliest wins
             rng("Entrez libres", date(2026, 11, 4), date(2026, 11, 7)),
             rng("Expo finie", date(2026, 10, 1), date(2026, 11, 2))]
    today = [_event("Unworlding", date(2026, 11, d)) for d in range(6, 16)]
    today += [_event("Entrez libres", date(2026, 11, d)) for d in (6, 7)]  # the last two days: not folded
    today += [_event("Entrez libres", date(2026, 11, d)) for d in (11, 12, 13, 14)]  # next week's session
    today += [_event("Unworlding", date(2026, 11, d), url="https://x.test/other") for d in range(6, 9)]
    today += [_event("Expo finie", date(2026, 11, d)) for d in range(6, 9)]  # more than 3 days after the end: new run
    out = continue_known_runs(collapse_daily_runs(today), known)
    assert sorted((e.title, e.url[-5:], e.start_date.day, (e.end_date or e.start_date).day) for e in out) == [
        ("Entrez libres", "est/e", 4, 7),
        ("Entrez libres", "est/e", 11, 14),
        ("Expo finie", "est/e", 6, 8),
        ("Unworlding", "est/e", 4, 15),
        ("Unworlding", "other", 6, 8),
    ]
    assert continue_known_runs(today[:1], []) == today[:1]


def test_published_events_folds_copies_of_one_run_with_different_first_days():
    from marseille_agenda.output import published_events
    from marseille_agenda.schema import State

    def ev(title, start, end, kind="html", venue="friche"):
        e = _event(title, start)
        e.venue_id, e.end_date, e.source_kind = venue, end, kind
        e.source_url = "https://x.test/agenda/" if kind == "html" else f"{kind}:friche"
        e.uid = make_uid(venue, title, start) + kind
        return e

    end = date(2026, 12, 13)
    st = State()
    for e in [
        ev("La vie farpaite", date(2026, 11, 5), end),
        ev("La vie farpaite", date(2026, 11, 4), end),  # same listing, folded a day earlier: the earliest stays
        ev("LA VIE FARPAITE 🤖", date(2026, 10, 24), end, kind="facebook"),  # announced from the opening: folded
        ev("La vie farpaite", date(2026, 11, 4), date(2026, 12, 6)),  # another last day: kept
        ev("La vie farpaite", date(2026, 11, 4), end, venue="mucem"),  # another venue: kept
        ev("Désobéir à l’ère numérique", date(2026, 11, 4), end),  # another title: kept
    ]:
        st.events[f"{e.uid}-{e.end_date}"] = e
    out = published_events(st)
    assert sorted((e.venue_id, e.title, e.source_kind, e.start_date.day, e.end_date.day) for e in out) == [
        ("friche", "Désobéir à l’ère numérique", "html", 4, 13),
        ("friche", "La vie farpaite", "html", 4, 6),
        ("friche", "La vie farpaite", "html", 4, 13),
        ("mucem", "La vie farpaite", "html", 4, 13),
    ]


def test_forget_events_drops_a_venue_by_source_kind_and_cleans_source_bookkeeping():
    from marseille_agenda.merge import forget_events
    from marseille_agenda.schema import SourceState, State

    web = _event("Concert", date(2026, 9, 19))
    insta = _event("Concert", date(2026, 9, 20))
    insta.source_kind = "instagram"
    insta.source_url = "instagram:friche"
    other = _event("Autre", date(2026, 9, 19))
    other.venue_id = "other"
    state = State(events={e.uid: e for e in (web, insta, other)})
    state.sources["https://x.test/agenda/"] = SourceState(url="https://x.test/agenda/", uids=[web.uid, other.uid],
                                                          missing_runs={web.uid: 2})
    assert forget_events(state, "friche", {"instagram", "facebook"}) == 1
    assert set(state.events) == {web.uid, other.uid}
    assert forget_events(state, "friche") == 1
    assert set(state.events) == {other.uid}
    src = state.sources["https://x.test/agenda/"]
    assert src.uids == [other.uid] and src.missing_runs == {}
