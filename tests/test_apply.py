"""The declarative engine on real fixtures, with hand-written reference schemas.

These schemas are what the generator agent is expected to produce (or something
equivalent). They double as regression tests for the engine itself.
"""

import json
from datetime import date, time
from pathlib import Path

from marseille_agenda.apply import apply_schema
from marseille_agenda.extraction_schema import ExtractionSchema, FieldSpec, HtmlRule, JsonRule
from marseille_agenda.validate import check_event

AMIS_DIPLO_SCHEMA = ExtractionSchema(
    rules=[
        HtmlRule(
            container_selector="div.menu.rendezvous",
            exclude_selectors=[],
            item_selector="li.hentry",
            fields={
                "title": FieldSpec(selector="h3.entry-title a"),
                "date": FieldSpec(selector="div.introduction"),
                "url": FieldSpec(selector="h3.entry-title a", attr="href"),
                "location": FieldSpec(selector="div.introduction", regex=r"\b(?:au|à la|à l'|au bar|à)\s+((?:Grand Bar|librairie|Théâtre|Brasserie)[^,]+)"),
                "summary": FieldSpec(selector="div.introduction"),
            },
            notes="Upcoming events live under 'Prochain(s) rendez-vous'; past ones are in a different block.",
        )
    ],
    notes="SPIP site of Les Amis du Monde diplomatique, Marseille group page.",
)

MUCEM_SCHEMA = ExtractionSchema(
    rules=[
        JsonRule(
            items_path="list",
            expand_path="meta.sessions.event__unique",
            fields={
                "title": FieldSpec(selector="item.post_title"),
                "date": FieldSpec(selector="date"),
                "time": FieldSpec(selector="start_time"),
                "end_time": FieldSpec(selector="end_time"),
                "url": FieldSpec(selector="item.link"),
                "location": FieldSpec(selector="item.meta.practical_info.location.post_title", join=" / "),
                "event_type": FieldSpec(selector="item.taxonomies.event_type.name", join=", "),
                "free": FieldSpec(selector="item.meta.free_event"),
            },
            date_format="%Y%m%d",
            time_format="%H:%M",
            notes="One event per dated session. Recurring events (event__recurrent) handled by a second rule.",
        ),
        JsonRule(
            items_path="list",
            fields={
                "title": FieldSpec(selector="post_title"),
                "date": FieldSpec(selector="meta.sessions.event__recurrent.start_date"),
                "end_date": FieldSpec(selector="meta.sessions.event__recurrent.end_date"),
                "time": FieldSpec(selector="meta.sessions.event__recurrent.start_time"),
                "end_time": FieldSpec(selector="meta.sessions.event__recurrent.end_time"),
                "url": FieldSpec(selector="link"),
                "location": FieldSpec(selector="meta.practical_info.location.post_title", join=" / "),
                "free": FieldSpec(selector="meta.free_event"),
            },
            date_format="%Y%m%d",
            time_format="%H:%M",
            notes="Recurring events: items without event__recurrent simply fail the date field and are skipped.",
        ),
    ],
    notes="Mucem WordPress API behind the Nuxt front.",
)


def test_amis_diplo_schema_extracts_only_upcoming(amis_diplo_doc, today):
    res = apply_schema(AMIS_DIPLO_SCHEMA, amis_diplo_doc, today)
    assert res.items_seen == 3 and not res.failures
    got = {(e.title, e.start_date, e.start_time) for e in res.events}
    assert got == {
        ("Café-Diplo à Marseille", date(2026, 9, 15), time(18, 30)),
        ("Café-Diplo à Marseille", date(2026, 10, 20), time(18, 30)),
        ("Iran. Guerres, révoltes, espérances", date(2026, 10, 28), time(19, 0)),
    }
    iran = next(e for e in res.events if e.title.startswith("Iran"))
    assert iran.location_name.startswith("librairie Transit")
    assert iran.url == "https://www.amis.monde-diplomatique.fr/Iran-Guerres-revoltes-esperances.html"
    # Every event passes the same grounding checks as LLM output would.
    for e in res.events:
        assert check_event(e, amis_diplo_doc, today) == []


def test_amis_diplo_without_container_catches_past_events(amis_diplo_doc, today):
    """Sanity check of the trap: a lazy schema picks up 'Rendez-vous passés' too."""
    lazy = AMIS_DIPLO_SCHEMA.model_copy(deep=True)
    lazy.rules[0].container_selector = None
    res = apply_schema(lazy, amis_diplo_doc, today)
    assert res.items_seen > 3
    past = [e for e in res.events if e.start_date < today]
    assert past, "past events must parse to past dates (not be pushed into next year)"
    # ...and the deterministic date check rejects each of them.
    for e in past:
        assert any("past" in p for p in check_event(e, amis_diplo_doc, today))


def test_html_date_selector_catching_only_the_time_falls_back_to_item_text(today):
    """Seen on lemolotov.com: the generator pointed `date` at the `.time` span ('19:30')."""
    from marseille_agenda.fetch import document_from_body

    html = """<ul class="events">
      <li class="event"><span class="date">13 Sep 2026</span> @ Le Molotov / <span class="time">19:30</span>
        <h3><a href="https://x.test/e/1">AFTER MARY + BRIGHT SHOP</a></h3></li>
      <li class="event"><span class="date">17 Sep 2026</span> @ Le Molotov / <span class="time">20:30</span>
        <h3><a href="https://x.test/e/2">THEN COMES SILENCE</a></h3></li>
    </ul>"""
    doc = document_from_body("https://x.test/agenda/", "html", html)
    schema = ExtractionSchema(rules=[HtmlRule(
        item_selector="li.event",
        fields={"title": FieldSpec(selector="h3 a"), "date": FieldSpec(selector="span.time")},
    )])
    res = apply_schema(schema, doc, today)
    assert not res.failures
    assert {(e.title, e.start_date, e.start_time) for e in res.events} == {
        ("AFTER MARY + BRIGHT SHOP", date(2026, 9, 13), time(19, 30)),
        ("THEN COMES SILENCE", date(2026, 9, 17), time(20, 30)),
    }


def test_html_date_falls_back_to_data_date_attribute_when_card_shows_only_the_time(today):
    """Seen on lafriche.org: cards grouped by day carry data-date, their text only has '14h-19h'."""
    from marseille_agenda.fetch import document_from_body

    html = """<div class="events" data-date="2026-09-13">
      <h2>Aujourd'hui, dimanche 13 septembre</h2>
      <div class="event"><h3><a href="https://x.test/e/1">Expo-pause</a></h3><span class="hours">14h-19h</span></div>
      <div class="event" data-date="2026-09-14"><h3><a href="https://x.test/e/2">Marché paysan</a></h3><span class="hours">11h</span></div>
    </div>"""
    doc = document_from_body("https://x.test/agenda/", "html", html)
    schema = ExtractionSchema(rules=[HtmlRule(
        item_selector="div.event",
        fields={"title": FieldSpec(selector="h3 a"), "date": FieldSpec(selector="span.hours")},
    )])
    res = apply_schema(schema, doc, today)
    assert not res.failures
    expected = {
        ("Expo-pause", date(2026, 9, 13), time(14, 0), time(19, 0)),
        ("Marché paysan", date(2026, 9, 14), time(11, 0), None),
    }
    assert {(e.title, e.start_date, e.start_time, e.end_time) for e in res.events} == expected

    # Same page, generator variant seen on cinemalegyptis.org: a `time` field on the hours span
    # and a `date` selector matching nothing.
    schema = ExtractionSchema(rules=[HtmlRule(
        item_selector="div.event",
        fields={"title": FieldSpec(selector="h3 a"), "date": FieldSpec(selector=".nope"), "time": FieldSpec(selector="span.hours")},
    )])
    res = apply_schema(schema, doc, today)
    assert not res.failures
    assert {(e.title, e.start_date, e.start_time, e.end_time) for e in res.events} == expected


def test_mucem_schema(mucem_doc, today):
    res = apply_schema(MUCEM_SCHEMA, mucem_doc, today)
    titles = {e.title for e in res.events}
    assert "Mahmoud Darwich Poetry Day" in titles
    darwich = next(e for e in res.events if e.title == "Mahmoud Darwich Poetry Day")
    assert (darwich.start_date, darwich.start_time, darwich.end_time) == (date(2026, 9, 25), time(18, 30), time(21, 30))
    assert darwich.location_name == "J4 / Auditorium / J4 / Forum"
    assert darwich.url == "https://mucem.org/evenement/mahmoud-darwich-poetry-day/"
    assert darwich.event_type == "Conférence, Lecture, Table-ronde"
    assert darwich.free is False
    # Sessions in the past (e.g. 'Dimanche entrée libre' 2025 dates) are parsed but flagged by validation.
    flash = [e for e in res.events if e.title == "Visites Flash"]
    assert flash and flash[0].end_date == date(2026, 10, 11) and flash[0].free is True
    upcoming = [e for e in res.events if not check_event(e, mucem_doc, today, check_text=False)]
    assert len(upcoming) >= 60
    assert all(e.start_date >= today or (e.end_date and e.end_date >= today) for e in upcoming)


def test_truncated_titles_lose_their_marker_and_still_ground(today):
    """Seen on lafabulerie.com: the home cuts titles ('Le Grand Voyage aux Fertiles...')."""
    from marseille_agenda.fetch import document_from_body

    html = """<div class="list">
      <a class="card" href="/evenement/le-grand-voyage-aux-fertiles/"><div class="date">16/09/2026 - 08h30</div><h3>Le Grand Voyage aux Fertiles...</h3></a>
      <a class="card" href="/evenement/rencontres/"><div class="date">21/09/2026 - 14h00</div><h3>Rencontres partenaires :&nbsp; La…</h3></a>
      <a class="card" href="/evenement/atelier/"><div class="date">03/10/2026 - 10h00</div><h3>Atelier complet</h3></a>
    </div>"""
    doc = document_from_body("https://x.test/", "html", html)
    schema = ExtractionSchema(rules=[HtmlRule(item_selector="a.card", fields={
        "title": FieldSpec(selector="h3"), "date": FieldSpec(selector="div.date"), "url": FieldSpec(selector="", attr="href"),
    })])
    res = apply_schema(schema, doc, today)
    assert not res.failures
    assert [(e.title, e.title_truncated, e.start_date, e.start_time) for e in res.events] == [
        ("Le Grand Voyage aux Fertiles", True, date(2026, 9, 16), time(8, 30)),
        ("Rencontres partenaires :\xa0 La", True, date(2026, 9, 21), time(14, 0)),  # verbatim prefix, never completed
        ("Atelier complet", False, date(2026, 10, 3), time(10, 0)),
    ]
    assert res.events[0].url == "https://x.test/evenement/le-grand-voyage-aux-fertiles/"
    for e in res.events:
        assert check_event(e, doc, today) == [], e.title


# ------------------------------------------------------------------ images

LISTINGS = Path(__file__).parent / "fixtures" / "listings"


def _listing(name: str, url: str):
    from marseille_agenda.fetch import document_from_body

    return document_from_body(url, "html", (LISTINGS / name).read_text(encoding="utf-8"))


def test_html_image_field_and_engine_fallback_agree_on_friche_and_gyptis(today):
    """With an explicit `image` field or without one, each card's poster is found."""
    from marseille_agenda.induce import induce_schema

    for name, url in (("friche_agenda.html", "https://www.lafriche.org/agenda/"),
                      ("gyptis_seances.html", "https://cinemalegyptis.org/seances-et-evenements/")):
        doc = _listing(name, url)
        schema = induce_schema(doc, today).schema
        explicit = schema.rules[0].fields
        assert explicit["image"] == FieldSpec(selector="img", attr="src")
        with_field = apply_schema(schema, doc, today)
        silent = schema.model_copy(deep=True)
        del silent.rules[0].fields["image"]
        fallback = apply_schema(silent, doc, today)
        got = [e.image for e in with_field.events]
        assert got and all(got), name
        assert all(u.startswith("https://") and not u.endswith(".svg") for u in got)
        assert got == [e.image for e in fallback.events], name
    # A poster repeated on every screening of the same film is still that film's poster.
    imgs = [e.image for e in with_field.events]
    assert len(set(imgs)) == len({e.title for e in with_field.events})


def test_shared_image_rule_drops_the_venue_logo_but_keeps_the_own_poster(today):
    from marseille_agenda.fetch import document_from_body

    cards = "".join(
        f'<li class="event"><img src="/static/logo.png" width="300" height="100"><h3><a href="/e/{i}">Event {i}</a></h3>'
        f'<span class="d">{13 + i} Sep 2026</span></li>'
        for i in range(4)
    )
    cards += ('<li class="event"><img src="/uploads/poster.jpg"><h3><a href="/e/9">Own poster</a></h3>'
              '<span class="d">20 Sep 2026</span></li>')
    doc = document_from_body("https://x.test/agenda/", "html", f"<ul>{cards}</ul>")
    for fields in (
        {"title": FieldSpec(selector="h3 a"), "date": FieldSpec(selector="span.d")},
        {"title": FieldSpec(selector="h3 a"), "date": FieldSpec(selector="span.d"), "image": FieldSpec(selector="img", attr="src")},
    ):
        res = apply_schema(ExtractionSchema(rules=[HtmlRule(item_selector="li.event", fields=fields)]), doc, today)
        assert not res.failures and len(res.events) == 5
        by_title = {e.title: e.image for e in res.events}
        assert by_title["Own poster"] == "https://x.test/uploads/poster.jpg"
        assert all(by_title[f"Event {i}"] is None for i in range(4)), by_title


def test_lazy_loading_attributes_srcset_and_placeholders(today):
    from marseille_agenda.fetch import document_from_body

    html = """<ul>
      <li class="event"><img class="lazy" src="data:image/gif;base64,R0lGOD" data-src="/img/a.jpg"><h3>A</h3><span>13 Sep 2026</span></li>
      <li class="event"><img srcset="/img/b-480.jpg 480w, /img/b-960.jpg 960w"><h3>B</h3><span>14 Sep 2026</span></li>
      <li class="event"><picture><source srcset="/img/c.webp" type="image/webp"><img src="/pixel.gif" width="1" height="1"></picture>
          <img data-lazy-src="/img/c2.jpg"><h3>C</h3><span>15 Sep 2026</span></li>
      <li class="event"><img src="/icons/calendar.svg"><img src="/img/d.png?v=2"><h3>D</h3><span>16 Sep 2026</span></li>
      <li class="event"><img src="/icons/star.png" width="16" height="16"><h3>E</h3><span>17 Sep 2026</span></li>
    </ul>"""
    doc = document_from_body("https://x.test/agenda/", "html", html)
    schema = ExtractionSchema(rules=[HtmlRule(item_selector="li.event", fields={"title": FieldSpec(selector="h3"), "date": FieldSpec(selector="span")})])
    res = apply_schema(schema, doc, today)
    assert {e.title: e.image for e in res.events} == {
        "A": "https://x.test/img/a.jpg",
        "B": "https://x.test/img/b-480.jpg",
        "C": "https://x.test/img/c2.jpg",  # the 1x1 pixel is skipped, its <picture> too; the next picture wins
        "D": "https://x.test/img/d.png?v=2",  # SVG icons are never posters
        "E": None,  # a 16x16 icon is not a visual
    }


def test_json_image_fallback_prefers_a_medium_rendition_on_mucem(mucem_doc, today):
    res = apply_schema(MUCEM_SCHEMA, mucem_doc, today)
    darwich = next(e for e in res.events if e.title == "Mahmoud Darwich Poetry Day")
    assert darwich.image and darwich.image.startswith("https://mucem.org/uploads/") and "-768x480" in darwich.image
    with_image = [e for e in res.events if e.image]
    assert len(with_image) == len(res.events)
    # One visual per event, not one per session, and no site-wide default picture.
    assert len({e.image for e in with_image}) >= 30


def test_json_image_walk_takes_image_keys_and_extensions_and_skips_shared_ones(today):
    from marseille_agenda.fetch import document_from_body

    items = [
        {"name": "A", "when": "2026-09-13", "visuel": "https://x.test/a.jpg"},
        {"name": "B", "when": "2026-09-14", "media": {"cover": {"sizes": {"medium": "https://x.test/b-m.jpg", "large": "https://x.test/b-l.jpg"}}}},
        {"name": "C", "when": "2026-09-15", "attachments": [{"file": "https://x.test/c.png"}]},
        {"name": "D", "when": "2026-09-16", "cover_color": "#ffffff", "logo": "https://x.test/logo.svg"},
    ]
    doc = document_from_body("https://x.test/api", "json", json.dumps(items))
    schema = ExtractionSchema(rules=[JsonRule(items_path="", fields={"title": FieldSpec(selector="name"), "date": FieldSpec(selector="when")})])
    res = apply_schema(schema, doc, today)
    assert {e.title: e.image for e in res.events} == {
        "A": "https://x.test/a.jpg", "B": "https://x.test/b-m.jpg", "C": "https://x.test/c.png", "D": None,
    }
    shared = [{"name": f"E{i}", "when": f"2026-09-2{i}", "thumbnail": "https://x.test/default.jpg"} for i in range(3)]
    res = apply_schema(schema, document_from_body("https://x.test/api", "json", json.dumps(shared)), today)
    assert [e.image for e in res.events] == [None, None, None]
