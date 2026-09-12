"""The declarative engine on real fixtures, with hand-written reference schemas.

These schemas are what the generator agent is expected to produce (or something
equivalent). They double as regression tests for the engine itself.
"""

from datetime import date, time

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
