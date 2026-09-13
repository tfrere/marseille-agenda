"""Deterministic schema induction on real listing pages (offline fixtures, no model)."""

from datetime import date
from pathlib import Path

import pytest

from marseille_agenda import pipeline
from marseille_agenda.config import Settings
from marseille_agenda.extraction_schema import DiscoveredSource, FieldSpec, SourceRecord, SourcesFile
from marseille_agenda.fetch import document_from_body
from marseille_agenda.induce import induce_schema
from marseille_agenda.schema import State, Venue

LISTINGS = Path(__file__).parent / "fixtures" / "listings"

PAGES = {
    "molotov": ("molotov_agenda.html", "https://lemolotov.com/agenda/"),
    "gyptis": ("gyptis_seances.html", "https://cinemalegyptis.org/seances-et-evenements/"),
    "friche": ("friche_agenda.html", "https://www.lafriche.org/agenda/"),
    "transit": ("transit_evenements.html", "https://transit-librairie.org/-Evenements-"),
    "mac": ("mac_expositions.html", "https://musees.marseille.fr/liste-des-expositions?museum=musee-dart-contemporain-mac-0"),
    "millebabords": ("millebabords_agenda.html", "https://www.millebabords.org/spip.php?rubrique2"),
    "cira": ("cira_a_venir.html", "https://www.cira-marseille.info/2025/08/a-venir/"),
}


def _doc(key: str):
    name, url = PAGES[key]
    return document_from_body(url, "html", (LISTINGS / name).read_text(encoding="utf-8"))


def _titles(ind) -> set[str]:
    return set(ind.sample_titles)


def test_molotov_text_dates_inside_link_cards(today):
    ind = induce_schema(_doc("molotov"), today)
    assert ind is not None and ind.events == 30 and ind.coverage == 1.0
    fields = ind.schema.rules[0].fields
    assert fields["date"].selector == "div.eveDate" and "time" in fields and fields["url"].attr == "href"
    assert "AFTER MARY + BRIGHT SHOP + CASTAWAYS" in _titles(ind)


def test_gyptis_cards_carry_data_date_and_title_is_not_the_time(today):
    ind = induce_schema(_doc("gyptis"), today)
    assert ind is not None and ind.machine_dates and ind.events >= 30
    fields = ind.schema.rules[0].fields
    assert fields["date"].attr == "data-date" and fields["title"].selector == "h3.is-5.title a"
    assert not any(t.endswith("h") and t[:-1].isdigit() for t in ind.sample_titles), ind.sample_titles


def test_friche_repeated_exhibition_urls_still_yield_a_url_field(today):
    ind = induce_schema(_doc("friche"), today)
    assert ind is not None and ind.machine_dates and ind.events >= 100
    assert ind.schema.rules[0].fields["url"].attr == "href"


def test_transit_ignores_empty_placeholder_cards(today):
    ind = induce_schema(_doc("transit"), today)
    assert ind is not None and ind.events == 4 and ind.coverage == 1.0
    assert ind.schema.rules[0].fields["title"].selector == "h2 a"
    assert 'Présentation de "Le courage et la joie"' in _titles(ind)


def test_mac_exhibitions_with_two_time_elements_become_ranges(today):
    ind = induce_schema(_doc("mac"), today)
    assert ind is not None and ind.events == 3 and ind.machine_dates
    fields = ind.schema.rules[0].fields
    assert fields["date"].selector == "time:first-of-type" and fields["end_date"].selector == "time:last-of-type"
    # The museum name sits in a `.place` element on every card.
    assert fields["location"].selector == "div.museum-36.place"
    assert any("Triennale" in t for t in ind.sample_titles)


def test_image_field_is_induced_only_where_cards_carry_their_own_visual(today):
    from marseille_agenda.apply import apply_schema
    from marseille_agenda.extraction_schema import FieldSpec

    for key in ("friche", "gyptis", "mac"):
        fields = induce_schema(_doc(key), today).schema.rules[0].fields
        assert fields.get("image") == FieldSpec(selector="img", attr="src"), key
    # Transit: 3 of 15 cards show a book cover; the field would promise what most cards lack.
    ind = induce_schema(_doc("transit"), today)
    assert "image" not in ind.schema.rules[0].fields
    # ...but the engine fallback still gives those three their own cover.
    res = apply_schema(ind.schema, _doc("transit"), today)
    with_image = {e.title: e.image for e in res.events if e.image}
    assert len(with_image) == 3 and 'Présentation de "Le courage et la joie"' in with_image
    assert all(u.startswith("https://transit-librairie.org/local/adapt-img/") for u in with_image.values())
    # Molotov cards have no picture at all.
    assert "image" not in induce_schema(_doc("molotov"), today).schema.rules[0].fields


def test_amis_diplo_date_selector_covers_dated_cards_only(today, amis_diplo_doc):
    # Two of the five cards are plain articles (videos) whose prose mentions past dates.
    ind = induce_schema(amis_diplo_doc, today)
    assert ind is not None and ind.events == 3 and ind.coverage == 1.0
    assert ind.schema.rules[0].fields["date"].selector == "div.daterdv"


def test_flat_spip_listing_is_induced_in_run_mode(today):
    from marseille_agenda.apply import apply_schema

    # Mille Bâbords: no wrapper per event, a `div.evenement-date` starts each one.
    ind = induce_schema(_doc("millebabords"), today)
    assert ind is not None and ind.events == 23 and ind.quality >= 0.8
    rule = ind.schema.rules[0]
    assert rule.item_mode == "run" and rule.item_selector == "div.evenement-date"
    assert rule.fields["title"].selector == "div.donnee-titre" and rule.fields["url"] == FieldSpec(selector="div.donnee-titre a", attr="href")
    assert rule.fields["date"].selector == "div.evenement-date" and rule.fields["time"].selector == "div.donnee-heure"
    assert rule.fields["location"].selector == "div.donnee-lieu"
    assert any(t.startswith("Cycle mensuel de conférences de René Naba") for t in ind.sample_titles)
    res = apply_schema(ind.schema, _doc("millebabords"), today)
    assert sum(1 for e in res.events if e.location_name) >= 0.9 * len(res.events)
    assert res.events[0].start_date == date(2026, 9, 13) and res.events[0].start_time.hour == 14
    assert all(e.url and e.url.startswith("https://www.millebabords.org/spip.php?article") for e in res.events)


def test_run_mode_is_induced_from_headings_too(today):
    body = "<html><body><div class='agenda'>" + "".join(
        f"<h3>{d} septembre 2026</h3><p><a href='/e{d}'>Rendez-vous numéro {d}</a></p><hr>" for d in (15, 17, 21, 24)
    ) + "</div></body></html>"
    ind = induce_schema(document_from_body("https://x.test/", "html", body), today)
    assert ind is not None and ind.events == 4 and ind.schema.rules[0].item_mode == "run" and ind.schema.rules[0].item_selector == "h3"


def test_non_listing_pages_return_none(today):
    # CIRA lists two causeries in prose: nothing repeated to induce from.
    assert induce_schema(_doc("cira"), today) is None


def test_prose_page_returns_none(today):
    body = "<html><body><h1>Programme</h1><p>Le 20 septembre 2026 nous ouvrons. Le 25 septembre 2026 concert. Le 30 septembre 2026 fin.</p></body></html>"
    assert induce_schema(document_from_body("https://x.test/", "html", body), today) is None


def test_json_documents_are_not_induced(today):
    assert induce_schema(document_from_body("https://x.test/a.json", "json", "[]"), today) is None


async def test_pipeline_uses_induction_before_any_model(tmp_path, monkeypatch, today):
    name, url = PAGES["molotov"]
    body = (LISTINGS / name).read_text(encoding="utf-8")
    monkeypatch.setattr(pipeline, "fetch_document", lambda u, kind, client=None: document_from_body(u, kind, body))
    settings = Settings(
        openrouter_api_key=None, extractor_model="x", verifier_model="x", discover_model="x", search_model="x",
        vision_model="x", vision_verifier_model="x", apify_token=None,
        data_dir=tmp_path, venues_file=tmp_path / "venues.json",
    )
    venue = Venue(name="Le Molotov", category="bars", social=False)
    sources = SourcesFile(sources={venue.id: SourceRecord(
        venue_id=venue.id, source=DiscoveredSource(url=url, kind="html", confidence=0.9, reasoning="test"), discovered=today,
    )})
    state = State()
    report = await pipeline.Runner(settings, today, allow_llm=False).run([venue], sources, state)
    record = sources.sources[venue.id]
    assert report.llm_calls == 0 and record.schema_record is not None
    assert record.schema_record.generator_model == "induction" and not record.schema_record.validated_against_llm
    assert len([e for e in state.events.values() if e.venue_id == venue.id]) == 30


async def test_pipeline_induces_run_mode_and_applies_the_location_filter(tmp_path, monkeypatch, today):
    name, url = PAGES["millebabords"]
    body = (LISTINGS / name).read_text(encoding="utf-8")
    monkeypatch.setattr(pipeline, "fetch_document", lambda u, kind, client=None: document_from_body(u, kind, body))
    settings = Settings(
        openrouter_api_key=None, extractor_model="x", verifier_model="x", discover_model="x", search_model="x",
        vision_model="x", vision_verifier_model="x", apify_token=None,
        data_dir=tmp_path, venues_file=tmp_path / "venues.json",
    )
    venue = Venue(name="Mille Bâbords", category="militant", social=False, location_filter=r"mille b.bords|rue consolat")
    sources = SourcesFile(sources={venue.id: SourceRecord(
        venue_id=venue.id, source=DiscoveredSource(url=url, kind="html", confidence=0.9, reasoning="test"), discovered=today,
    )})
    state = State()
    report = await pipeline.Runner(settings, today, allow_llm=False, enrich=False).run([venue], sources, state)
    record = sources.sources[venue.id]
    assert report.llm_calls == 0 and record.schema_record.generator_model == "induction"
    assert record.schema_record.schema_.rules[0].item_mode == "run"
    mine = [e for e in state.events.values() if e.venue_id == venue.id]
    assert len(mine) == 6 and all("Consolat" in e.location_name for e in mine)
