"""Pipeline behaviour without any LLM: schemas already known, fetch replaced by fixtures."""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from marseille_agenda import pipeline
from marseille_agenda.config import Settings
from marseille_agenda.extraction_schema import DiscoveredSource, SchemaRecord, SourceRecord, SourcesFile
from marseille_agenda.output import load_state, write_events_json, write_ics
from marseille_agenda.schema import State, Venue
from tests.test_apply import AMIS_DIPLO_SCHEMA, MUCEM_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        openrouter_api_key=None, extractor_model="x", verifier_model="x", discover_model="x", search_model="x",
        vision_model="x", vision_verifier_model="x", apify_token=None,
        data_dir=tmp_path, venues_file=tmp_path / "venues.json",
    )


def _sources(today: date) -> SourcesFile:
    return SourcesFile(sources={
        "amis-du-monde-diplomatique": SourceRecord(
            venue_id="amis-du-monde-diplomatique",
            source=DiscoveredSource(url="https://www.amis.monde-diplomatique.fr/-Marseille-.html", kind="html", confidence=0.9, reasoning="test"),
            discovered=today,
            schema_record=SchemaRecord(schema=AMIS_DIPLO_SCHEMA, created=today, validated_against_llm=True, agreement=1.0, generator_model="test"),
        ),
        "mucem": SourceRecord(
            venue_id="mucem",
            source=DiscoveredSource(url="https://mucem.org/api/mainApi/posts/evenement?upcoming=1&perPage=100", kind="json", confidence=0.9, reasoning="test"),
            discovered=today,
            schema_record=SchemaRecord(schema=MUCEM_SCHEMA, created=today, validated_against_llm=True, agreement=1.0, generator_model="test"),
        ),
    })


@pytest.fixture
def offline_fetch(monkeypatch):
    bodies = {
        "https://www.amis.monde-diplomatique.fr/-Marseille-.html": (FIXTURES / "amis_diplo" / "marseille.html").read_text(encoding="utf-8"),
        "https://mucem.org/api/mainApi/posts/evenement?upcoming=1&perPage=100": (FIXTURES / "mucem" / "evenement_upcoming.json").read_text(encoding="utf-8"),
    }

    def fake_fetch(url, kind, client=None):
        from marseille_agenda.fetch import document_from_body
        if url not in bodies:
            raise ConnectionError(f"offline: {url}")
        return document_from_body(url, kind, bodies[url])

    monkeypatch.setattr(pipeline, "fetch_document", fake_fetch)
    return bodies


VENUES = [Venue(name="Amis du Monde diplomatique", category="milite"), Venue(name="Mucem", category="culture")]


async def test_full_run_offline(tmp_path, offline_fetch, today):
    settings = _settings(tmp_path)
    sources, state = _sources(today), State()
    runner = pipeline.Runner(settings, today, verify=True, allow_llm=False)
    report = await runner.run(VENUES, sources, state)

    assert report.sources_ok == 2 and report.llm_calls == 0
    by_venue = {}
    for e in state.events.values():
        by_venue.setdefault(e.venue_id, []).append(e)
    assert len(by_venue["amis-du-monde-diplomatique"]) == 3
    assert len(by_venue["mucem"]) >= 60
    assert all(e.start_date >= today or (e.end_date and e.end_date >= today) for e in state.events.values())
    # HTML events without a verifier verdict are still published when the LLM is disabled...
    assert all(e.verifier is None for e in by_venue["amis-du-monde-diplomatique"])
    # ...and the health tracking is updated.
    assert sources.sources["mucem"].last_ok == today and sources.sources["mucem"].consecutive_failures == 0

    write_events_json(tmp_path / "events.json", state, VENUES, today)
    write_ics(tmp_path / "events.ics", state)
    payload = json.loads((tmp_path / "events.json").read_text())
    assert payload["events"][0]["start_date"] <= payload["events"][-1]["start_date"]
    assert {v["id"] for v in payload["venues"]} == {"amis-du-monde-diplomatique", "mucem"}
    assert b"BEGIN:VEVENT" in (tmp_path / "events.ics").read_bytes()


async def test_second_run_is_stable_and_keeps_first_seen(tmp_path, offline_fetch, today):
    settings = _settings(tmp_path)
    sources, state = _sources(today), State()
    await pipeline.Runner(settings, today, allow_llm=False).run(VENUES, sources, state)
    uids_1 = set(state.events)
    tomorrow = today + timedelta(days=1)
    report = await pipeline.Runner(settings, tomorrow, allow_llm=False).run(VENUES, sources, state)
    assert report.events_new == 0
    assert set(state.events) <= uids_1  # same uids (minus anything that expired overnight)
    assert all(e.first_seen == today and e.last_seen == tomorrow for e in state.events.values())


async def test_fetch_failure_keeps_previous_state(tmp_path, offline_fetch, today, monkeypatch):
    settings = _settings(tmp_path)
    sources, state = _sources(today), State()
    await pipeline.Runner(settings, today, allow_llm=False).run(VENUES, sources, state)
    tomorrow = today + timedelta(days=1)
    still_upcoming = {uid for uid, e in state.events.items() if (e.end_date or e.start_date) >= tomorrow}

    def broken(url, kind, client=None):
        raise ConnectionError("boom")

    monkeypatch.setattr(pipeline, "fetch_document", broken)
    report = await pipeline.Runner(settings, tomorrow, allow_llm=False).run(VENUES, sources, state)
    assert report.sources_ok == 0
    assert set(state.events) == still_upcoming, "a failed fetch must never wipe the published agenda"
    assert all(a.level == "error" for a in report.alerts if "kept" in a.message)
    assert sources.sources["mucem"].consecutive_failures == 1


async def test_broken_schema_without_llm_keeps_state(tmp_path, offline_fetch, today):
    settings = _settings(tmp_path)
    sources, state = _sources(today), State()
    await pipeline.Runner(settings, today, allow_llm=False).run(VENUES, sources, state)
    n_amis = sum(1 for e in state.events.values() if e.venue_id == "amis-du-monde-diplomatique")
    # Simulate a site redesign: the container selector no longer matches.
    sources.sources["amis-du-monde-diplomatique"].schema_record.schema_.rules[0].container_selector = "div.does-not-exist"
    report = await pipeline.Runner(settings, today + timedelta(days=1), allow_llm=False).run(VENUES, sources, state)
    assert sum(1 for e in state.events.values() if e.venue_id == "amis-du-monde-diplomatique") == n_amis
    assert sources.sources["amis-du-monde-diplomatique"].consecutive_failures == 1
    assert any("unhealthy" in a.message for a in report.alerts)


async def test_empty_source_is_not_an_error_and_is_retried_later(tmp_path, offline_fetch, today, monkeypatch):
    """A discovered source that lists nothing upcoming (reader agrees) yields 0 events without
    alerts, and generation is postponed instead of retried (and billed) every day."""
    settings = _settings(tmp_path)
    venue = Venue(name="Manifesten", category="milite")
    url = "https://www.amis.monde-diplomatique.fr/-Marseille-.html"  # any fetchable body will do
    sources = SourcesFile(sources={venue.id: SourceRecord(
        venue_id=venue.id, source=DiscoveredSource(url=url, kind="html", confidence=0.4, reasoning="test"), discovered=today,
    )})
    state = State()
    calls = []

    async def fake_generate(self, v, record, doc, out):
        calls.append(self.today)
        record.next_generation = self.today + timedelta(days=pipeline.EMPTY_SOURCE_RETRY_DAYS)
        return "empty"

    monkeypatch.setattr(pipeline.Runner, "generate", fake_generate)
    report = await pipeline.Runner(settings, today, allow_llm=True).run([venue], sources, state)
    assert report.sources_ok == 1 and not report.alerts and not state.events
    rec = sources.sources[venue.id]
    assert rec.schema_record is None and rec.last_error is None and rec.consecutive_failures == 0

    # Next day: no generation attempt at all.
    report = await pipeline.Runner(settings, today + timedelta(days=1), allow_llm=True).run([venue], sources, state)
    assert report.sources_ok == 1 and len(calls) == 1
    # After the retry delay: generation is attempted again.
    await pipeline.Runner(settings, today + timedelta(days=pipeline.EMPTY_SOURCE_RETRY_DAYS), allow_llm=True).run([venue], sources, state)
    assert len(calls) == 2


def test_state_roundtrip(tmp_path, offline_fetch, today):
    import asyncio

    settings = _settings(tmp_path)
    sources, state = _sources(today), State()
    asyncio.run(pipeline.Runner(settings, today, allow_llm=False).run(VENUES, sources, state))
    from marseille_agenda.output import save_state
    save_state(tmp_path / "state.json", state)
    assert load_state(tmp_path / "state.json").events.keys() == state.events.keys()
    pipeline.save_sources(settings, sources)
    reloaded = pipeline.load_sources(settings)
    assert reloaded.sources["mucem"].schema_record.schema_.rules[0].kind == "json"
