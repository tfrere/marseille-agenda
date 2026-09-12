"""Tests that call real models through OpenRouter. Run with: uv run pytest -m live -s

They are the acceptance tests of the agentic parts, on frozen fixtures:
- the grounded extractor reads only the upcoming events of the Amis du Diplo page,
- the schema generator produces a validated schema for an HTML page and for a JSON API,
- the adversarial verifier rejects a past event dressed up as an upcoming one,
- the discovery agent finds a working source from a venue name alone (network).
"""

from datetime import date, time

import pytest

from marseille_agenda.apply import apply_schema
from marseille_agenda.config import load_settings
from marseille_agenda.discover import build_discoverer, discover_source
from marseille_agenda.extract import ExtractDeps, build_extractor, extract_events
from marseille_agenda.fetch import fetch_document
from marseille_agenda.llm import make_model
from marseille_agenda.schema import ExtractedEvent
from marseille_agenda.schema_gen import build_generator, generate_schema
from marseille_agenda.validate import check_event
from marseille_agenda.verify import build_verifier, verify_event

pytestmark = pytest.mark.live

settings = load_settings()
if not settings.has_llm:
    pytest.skip("OPENROUTER_API_KEY not set", allow_module_level=True)

EXPECTED_AMIS = {
    ("Café-Diplo à Marseille", date(2026, 9, 15), time(18, 30)),
    ("Café-Diplo à Marseille", date(2026, 10, 20), time(18, 30)),
    ("Iran. Guerres, révoltes, espérances", date(2026, 10, 28), time(19, 0)),
}


async def test_extractor_reads_only_upcoming(amis_diplo_doc, today):
    agent = build_extractor(make_model(settings, settings.extractor_model))
    deps = ExtractDeps(doc=amis_diplo_doc, today=today, venue_name="Amis du Monde diplomatique")
    run = await extract_events(agent, deps)
    got = {(e.title, e.start_date, e.start_time) for e in run.output.events}
    print("\nextractor:", got, "| rejected:", [(e.title, str(e.start_date), p) for e, p in deps.rejected])
    assert got == EXPECTED_AMIS
    for e in run.output.events:
        assert check_event(e, amis_diplo_doc, today) == []


async def test_generator_html(amis_diplo_doc, today):
    gen = build_generator(make_model(settings, settings.extractor_model))
    ext = build_extractor(make_model(settings, settings.extractor_model))
    out = await generate_schema(gen, ext, amis_diplo_doc, "Amis du Monde diplomatique", today)
    print("\nschema:", out.schema.model_dump_json(indent=1), "\nattempts:", out.attempts)
    assert out.validated, out.attempts
    res = apply_schema(out.schema, amis_diplo_doc, today)
    valid = [e for e in res.events if not check_event(e, amis_diplo_doc, today)]
    assert {(e.title, e.start_date, e.start_time) for e in valid} == EXPECTED_AMIS


async def test_generator_json(mucem_doc, today):
    gen = build_generator(make_model(settings, settings.extractor_model))
    ext = build_extractor(make_model(settings, settings.extractor_model))
    out = await generate_schema(gen, ext, mucem_doc, "Mucem", today)
    print("\nschema:", out.schema.model_dump_json(indent=1), "\nattempts:", out.attempts)
    assert out.validated, out.attempts
    res = apply_schema(out.schema, mucem_doc, today)
    valid = [e for e in res.events if not check_event(e, mucem_doc, today, check_text=False)]
    titles = {e.title for e in valid}
    assert "Mahmoud Darwich Poetry Day" in titles
    assert len(valid) >= 40


async def test_verifier_rejects_past_event_with_wrong_year(amis_diplo_doc, today):
    verifier = build_verifier(make_model(settings, settings.verifier_model))
    trap = ExtractedEvent(
        title="Café-Diplo à Marseille",
        start_date=date(2027, 3, 17),  # "mardi 17 mars" from the past-events section, pushed to next year
        start_time=time(18, 30),
        evidence=["Le mardi 17 mars à 18 h 30, au Grand Bar du Chapitre, 4, rue Consolat, Marseille 1er"],
    )
    run = await verify_event(verifier, trap, amis_diplo_doc, "Amis du Monde diplomatique", today)
    print("\ntrap verdict:", run.output)
    assert run.output.verdict != "accept"

    real = ExtractedEvent(
        title="Iran. Guerres, révoltes, espérances",
        start_date=date(2026, 10, 28),
        start_time=time(19, 0),
        location_name="librairie Transit",
        evidence=["Le mercredi 28 octobre, à 19 heures, à la librairie Transit, 51, Bd de la Libération à Marseille 1er"],
    )
    run = await verify_event(verifier, real, amis_diplo_doc, "Amis du Monde diplomatique", today)
    print("real verdict:", run.output)
    assert run.output.verdict == "accept"


@pytest.mark.parametrize("venue, must_contain", [
    ("Amis du Monde diplomatique", "amis.monde-diplomatique.fr"),
    ("Mucem", "mucem.org"),
])
async def test_discovery_finds_a_working_source(venue, must_contain, today):
    agent = build_discoverer(make_model(settings, settings.discover_model))
    run = await discover_source(agent, settings, venue, None, today)
    src = run.output
    print(f"\n{venue}: {src.url} ({src.kind}, {src.confidence:.2f}) {src.reasoning}\n  samples: {src.sample_titles}\n  requests: {run.usage.requests}")
    assert must_contain in src.url
    doc = fetch_document(src.url, src.kind)
    assert len(doc.text) > 500


# ------------------------------------------------------------------ social posts (vision models)


def _flyer_post(text: str, published: date) -> "SocialPost":
    from datetime import datetime, time as _time, timezone

    from marseille_agenda.apify import SocialPost

    return SocialPost(platform="instagram", id="live-1", url="https://www.instagram.com/p/live/", text=text,
                      published=datetime.combine(published, _time(12, 0), tzinfo=timezone.utc),
                      image_urls=["https://cdn.example.test/flyer.jpg"])


def _flyer_image():
    from pathlib import Path

    from pydantic_ai import BinaryContent

    data = (Path(__file__).parent / "fixtures" / "social" / "flyer.jpg").read_bytes()
    return BinaryContent(data=data, media_type="image/jpeg")


async def test_vision_extractor_reads_date_and_time_from_flyer(today):
    """Caption says nothing useful: title, weekday, day, month and time are only on the flyer.
    The year is not written anywhere and must be pinned by the publication date."""
    from marseille_agenda.social_extract import PostDeps, build_post_extractor, check_post_event, extract_post

    agent = build_post_extractor(make_model(settings, settings.vision_model))
    post = _flyer_post("On vous attend nombreux et nombreuses ! Infos en image.", date(2026, 9, 5))
    deps = PostDeps(post=post, today=today, venue_name="Manifesten", images=[_flyer_image()])
    run = await extract_post(agent, deps)
    got = [(e.title, e.start_date, e.start_time, e.from_image, e.price) for e in run.output.events]
    print("\nvision extractor:", got, "| notes:", run.output.notes, "| rejected:", [(e.title, p) for e, p in deps.rejected],
          "| requests:", run.usage.requests, "| tokens in:", run.usage.input_tokens)
    assert len(run.output.events) == 1
    ev = run.output.events[0]
    assert ev.start_date == date(2026, 9, 25) and ev.start_time == time(19, 0)
    assert "soutien" in ev.title.lower() or "fauves" in ev.title.lower()
    assert ev.from_image is True
    assert check_post_event(ev, post, today, has_images=True) == []


async def test_vision_verifier_rejects_recap_of_past_event(today):
    """The same flyer re-posted after the event with a thank-you caption is not an announcement."""
    from marseille_agenda.social_extract import PostEvent, build_post_verifier, verify_post_event

    agent = build_post_verifier(make_model(settings, settings.vision_verifier_model))
    post = _flyer_post("Merci à toutes et tous pour ce concert de soutien hier soir, la salle était pleine ! On remet ça bientôt.",
                       date(2026, 9, 26))
    ev = PostEvent(title="Concert de soutien - Les Fauves Sonores", start_date=date(2027, 9, 25), start_time=time(19, 0),
                   evidence=["Vendredi 25 sept. 19h"], from_image=True)
    run = await verify_post_event(agent, ev, post, [_flyer_image()], "Manifesten", date(2026, 9, 27))
    print("\nvision verifier:", run.output.verdict, "-", run.output.reason)
    assert run.output.verdict in ("reject", "uncertain")
    assert run.output.verdict == "reject"


async def test_vision_verifier_accepts_consistent_announcement(today):
    from marseille_agenda.social_extract import PostEvent, build_post_verifier, verify_post_event

    agent = build_post_verifier(make_model(settings, settings.vision_verifier_model))
    post = _flyer_post("Vendredi 25 septembre : concert de soutien avec Les Fauves Sonores, 19h, prix libre !", date(2026, 9, 5))
    ev = PostEvent(title="concert de soutien avec Les Fauves Sonores", start_date=date(2026, 9, 25), start_time=time(19, 0),
                   price="prix libre", evidence=["Vendredi 25 septembre : concert de soutien avec Les Fauves Sonores, 19h"])
    run = await verify_post_event(agent, ev, post, [_flyer_image()], "Manifesten", today)
    print("\nvision verifier:", run.output.verdict, "-", run.output.reason)
    assert run.output.verdict == "accept"
