"""Instagram / Facebook sources without network: Apify and the vision model are replaced by fakes."""

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from marseille_agenda import apify, pipeline, social
from marseille_agenda.apify import SocialPost
from marseille_agenda.config import Settings
from marseille_agenda.extraction_schema import DiscoveredSource, SourceRecord, SourcesFile
from marseille_agenda.schema import State, Venue
from marseille_agenda.social import SocialFile, SocialSource, StoredEvent, PostRecord, fb_event_to_schema, rebuild
from marseille_agenda.social_extract import PostEvent, PostExtraction, check_post_event

FIXTURES = Path(__file__).parent / "fixtures" / "social"


def _items(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ normalizers


def test_instagram_items_are_normalized_and_item_errors_dropped():
    posts = apify.instagram_posts([it for it in _items("instagram_posts.json") if not it.get("error")])
    assert [p.id for p in posts] == ["3456789012345678901", "3456789012345678902"]
    first = posts[0]
    assert first.platform == "instagram"
    assert first.published_date == date(2026, 9, 5)
    assert first.url == "https://www.instagram.com/p/DAbCdEfGhIj/"
    # displayUrl, images and child posts merged and de-duplicated, order kept
    assert first.image_urls == ["https://cdn.example.test/flyer-concert.jpg", "https://cdn.example.test/flyer-concert-2.jpg"]
    assert "Vendredi 25 septembre" in first.text


def test_facebook_posts_prefer_full_photo_uri():
    posts = apify.facebook_posts(_items("facebook_posts.json"))
    assert len(posts) == 1
    assert posts[0].image_urls == ["https://cdn.example.test/fb-flyer-full.jpg"]
    assert posts[0].published_date == date(2026, 9, 8)


def test_facebook_events_normalized():
    events = apify.facebook_events(_items("facebook_events.json"))
    assert [e.id for e in events] == ["1269179411830316", "1559468378412380", "1111"]
    ag = events[0]
    assert ag.location_name == "Manifesten" and ag.start == datetime(2026, 10, 1, 17, 0, tzinfo=timezone.utc)
    assert events[1].cancelled and events[1].price == "5 EUR" and events[1].ticket_url == "https://example.test/billetterie"
    assert events[2].is_past


def test_handles_to_urls():
    assert apify.instagram_url("@manifesten") == "https://www.instagram.com/manifesten/"
    assert apify.instagram_url("https://instagram.com/manifesten?hl=fr") == "https://www.instagram.com/manifesten/"
    assert apify.facebook_url("manifesten") == "https://www.facebook.com/manifesten/"
    assert apify.facebook_url("https://www.facebook.com/Amis.Monde.diplomatique/events") == "https://www.facebook.com/Amis.Monde.diplomatique/"


# ------------------------------------------------------------------ structured Facebook events


def test_fb_event_to_schema_converts_to_paris_time_and_handles_duration_and_cancellation():
    today = date(2026, 9, 12)
    events = apify.facebook_events(_items("facebook_events.json"))
    ag = fb_event_to_schema(events[0], today)
    assert ag is not None
    assert ag.start_date == date(2026, 10, 1) and ag.start_time == time(19, 0)  # 17:00 UTC in summer time
    assert ag.end_date is None and ag.location_name == "Manifesten" and ag.url == "https://www.facebook.com/events/1269179411830316/"
    assert ag.summary.startswith("On fait le point")
    assert ag.image == "https://cdn.example.test/ag.jpg"
    fest = fb_event_to_schema(events[1], today)
    assert fest.image is None
    assert fest.status == "cancelled" and fest.end_date == date(2026, 11, 8) and fest.price == "5 EUR"
    assert fest.url == "https://example.test/billetterie"
    assert fb_event_to_schema(events[2], today) is None


# ------------------------------------------------------------------ deterministic post checks


def _post(text: str, published: date = date(2026, 9, 5), images: int = 1) -> SocialPost:
    return SocialPost(platform="instagram", id="1", url="https://www.instagram.com/p/x/",
                      published=datetime.combine(published, time(12, 0), tzinfo=timezone.utc), text=text,
                      image_urls=[f"https://cdn.example.test/{i}.jpg" for i in range(images)])


def _ev(**kw) -> PostEvent:
    base = dict(title="concert de soutien avec Les Fauves Sonores", start_date=date(2026, 9, 25), start_time=time(19, 0),
                evidence=["Vendredi 25 septembre à 19h : concert de soutien"])
    base.update(kw)
    return PostEvent(**base)


def test_caption_grounded_event_passes_and_flyer_url_is_dropped_not_rejected():
    today = date(2026, 9, 12)
    post = _post("Vendredi 25 septembre à 19h : concert de soutien avec Les Fauves Sonores. Infos : https://manifesten.fr/concert")
    ev = _ev(url="https://manifesten.fr/concert")
    assert check_post_event(ev, post, today, has_images=True) == []
    assert ev.url == "https://manifesten.fr/concert"
    ev = _ev(url="https://billetterie.example/xyz")  # not in the caption: transcribed from the flyer at best
    assert check_post_event(ev, post, today, has_images=True) == []
    assert ev.url is None


def test_paraphrased_caption_evidence_is_rejected_but_allowed_from_image():
    today = date(2026, 9, 12)
    post = _post("Programme de la rentrée en image !")
    ev = _ev(evidence=["25 septembre 19h concert"], title="Concert de soutien")
    problems = check_post_event(ev, post, today, has_images=True)
    assert any("evidence not found" in p for p in problems) and any("title not found" in p for p in problems)
    assert check_post_event(_ev(evidence=["25 septembre 19h concert"], title="Concert de soutien", from_image=True), post, today, has_images=True) == []
    assert check_post_event(_ev(from_image=True), _post("x", images=0), today, has_images=False)


def test_publication_date_pins_the_year_and_rejects_recaps():
    today = date(2026, 9, 12)
    post = _post("Vendredi 25 septembre à 19h : concert de soutien avec Les Fauves Sonores", published=date(2026, 9, 5))
    assert any("wrong year" in p for p in check_post_event(_ev(start_date=date(2027, 9, 24), evidence=["concert de soutien avec Les Fauves Sonores"]), post, today, has_images=True))
    recap = _post("Merci pour le concert de soutien avec Les Fauves Sonores de vendredi", published=date(2026, 9, 27))
    problems = check_post_event(_ev(start_date=date(2026, 9, 25), evidence=["concert de soutien avec Les Fauves Sonores"]), recap, date(2026, 9, 20), has_images=True)
    assert any("before the post publication date" in p for p in problems)


def test_weekday_named_in_evidence_must_match():
    post = _post("Samedi 25 septembre à 19h : concert de soutien avec Les Fauves Sonores")
    problems = check_post_event(_ev(evidence=["Samedi 25 septembre à 19h : concert de soutien"]), post, date(2026, 9, 12), has_images=True)
    assert any("weekday mismatch" in p for p in problems)


# ------------------------------------------------------------------ rebuild from cache


def test_rebuild_publishes_accepted_upcoming_events_only():
    today = date(2026, 9, 12)
    src = SocialSource(kind="instagram", handle="m", url="https://www.instagram.com/m/")
    src.posts["a"] = PostRecord(id="a", url="https://www.instagram.com/p/a/", published=date(2026, 9, 1), analyzed=today, events=[
        StoredEvent(title="Upcoming", start_date=date(2026, 9, 20), evidence=["x" * 20], verifier="accept", verifier_reason="ok"),
        StoredEvent(title="Rejected", start_date=date(2026, 9, 21), evidence=["x" * 20], verifier="reject"),
        StoredEvent(title="Uncertain", start_date=date(2026, 9, 21), evidence=["x" * 20], verifier="uncertain"),
        StoredEvent(title="Past", start_date=date(2026, 9, 10), evidence=["x" * 20], verifier="accept"),
        StoredEvent(title="Unverified", start_date=date(2026, 9, 22), evidence=["x" * 20]),
    ])
    published = rebuild(src, today)
    assert sorted(ev.title for ev, _, _ in published) == ["Unverified", "Upcoming"]
    ev, verdict, reason = next(p for p in published if p[0].title == "Upcoming")
    assert verdict == "accept" and reason == "ok" and ev.url == "https://www.instagram.com/p/a/" and ev.grounded_text is False
    assert ev.image is None  # a record cached before images were kept
    src.posts["a"].image_urls = ["https://cdn.example.test/flyer.jpg", "https://cdn.example.test/flyer-2.jpg"]
    assert all(ev.image == "https://cdn.example.test/flyer.jpg" for ev, _, _ in rebuild(src, today))


def test_fetch_window_and_due():
    today = date(2026, 9, 12)
    src = SocialSource(kind="instagram", handle="m", url="u")
    assert social.due(src, today)
    assert social.fetch_window(src, today) == today - timedelta(days=30)
    src.last_fetched = src.last_ok = today - timedelta(days=1)
    assert social.due(src, today) and social.fetch_window(src, today) == today - timedelta(days=7)
    src.last_fetched = today
    assert not social.due(src, today)
    fb = SocialSource(kind="facebook-events", handle="m", url="u", last_fetched=today - timedelta(days=3))
    assert not social.due(fb, today) and social.due(fb, today + timedelta(days=4))


def test_venue_social_sources_fall_back_to_discovered_handles():
    venue = Venue(name="Manifesten", web=False)
    record = SourceRecord(venue_id="manifesten", discovered=date(2026, 9, 1),
                          source=DiscoveredSource(url="https://x", kind="html", confidence=0.5, reasoning="t", instagram="manifesten", facebook="manifesten"))
    sf = SocialFile()
    keys = [s.key for s in social.venue_social_sources(venue, record, sf)]
    assert keys == ["instagram:manifesten", "facebook:manifesten", "facebook-events:manifesten"]
    assert sf.sources["facebook-events:manifesten"].url == "https://www.facebook.com/manifesten/upcoming_hosted_events"
    venue2 = Venue(name="Manifesten", web=False, instagram="@other")
    assert [s.key for s in social.venue_social_sources(venue2, record, sf)][0] == "instagram:other"


# ------------------------------------------------------------------ end to end with fakes


def _settings(tmp_path: Path) -> Settings:
    return Settings(openrouter_api_key="test", extractor_model="x", verifier_model="x", discover_model="x", search_model="x",
                    vision_model="deepseek/x", vision_verifier_model="qwen/x", apify_token="apify_test",
                    data_dir=tmp_path, venues_file=tmp_path / "venues.json")


@pytest.fixture
def fakes(monkeypatch):
    calls = SimpleNamespace(actors=[], extracted=[], verified=[])

    def fake_run_actor(token, actor, run_input, **kw):
        calls.actors.append((actor, run_input))
        name = {apify.IG_ACTOR: "instagram_posts.json", apify.FB_POSTS_ACTOR: "facebook_posts.json",
                apify.FB_EVENTS_ACTOR: "facebook_events.json"}[actor]
        return [it for it in _items(name) if not it.get("error")]

    async def fake_extract(agent, deps):
        calls.extracted.append(deps.post.id)
        text = deps.post.text
        if "Merci" in text:
            out = PostExtraction(events=[], is_announcement=False, notes="recap")
        elif "concert" in text:
            out = PostExtraction(is_announcement=True, events=[PostEvent(
                title="concert de soutien avec Les Fauves Sonores", start_date=date(2026, 9, 25), start_time=time(19, 0),
                evidence=["Vendredi 25 septembre à 19h : concert de soutien"], url="https://manifesten.fr/concert")])
        else:
            out = PostExtraction(is_announcement=True, events=[PostEvent(
                title='Projection-débat "Terres communes"', start_date=date(2026, 10, 3), start_time=time(20, 30),
                evidence=['Samedi 3 octobre - Projection-débat "Terres communes" à 20h30'])])
        # the real validator runs inside the agent; mimic it so the fake stays honest
        for ev in out.events:
            assert check_post_event(ev, deps.post, deps.today, has_images=bool(deps.images)) == []
        return SimpleNamespace(output=out, usage=SimpleNamespace(requests=1))

    async def fake_verify(agent, ev, post, images, venue_name, today):
        calls.verified.append(ev.title)
        return SimpleNamespace(output=SimpleNamespace(verdict="accept", reason="date and time match the caption"),
                               usage=SimpleNamespace(requests=1))

    monkeypatch.setattr(apify, "run_actor", fake_run_actor)
    monkeypatch.setattr(social, "extract_post", fake_extract)
    monkeypatch.setattr(social, "verify_post_event", fake_verify)
    monkeypatch.setattr(social, "download_images", lambda client, urls: [])
    return calls


@pytest.mark.asyncio
async def test_social_only_venue_publishes_and_caches_post_analysis(tmp_path, fakes):
    today = date(2026, 9, 12)
    settings = _settings(tmp_path)
    venues = [Venue(name="Manifesten", category="militant", web=False, instagram="manifesten", facebook="manifesten")]
    sources, state, sf = SourcesFile(), State(), SocialFile()

    report = await pipeline.Runner(settings, today).run(venues, sources, state, sf)
    assert report.sources_total == 3 and report.sources_ok == 3 and report.apify_runs == 3
    assert report.posts_analyzed == 3 and sorted(fakes.extracted) == ["1315686150601111", "3456789012345678901", "3456789012345678902"]
    assert fakes.verified == ["concert de soutien avec Les Fauves Sonores", 'Projection-débat "Terres communes"']
    kinds = sorted((e.source_kind, e.title) for e in state.events.values())
    assert kinds == [
        ("facebook", "Assemblée générale de rentrée"),
        ("facebook", "Festival des luttes"),
        ("facebook", 'Projection-débat "Terres communes"'),
        ("instagram", "concert de soutien avec Les Fauves Sonores"),
    ]
    concert = next(e for e in state.events.values() if e.source_kind == "instagram")
    assert concert.verifier == "accept" and concert.url == "https://manifesten.fr/concert" and concert.start_time == time(19, 0)
    assert concert.source_url == "https://www.instagram.com/manifesten/"
    # The post's flyer is the event's visual (first image of the post); the events tab has its own cover.
    assert concert.image_source == "https://cdn.example.test/flyer-concert.jpg" and concert.image is None
    ag = next(e for e in state.events.values() if e.title == "Assemblée générale de rentrée")
    assert ag.image_source == "https://cdn.example.test/ag.jpg"
    assert not [a for a in report.alerts if a.level == "error"]

    # Next day: posts are fetched again (daily) but nothing is re-analysed; the events tab waits a week.
    fakes.actors.clear(); fakes.extracted.clear(); fakes.verified.clear()
    report2 = await pipeline.Runner(settings, today + timedelta(days=1)).run(venues, sources, state, sf)
    assert [a for a, _ in fakes.actors] == [apify.IG_ACTOR, apify.FB_POSTS_ACTOR]
    assert fakes.extracted == [] and report2.posts_analyzed == 0 and report2.llm_calls == 0
    assert len(state.events) == 4
    ig_input = next(i for a, i in fakes.actors if a == apify.IG_ACTOR)
    assert ig_input["onlyPostsNewerThan"] == (today + timedelta(days=1) - timedelta(days=7)).isoformat()

    # State survives a save/load round trip.
    social.save_social(tmp_path / "social.json", sf)
    again = social.load_social(tmp_path / "social.json")
    assert again.sources["instagram:manifesten"].posts.keys() == sf.sources["instagram:manifesten"].posts.keys()


@pytest.mark.asyncio
async def test_apify_failure_keeps_cached_events_and_alerts_after_two_failures(tmp_path, fakes, monkeypatch):
    today = date(2026, 9, 12)
    settings = _settings(tmp_path)
    venues = [Venue(name="Manifesten", web=False, instagram="manifesten")]
    sources, state, sf = SourcesFile(), State(), SocialFile()
    await pipeline.Runner(settings, today).run(venues, sources, state, sf)
    assert len(state.events) == 1

    def boom(*a, **k):
        raise apify.ApifyError("actor run timed out")

    monkeypatch.setattr(apify, "run_actor", boom)
    r1 = await pipeline.Runner(settings, today + timedelta(days=1)).run(venues, sources, state, sf)
    assert len(state.events) == 1 and r1.sources_ok == 1 and not [a for a in r1.alerts if a.level == "error"]
    r2 = await pipeline.Runner(settings, today + timedelta(days=2)).run(venues, sources, state, sf)
    assert len(state.events) == 1 and r2.sources_ok == 0
    assert any("fetch failed 2x" in a.message for a in r2.alerts if a.level == "error")


@pytest.mark.asyncio
async def test_without_llm_new_posts_are_not_recorded(tmp_path, fakes):
    today = date(2026, 9, 12)
    settings = _settings(tmp_path)
    venues = [Venue(name="Manifesten", web=False, instagram="manifesten")]
    sf = SocialFile()
    report = await pipeline.Runner(settings, today, allow_llm=False).run(venues, SourcesFile(), State(), sf)
    assert fakes.extracted == [] and sf.sources["instagram:manifesten"].posts == {}
    assert any("LLM disabled" in a.message for a in report.alerts)


def test_run_actor_treats_no_results_as_empty_and_other_errors_as_failure():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "empty" in body["startUrls"][0]:
            return httpx.Response(200, json=[{"error": "no_results", "errorDescription": "No events"}])
        if "missing" in body["startUrls"][0]:
            return httpx.Response(200, json=[{"error": "not_found", "errorDescription": "Page not found"}])
        return httpx.Response(408, text="timeout")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert apify.run_actor("t", apify.FB_EVENTS_ACTOR, {"startUrls": ["https://www.facebook.com/empty/"]}, client=client) == []
        with pytest.raises(apify.ApifyError, match="not_found"):
            apify.run_actor("t", apify.FB_EVENTS_ACTOR, {"startUrls": ["https://www.facebook.com/missing/"]}, client=client)
        with pytest.raises(apify.ApifyError, match="timed out"):
            apify.run_actor("t", apify.FB_EVENTS_ACTOR, {"startUrls": ["https://www.facebook.com/slow/"]}, client=client)


def test_platform_image_text_is_carried_as_a_hint():
    posts = apify.facebook_posts(_items("facebook_posts.json"))
    assert posts[0].image_text.startswith("TERRES COMMUNES")
