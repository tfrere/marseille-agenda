"""Truncated titles completed from the event's own page, without network (mock transport)."""

from datetime import date

import httpx

from marseille_agenda import enrich
from marseille_agenda.apply import apply_schema
from marseille_agenda.extraction_schema import ExtractionSchema, FieldSpec, HtmlRule
from marseille_agenda.fetch import document_from_body
from marseille_agenda.merge import make_uid
from marseille_agenda.pipeline import to_event
from marseille_agenda.schema import State, Venue

LISTING = """<div class="list">
  <a class="card" href="/agenda/le-grand-voyage/"><div class="date">16/09/2026 - 08h30</div><h3>Le Grand Voyage aux Fertiles...</h3></a>
  <a class="card" href="/agenda/rencontres/"><div class="date">17/09/2026 - 18h30</div><h3>Rencontres partenaires :&nbsp; La...</h3></a>
  <a class="card" href="/agenda/appart/"><div class="date">21/09/2026 - 09h00</div><h3>L’Appart du Futur...</h3></a>
  <a class="card" href="https://elsewhere.test/expo/"><div class="date">03/10/2026 - 10h00</div><h3>L'exposition...</h3></a>
  <a class="card" href="/agenda/vivantes/"><div class="date">30/09/2026 - 10h00</div><h3>VIVANT.E.S en itinérance</h3></a>
</div>"""

PAGES = {
    # WordPress style: the site name is appended to og:title too, and declared as og:site_name.
    "/agenda/le-grand-voyage/": """<html><head><meta property="og:site_name" content="La Fabulerie">
        <meta property="og:title" content="Le Grand Voyage aux Fertiles : rencontres - La Fabulerie">
        <title>Le Grand Voyage aux Fertiles : rencontres - La Fabulerie</title></head><body><h1>Agenda</h1></body></html>""",
    # No og:title, the h1 is generic, only <title> (with the site name) carries the full title.
    "/agenda/rencontres/": """<html><head><title>Rencontres partenaires : La pédagogie en folie ! | La Fabulerie</title></head>
        <body><h1>La Fabulerie</h1></body></html>""",
    # The detail page's title does not extend the prefix: the prefix stays.
    "/agenda/appart/": """<html><head><meta property="og:title" content="Un appartement pour demain"><title>Un appartement - La Fabulerie</title></head>
        <body><h1>Un appartement pour demain</h1></body></html>""",
}


def _events(state: State):
    doc = document_from_body("https://x.test/", "html", LISTING)
    schema = ExtractionSchema(rules=[HtmlRule(item_selector="a.card", fields={
        "title": FieldSpec(selector="h3"), "date": FieldSpec(selector="div.date"), "url": FieldSpec(selector="", attr="href"),
    })])
    res = apply_schema(schema, doc, date(2026, 9, 12))
    venue = Venue(name="La Fabulerie")
    return [to_event(e, venue, "https://x.test/", "html", date(2026, 9, 12)) for e in res.events]


def _client(seen: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        body = PAGES.get(request.url.path)
        return httpx.Response(200, text=body) if body else httpx.Response(404, text="nope")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_truncated_titles_are_completed_only_when_the_page_title_extends_the_prefix():
    state = State()
    events = _events(state)
    seen: list[str] = []
    stats = enrich.complete_truncated_titles(events, state, _client(seen))
    by_url = {e.url: e for e in events}

    grand = by_url["https://x.test/agenda/le-grand-voyage/"]
    assert grand.title == "Le Grand Voyage aux Fertiles : rencontres" and grand.title_truncated
    assert grand.title_completed_from == grand.url
    assert grand.uid == make_uid("la-fabulerie", "Le Grand Voyage aux Fertiles", date(2026, 9, 16)), "uid stays keyed on the listing prefix"
    # <title> minus the site name, matched across nbsp / case / accents.
    assert by_url["https://x.test/agenda/rencontres/"].title == "Rencontres partenaires : La pédagogie en folie !"
    # Negative case: page title does not start with the prefix -> unchanged, but remembered as consulted.
    appart = by_url["https://x.test/agenda/appart/"]
    assert appart.title == "L’Appart du Futur" and appart.title_completed_from == appart.url
    # Another host: never fetched. Not truncated: untouched.
    assert by_url["https://elsewhere.test/expo/"].title == "L'exposition" and by_url["https://elsewhere.test/expo/"].title_completed_from is None
    assert by_url["https://x.test/agenda/vivantes/"].title == "VIVANT.E.S en itinérance"
    assert (stats.completed, stats.unchanged, stats.skipped, stats.cached) == (2, 1, 1, 0)
    assert sorted(seen) == ["https://x.test/agenda/appart/", "https://x.test/agenda/le-grand-voyage/", "https://x.test/agenda/rencontres/"]

    # Next run: the state answers, nothing is fetched again (completed and unchanged alike).
    for e in events:
        state.events[e.uid] = e
    fresh = _events(state)
    seen.clear()
    stats = enrich.complete_truncated_titles(fresh, state, _client(seen))
    assert seen == [] and stats.cached == 3
    assert {e.url: e.title for e in fresh} == {e.url: e.title for e in events}


def test_fetch_budget_and_errors_leave_the_prefix():
    state = State()
    events = [e for e in _events(state) if e.title_truncated and e.url.startswith("https://x.test/")]
    seen: list[str] = []
    stats = enrich.complete_truncated_titles(events, state, _client(seen), max_fetches=1)
    assert len(seen) == 1 and stats.skipped == 2

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    state = State()
    events = _events(state)
    stats = enrich.complete_truncated_titles(events, state, httpx.Client(transport=httpx.MockTransport(boom)))
    assert stats.errors == 3 and all(e.title_completed_from is None for e in events)
    assert [e.title for e in events][0] == "Le Grand Voyage aux Fertiles"


def test_title_candidates_order_and_site_suffix():
    html = '<html><head><title>Concert de rentrée | Le Molotov</title></head><body><h1>Concert  de\nrentrée</h1></body></html>'
    assert enrich.title_candidates(html) == ["Concert de rentrée"], "the site name never becomes part of a title"
    assert enrich.full_title(html, "Concert de...") is None, "the ellipsis must be stripped before"
    assert enrich.full_title(html, "Concert de") == "Concert de rentrée"
    assert enrich.full_title(html, "Concert de rentrée") is None, "equal is not longer"
    # No og:site_name and a lone og:title with a dash: the segment after it is part of the title.
    html = ('<html><head><meta property="og:title" content="Nadin Nadine - Toponymie sensible">'
            '<title>Nadin Nadine - Toponymie sensible - La Fabulerie</title></head><body><h1>Nadin Nadine - Toponymie sensible</h1></body></html>')
    assert enrich.title_candidates(html) == ["Nadin Nadine - Toponymie sensible"]
    html = '<html><head><meta property="og:title" content="Nadin Nadine - Toponymie sensible"><title>Agenda</title></head><body></body></html>'
    assert enrich.title_candidates(html) == ["Nadin Nadine - Toponymie sensible", "Agenda"]
