"""Load-more pagination (listings that server-render only their first batch)."""

import json
from urllib.parse import parse_qs

import httpx

from marseille_agenda.extraction_schema import LoadMore
from marseille_agenda.fetch import fetch_document

PAGE = """<html><body><div class="agenda-wrapper">
<div class="events" data-date="2026-11-01"><div class="event"><h3>Marché paysan</h3></div></div>
<div class="events" data-date="2026-11-02"><div class="event"><h3>Visite flash</h3></div></div>
</div></body></html>"""

MORE = {
    "2026-11-02": {"html": '<div class="events" data-date="2026-11-20"><div class="event"><h3>Club Cabaret</h3></div></div>', "has_more": True},
    "2026-11-20": {"html": '<div class="events" data-date="2026-12-04"><div class="event"><h3>Arch Club</h3></div></div>', "has_more": False},
}

SPEC = LoadMore(url="https://example.test/wp-admin/admin-ajax.php",
                form={"action": "get_next_events", "data[date]": "{cursor}", "load_next": "true"},
                cursor_regex=r'data-date="(\d{4}-\d{2}-\d{2})"')


def _client(posts: list[dict], fail_after: int | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=PAGE)
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        posts.append(form)
        if fail_after is not None and len(posts) > fail_after:
            return httpx.Response(500)
        return httpx.Response(200, text=json.dumps(MORE.get(form["data[date]"], {"html": "", "has_more": False})))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_load_more_appends_every_page_until_has_more_is_false():
    posts: list[dict] = []
    doc = fetch_document("https://example.test/agenda/", "html", _client(posts), SPEC)
    assert [p["data[date]"] for p in posts] == ["2026-11-02", "2026-11-20"]
    assert posts[0]["action"] == "get_next_events" and posts[0]["load_next"] == "true"
    assert all(t in doc.text for t in ("Marché paysan", "Visite flash", "Club Cabaret", "Arch Club"))
    assert doc.raw.index("Arch Club") < doc.raw.index("</body>")


def test_load_more_failure_keeps_the_pages_gathered_so_far():
    posts: list[dict] = []
    doc = fetch_document("https://example.test/agenda/", "html", _client(posts, fail_after=1), SPEC)
    assert len(posts) == 2
    assert "Club Cabaret" in doc.text and "Arch Club" not in doc.text


def test_without_load_more_only_the_page_is_read():
    posts: list[dict] = []
    doc = fetch_document("https://example.test/agenda/", "html", _client(posts))
    assert posts == [] and "Club Cabaret" not in doc.text
