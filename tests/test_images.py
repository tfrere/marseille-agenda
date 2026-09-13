"""Image cache without network: httpx is replaced by a mock transport serving generated pictures."""

from datetime import date, time, timedelta
from io import BytesIO

import httpx
from PIL import Image

from marseille_agenda import images
from marseille_agenda.merge import make_uid
from marseille_agenda.schema import Event, State


def _png(width: int, height: int, mode: str = "RGB") -> bytes:
    buf = BytesIO()
    color = (200, 30, 30, 128) if mode == "RGBA" else (200, 30, 30)
    Image.new(mode, (width, height), color).save(buf, "PNG")
    return buf.getvalue()


def _event(title: str, image_source: str | None, source_url: str = "https://venue.test/agenda/") -> Event:
    day = date(2026, 9, 20)
    return Event(
        uid=make_uid("venue", title, day), venue_id="venue", venue_name="Venue", category="culture",
        title=title, start_date=day, start_time=time(20, 0), source_url=source_url, source_kind="html",
        evidence=["x" * 20], first_seen=day, last_seen=day, image_source=image_source,
    )


def _client(seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/poster.png":
            return httpx.Response(200, content=_png(1200, 1800), headers={"content-type": "image/png"})
        if path == "/transparent.png":
            return httpx.Response(200, content=_png(400, 300, "RGBA"), headers={"content-type": "image/png"})
        if path == "/tiny.png":
            return httpx.Response(200, content=_png(40, 40), headers={"content-type": "image/png"})
        if path == "/page.html":
            return httpx.Response(200, content=b"<html>not an image</html>", headers={"content-type": "text/html"})
        if path == "/garbage.jpg":
            return httpx.Response(200, content=b"\x00\x01\x02 not a picture", headers={"content-type": "image/jpeg"})
        return httpx.Response(404, content=b"gone")

    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": images.BROWSER_USER_AGENT})


def test_sync_images_downloads_resizes_and_records_failures(tmp_path):
    today = date(2026, 9, 12)
    events = [
        _event("Poster", "https://cdn.test/poster.png"),
        _event("Transparent", "https://cdn.test/transparent.png"),
        _event("Tiny", "https://cdn.test/tiny.png"),
        _event("Dead", "https://cdn.test/missing.jpg"),
        _event("Html", "https://cdn.test/page.html"),
        _event("Garbage", "https://cdn.test/garbage.jpg"),
        _event("No source", None),
    ]
    state = State()
    seen: list[httpx.Request] = []
    stats = images.sync_images(events, state, tmp_path, today, client=_client(seen))
    by_title = {e.title: e for e in events}

    poster = by_title["Poster"]
    assert poster.image == f"img/{poster.uid}.webp"
    with Image.open(tmp_path / "img" / f"{poster.uid}.webp") as im:
        assert im.format == "WEBP" and im.mode == "RGB" and max(im.size) == 640 and im.size == (427, 640)
    assert by_title["Transparent"].image == f"img/{by_title['Transparent'].uid}.webp"
    for title in ("Tiny", "Dead", "Html", "Garbage", "No source"):
        assert by_title[title].image is None, title
    assert (stats.downloaded, stats.failed, stats.without_source) == (2, 4, 1)
    assert state.image_failures == {
        "https://cdn.test/tiny.png": today, "https://cdn.test/missing.jpg": today,
        "https://cdn.test/page.html": today, "https://cdn.test/garbage.jpg": today,
    }
    assert ("cdn.test", "HTTP 404") in stats.failures and ("cdn.test", "too small") in stats.failures
    # Browser-like request: the venue page as Referer gets past hotlink protection.
    req = next(r for r in seen if r.url.path == "/poster.png")
    assert req.headers["Referer"] == "https://venue.test/agenda/" and req.headers["User-Agent"].startswith("Mozilla/5.0")

    # Next run: cached files are reused, recent failures are not retried, after 14 days they are.
    seen.clear()
    stats = images.sync_images(events, state, tmp_path, today + timedelta(days=1), client=_client(seen))
    assert stats.cached == 2 and stats.downloaded == 0 and stats.skipped_recent_failure == 4 and seen == []
    stats = images.sync_images(events, state, tmp_path, today + timedelta(days=images.RETRY_FAILED_AFTER_DAYS), client=_client(seen))
    assert stats.skipped_recent_failure == 0 and stats.failed == 4 and len(seen) == 4


def test_sync_images_removes_files_of_events_no_longer_published(tmp_path):
    today = date(2026, 9, 12)
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    (img_dir / "stale000000.webp").write_bytes(b"old")
    kept = _event("Poster", "https://cdn.test/poster.png")
    stats = images.sync_images([kept], State(), tmp_path, today, client=_client([]))
    assert stats.removed == 1 and sorted(p.name for p in img_dir.iterdir()) == [f"{kept.uid}.webp"]


def test_download_image_refuses_oversized_bodies():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/declared":
            return httpx.Response(200, content=b"x", headers={"content-length": str(images.MAX_BYTES + 1), "content-type": "image/jpeg"})
        return httpx.Response(200, content=b"x" * (images.MAX_BYTES + 1), headers={"content-type": "image/jpeg"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        for path in ("/declared", "/streamed"):
            try:
                images.download_image(client, f"https://cdn.test{path}")
            except images.ImageError as exc:
                assert "too big" in str(exc)
            else:
                raise AssertionError(f"{path} should have been refused")
