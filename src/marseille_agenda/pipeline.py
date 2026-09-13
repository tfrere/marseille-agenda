"""Daily run.

For each venue in venues.json:
  1. no source known      -> discovery agent (LLM, once)            -> data/sources.json
  2. no schema known      -> induction (no LLM) or generator (LLM), once -> data/sources.json
     (a listing with no upcoming date in its HTML is a client-rendered shell: the discovered
     fallback_url, then the site home, are tried first when they induce a schema)
  3. fetch source, execute schema (no LLM)
  4. schema unhealthy     -> regenerate (LLM), then re-discover after repeated failures
  5. deterministic checks -> adversarial verifier on NEW html events only (LLM, cheap)
     titles the listing cut short are completed from the event's own page (prefix-verified)
  6. merge into data/state.json, publish data/events.json + events.ics + report.json
  7. cache each published event's visual as site/img/<uid>.webp (no LLM, skip with --no-images)

Every LLM call is therefore conditional: an unchanged venue costs zero tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

import httpx

from .apply import SchemaEvent, apply_schema
from .config import TZ, Settings, load_settings
from .discover import build_discoverer, discover_source
from .enrich import complete_truncated_titles
from .extract import build_extractor
from .extraction_schema import ExtractionSchema, SchemaRecord, SourceRecord, SourcesFile
from .fetch import SourceDocument, fetch_document, make_client
from .induce import induce_schema, upcoming_date_count
from .llm import fetch_credits, make_model
from .filters import held_here, publishable
from .merge import collapse_daily_runs, expire_past, forget_events, make_uid, merge_source
from .images import sync_images
from .output import load_state, published_events, save_state, write_events_json, write_ics, write_report
from .schema import Alert, Event, RunReport, State, Venue, VenuesFile
from .schema_gen import build_generator, generate_schema
from .social import SocialFile, SocialRunner, load_social, save_social, venue_social_sources
from .social_extract import build_post_extractor, build_post_verifier
from .validate import check_event
from .verify import build_verifier, verify_event

log = logging.getLogger("marseille_agenda")

FAILURES_BEFORE_REDISCOVER = 3
EMPTY_SOURCE_RETRY_DAYS = 7
INDUCTION_MIN_QUALITY = 0.8  # below this the induced schema is discarded and the generator agent takes over
MIN_HTML_TEXT_CHARS = 300


def load_venues(settings: Settings) -> list[Venue]:
    data = json.loads(settings.venues_file.read_text(encoding="utf-8"))
    return VenuesFile.model_validate(data).venues


def load_sources(settings: Settings) -> SourcesFile:
    p = settings.data_dir / "sources.json"
    if p.exists():
        return SourcesFile.model_validate_json(p.read_text(encoding="utf-8"))
    return SourcesFile()


def save_sources(settings: Settings, sources: SourcesFile) -> None:
    (settings.data_dir / "sources.json").write_text(sources.model_dump_json(indent=1, by_alias=True), encoding="utf-8")


@dataclass
class VenueOutcome:
    events: list[Event] | None
    """None when nothing could be produced this run (previous state is kept)."""
    error: str | None = None
    content_hash: str | None = None
    llm_calls: int = 0
    rejected_checks: int = 0
    rejected_verifier: int = 0
    uncertain: int = 0
    notes: list[str] = field(default_factory=list)


def to_event(ex: SchemaEvent, venue: Venue, source_url: str, kind: str, today: date,
             verifier: str | None = None, verifier_reason: str | None = None) -> Event:
    return Event(
        uid=make_uid(venue.id, ex.title, ex.start_date),
        venue_id=venue.id, venue_name=venue.name, category=venue.category,
        title=ex.title, title_truncated=ex.title_truncated, start_date=ex.start_date, start_time=ex.start_time,
        end_date=ex.end_date, end_time=ex.end_time, location_name=ex.location_name,
        url=ex.url, price=ex.price, summary=ex.summary, status=ex.status,
        event_type=ex.event_type, free=ex.free, image_source=ex.image,
        recurring=bool(ex.end_date and ex.end_date > ex.start_date),
        source_url=source_url, source_kind=kind,  # type: ignore[arg-type]
        evidence=ex.evidence, verifier=verifier, verifier_reason=verifier_reason,  # type: ignore[arg-type]
        first_seen=today, last_seen=today, lat=venue.lat, lng=venue.lng,
    )


class Runner:
    def __init__(self, settings: Settings, today: date, *, verify: bool = True, allow_llm: bool = True, enrich: bool = True):
        self.settings = settings
        self.today = today
        self.verify = verify
        self.allow_llm = allow_llm and settings.has_llm
        self.enrich = enrich
        """Complete truncated titles from the events' own pages (deterministic, a few fetches)."""
        self._agents: dict[str, object] = {}
        self._social: SocialRunner | None = None

    def agent(self, name: str):
        if name not in self._agents:
            s = self.settings
            if name == "extractor":
                self._agents[name] = build_extractor(make_model(s, s.extractor_model))
            elif name == "generator":
                self._agents[name] = build_generator(make_model(s, s.extractor_model))
            elif name == "verifier":
                self._agents[name] = build_verifier(make_model(s, s.verifier_model))
            elif name == "discoverer":
                self._agents[name] = build_discoverer(make_model(s, s.discover_model))
            elif name == "post_extractor":
                self._agents[name] = build_post_extractor(make_model(s, s.vision_model))
            elif name == "post_verifier":
                self._agents[name] = build_post_verifier(make_model(s, s.vision_verifier_model))
        return self._agents[name]

    @property
    def social(self) -> SocialRunner:
        if self._social is None:
            llm = self.allow_llm
            self._social = SocialRunner(self.settings, self.today, allow_llm=llm, verify=self.verify,
                                        extractor=self.agent("post_extractor") if llm else None,
                                        verifier=self.agent("post_verifier") if llm and self.verify else None)
        return self._social

    # ------------------------------------------------------------------ LLM steps

    async def discover(self, venue: Venue, out: VenueOutcome) -> SourceRecord | None:
        if not self.allow_llm:
            out.notes.append("discovery needed but LLM disabled")
            return None
        log.info("[%s] discovering source...", venue.id)
        run = await discover_source(self.agent("discoverer"), self.settings, venue.name, venue.website, self.today)
        out.llm_calls += run.usage.requests
        src = run.output
        log.info("[%s] source: %s (%s, confidence %.2f) - %s", venue.id, src.url, src.kind, src.confidence, src.reasoning)
        return SourceRecord(venue_id=venue.id, source=src, discovered=self.today)

    async def generate(self, venue: Venue, record: SourceRecord, doc: SourceDocument, out: VenueOutcome) -> str:
        """Returns "ok" (schema validated and stored), "empty" (the independent reader found no
        upcoming event on the source, so nothing can be validated) or "failed"."""
        # Deterministic first: a listing of repeated dated cards needs no model to be read.
        induced = induce_schema(doc, self.today)
        if induced is not None and induced.quality >= INDUCTION_MIN_QUALITY:
            log.info("[%s] schema induced deterministically: %s", venue.id, induced.describe())
            out.notes.append(f"schema induced without a model: {induced.describe()}")
            self._store_schema(record, induced.schema, validated=False, agreement=induced.quality, model="induction")
            return "ok"
        if induced is not None:
            log.info("[%s] induction too weak (%s); asking the generator", venue.id, induced.describe())
        if not self.allow_llm:
            out.notes.append("schema generation needed but LLM disabled")
            return "failed"
        log.info("[%s] generating extraction schema...", venue.id)
        gen = await generate_schema(self.agent("generator"), self.agent("extractor"), doc, venue.name, self.today)
        out.llm_calls += gen.llm_calls
        log.info("[%s] schema agreement %.2f (validated=%s) after %d attempt(s); reader saw %d events",
                 venue.id, gen.agreement, gen.validated, len(gen.attempts), len(gen.reference.events))
        if not gen.validated and not gen.reference.events:
            record.next_generation = self.today + timedelta(days=EMPTY_SOURCE_RETRY_DAYS)
            out.notes.append(f"source lists no upcoming event yet; schema generation postponed to {record.next_generation}")
            return "empty"
        if not gen.validated:
            out.notes.append(f"schema not validated (agreement {gen.agreement:.2f}); attempts={gen.attempts}")
            return "failed"
        self._store_schema(record, gen.schema, validated=True, agreement=gen.agreement, model=self.settings.extractor_model)
        return "ok"

    async def client_rendered_fallback(self, venue: Venue, record: SourceRecord, doc: SourceDocument, client: httpx.Client,
                                       out: VenueOutcome) -> SourceDocument:
        """An HTML listing whose server-rendered text holds no upcoming date at all is most likely
        a client-rendered shell (the cards are injected by JavaScript), not an empty agenda.

        Before concluding "nothing announced" (which postpones generation for a week), try the
        discovered `fallback_url` then the site home: when one of them induces a schema
        deterministically, the source switches to it and the run goes on with that document.
        Returns the document to work with (`doc` itself when nothing better was found).
        """
        if doc.kind != "html" or upcoming_date_count(doc, self.today) > 0:
            return doc
        parts = urlsplit(doc.url)
        home = urlunsplit((parts.scheme, parts.netloc, "/", "", ""))
        candidates = [u for u in dict.fromkeys([record.source.fallback_url, home]) if u and u.rstrip("/") != doc.url.rstrip("/")]
        log.info("[%s] %s has no upcoming date in its server-rendered text (client-rendered listing?); trying %s",
                 venue.id, doc.url, ", ".join(candidates) or "nothing")
        for url in candidates:
            try:
                alt = await asyncio.to_thread(fetch_document, url, "html", client)
            except Exception as exc:  # noqa: BLE001
                log.info("[%s] fallback %s could not be fetched: %s", venue.id, url, exc)
                continue
            induced = induce_schema(alt, self.today)
            if induced is None or induced.quality < INDUCTION_MIN_QUALITY:
                log.info("[%s] fallback %s: %s", venue.id, url, induced.describe() if induced else "no repeated dated cards")
                continue
            note = (f"{doc.url} has no upcoming date in its server-rendered text (client-rendered listing); "
                    f"source switched to {url}: {induced.describe()}")
            log.info("[%s] %s", venue.id, note)
            out.notes.append(note)
            record.source.reasoning += f" [{self.today}] {note}"
            record.source.url = url
            record.source.kind = "html"
            return alt
        return doc

    def _store_schema(self, record: SourceRecord, schema: ExtractionSchema, *, validated: bool, agreement: float, model: str) -> None:
        prev = record.schema_record
        record.schema_record = SchemaRecord(
            schema=schema, created=self.today, validated_against_llm=validated, agreement=round(agreement, 3),
            generator_model=model, version=(prev.version + 1) if prev else 1,
        )
        if prev:
            record.regenerations += 1
        record.next_generation = None

    # ------------------------------------------------------------------ per venue

    async def process_venue(self, venue: Venue, sources: SourcesFile, state: State, client: httpx.Client) -> VenueOutcome:
        out = VenueOutcome(events=[])
        record = sources.sources.get(venue.id)

        if record is None or record.consecutive_failures >= FAILURES_BEFORE_REDISCOVER:
            if record is not None:
                out.notes.append(f"{record.consecutive_failures} consecutive failures on {record.source.url}: re-discovering")
            try:
                new = await self.discover(venue, out)
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] discovery failed", venue.id)
                return VenueOutcome(events=None, error=f"discovery: {type(exc).__name__}: {exc}", llm_calls=out.llm_calls)
            if new is None:
                return VenueOutcome(events=None, error="no source and discovery unavailable", notes=out.notes)
            if record is not None and new.source.url == record.source.url:
                record.consecutive_failures = 0  # same source re-confirmed; keep schema, try again
            else:
                record = new
            sources.sources[venue.id] = record

        # Fetch.
        try:
            doc = await asyncio.to_thread(fetch_document, record.source.url, record.source.kind, client)
        except Exception as exc:  # noqa: BLE001
            record.consecutive_failures += 1
            record.last_error = f"fetch: {exc}"
            return VenueOutcome(events=None, error=record.last_error, llm_calls=out.llm_calls, notes=out.notes)
        out.content_hash = doc.content_hash

        # A freshly discovered HTML source with (almost) no server-rendered text is a client-side
        # app shell: nothing can be extracted from it, and "no events" would be a false conclusion.
        if record.schema_record is None and doc.kind == "html" and len(doc.text.strip()) < MIN_HTML_TEXT_CHARS:
            record.consecutive_failures = FAILURES_BEFORE_REDISCOVER  # re-discover next run
            record.last_error = f"source is a client-rendered shell ({len(doc.text.strip())} chars of text)"
            return VenueOutcome(events=None, error=record.last_error, llm_calls=out.llm_calls, notes=out.notes)

        # Schema: generate if missing.
        if record.schema_record is None:
            if record.next_generation and self.today < record.next_generation:
                out.notes.append(f"no upcoming event on source at last check; next generation attempt {record.next_generation}")
                return self._empty_ok(record, doc, out)
            doc = await self.client_rendered_fallback(venue, record, doc, client, out)
            out.content_hash = doc.content_hash
            try:
                status = await self.generate(venue, record, doc, out)
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] schema generation failed", venue.id)
                status, out.notes = "failed", out.notes + [f"generation error: {type(exc).__name__}: {exc}"]
            if status == "empty":
                return self._empty_ok(record, doc, out)
            if status != "ok":
                record.consecutive_failures += 1
                record.last_error = "schema generation failed"
                return VenueOutcome(events=None, error=record.last_error, llm_calls=out.llm_calls, notes=out.notes)

        # Execute (free) and judge health.
        res = apply_schema(record.schema_record.schema_, doc, self.today)
        unhealthy = self._unhealthy(res, record)
        if unhealthy:
            log.warning("[%s] schema unhealthy (%s), regenerating", venue.id, unhealthy)
            out.notes.append(f"schema unhealthy: {unhealthy}")
            status = "failed"
            try:
                doc = await self.client_rendered_fallback(venue, record, doc, client, out)
                out.content_hash = doc.content_hash
                status = await self.generate(venue, record, doc, out)
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] regeneration failed", venue.id)
                out.notes.append(f"regeneration error: {type(exc).__name__}: {exc}")
            if status == "ok":
                res = apply_schema(record.schema_record.schema_, doc, self.today)
                unhealthy = self._unhealthy(res, record)
            elif status == "empty":
                # The independent reader confirms the source currently lists nothing upcoming:
                # the schema is fine, the agenda is just empty for now.
                return self._empty_ok(record, doc, out)
            if unhealthy:
                record.consecutive_failures += 1
                record.last_error = f"unhealthy schema: {unhealthy}"
                return VenueOutcome(events=None, error=record.last_error, llm_calls=out.llm_calls, notes=out.notes)

        # Deterministic checks.
        kept: list[SchemaEvent] = []
        for ev in res.events:
            problems = check_event(ev, doc, self.today, check_text=ev.grounded_text)
            if problems:
                out.rejected_checks += 1
                if not any("past" in p for p in problems):  # past events are expected noise, not worth an alert
                    out.notes.append(f"rejected {ev.title!r} {ev.start_date}: {'; '.join(problems)}")
            else:
                kept.append(ev)

        # Day-by-day agendas repeat running exhibitions under every date: one range entry each.
        kept = collapse_daily_runs(kept)
        # Category rules (cinemas: special sessions only), before paying for any verification.
        kept, dropped = publishable(venue, kept)
        if dropped:
            log.info("[%s] %d regular screening(s) left out by the %s rule", venue.id, dropped, venue.category)
        # Aggregators: only the events held at the venue itself when a location_filter is set.
        kept, dropped = held_here(venue, kept)
        if dropped:
            log.info("[%s] %d event(s) held elsewhere left out by location_filter %r", venue.id, dropped, venue.location_filter)

        # Adversarial verification of NEW events from free-form HTML.
        events: list[Event] = []
        to_verify: list[SchemaEvent] = []
        for ev in kept:
            uid = make_uid(venue.id, ev.title, ev.start_date)
            prev = state.events.get(uid)
            if prev is not None and prev.verifier == "accept":
                events.append(to_event(ev, venue, record.source.url, record.source.kind, self.today, "accept", prev.verifier_reason))
            elif record.source.kind == "html" and self.verify and self.allow_llm and ev.grounded_text:
                to_verify.append(ev)
            else:
                events.append(to_event(ev, venue, record.source.url, record.source.kind, self.today))
        if to_verify:
            sem = asyncio.Semaphore(6)

            async def check(ev: SchemaEvent):
                async with sem:
                    return ev, await verify_event(self.agent("verifier"), ev, doc, venue.name, self.today)

            for ev, vres in await asyncio.gather(*[check(ev) for ev in to_verify]):
                out.llm_calls += vres.usage.requests
                verdict = vres.output
                if verdict.verdict == "accept":
                    events.append(to_event(ev, venue, record.source.url, record.source.kind, self.today, "accept", verdict.reason))
                elif verdict.verdict == "uncertain":
                    out.uncertain += 1
                    out.notes.append(f"verifier uncertain {ev.title!r} {ev.start_date}: {verdict.reason}")
                else:
                    out.rejected_verifier += 1
                    out.notes.append(f"verifier rejected {ev.title!r} {ev.start_date}: {verdict.reason}")

        record.consecutive_failures = 0
        record.last_error = None
        record.last_ok = self.today
        record.last_event_count = len(events)
        record.content_hash = doc.content_hash
        out.events = events
        return out

    # ------------------------------------------------------------------ social sources

    async def process_social(self, venue: Venue, sources: SourcesFile, social: SocialFile, state: State,
                             client: httpx.Client, report: RunReport) -> None:
        for src in venue_social_sources(venue, sources.sources.get(venue.id), social):
            report.sources_total += 1
            out = await self.social.process(venue, src, client)
            report.llm_calls += out.llm_calls
            report.posts_analyzed += out.posts_analyzed
            report.apify_runs += int(out.fetched)
            report.events_rejected_grounding += out.rejected_checks
            report.events_rejected_verifier += out.rejected_verifier
            report.events_uncertain += out.uncertain
            kind = "facebook" if src.kind.startswith("facebook") else "instagram"
            events: list[Event] = []
            for ev, verdict, reason in out.events:
                e = to_event(ev, venue, src.url, kind, self.today, verdict, reason)
                prev = state.events.get(e.uid)
                if prev is not None and prev.source_url != src.url and prev.last_seen == self.today:
                    continue  # already published from another source this run (the website is canonical)
                events.append(e)
            events, dropped = publishable(venue, events)
            if dropped:
                log.info("[%s] %s: %d regular screening(s) left out by the %s rule", venue.id, src.key, dropped, venue.category)
            if out.error is None:
                report.sources_ok += 1
            alerts = merge_source(state, src.url, venue.id, events, self.today, None, None)
            if out.error:
                alerts.append(Alert(level="error", venue_id=venue.id, source_url=src.url, message=out.error))
            report.alerts.extend(alerts)
            for n in out.notes:
                report.alerts.append(Alert(level="warning", venue_id=venue.id, source_url=src.url, message=n))
            log.info("[%s] %s: %d event(s), %d post(s) analysed", venue.id, src.key, len(events), out.posts_analyzed)

    def _empty_ok(self, record: SourceRecord, doc: SourceDocument, out: VenueOutcome) -> VenueOutcome:
        """Successful run with zero events (source confirmed empty)."""
        record.consecutive_failures = 0
        record.last_error = None
        record.last_ok = self.today
        record.last_event_count = 0
        record.content_hash = doc.content_hash
        out.events = []
        return out

    @staticmethod
    def _unhealthy(res, record: SourceRecord) -> str | None:
        if res.items_seen == 0:
            return "no items matched"
        if res.items_seen and not res.events and res.undated == res.items_seen:
            return "date selector matches nothing in any item"
        if res.failure_ratio > 0.5 and record.source.kind == "html":
            return f"{len(res.failures)}/{res.items_seen} items failed to parse"
        if not res.events and record.last_event_count >= 3:
            return f"0 events where the last run had {record.last_event_count}"
        return None

    # ------------------------------------------------------------------ run

    async def run(self, venues: list[Venue], sources: SourcesFile, state: State,
                  social: SocialFile | None = None) -> RunReport:
        report = RunReport(run_date=self.today, sources_total=0, sources_ok=0, sources_skipped_unchanged=0,
                           events_published=0, events_new=0, events_rejected_grounding=0,
                           events_rejected_verifier=0, events_uncertain=0)
        before = set(state.events)
        expire_past(state, self.today)
        with make_client() as client:
            for venue in venues:
                if not venue.enabled:
                    continue
                log.info("[%s] %s", venue.id, venue.name)
                if venue.web:
                    report.sources_total += 1
                    outcome = await self.process_venue(venue, sources, state, client)
                    if self.enrich and outcome.events and any(e.title_truncated for e in outcome.events):
                        stats = await asyncio.to_thread(complete_truncated_titles, outcome.events, state, client)
                        log.info("[%s] truncated titles: %s", venue.id, stats.describe())
                    report.llm_calls += outcome.llm_calls
                    report.events_rejected_grounding += outcome.rejected_checks
                    report.events_rejected_verifier += outcome.rejected_verifier
                    report.events_uncertain += outcome.uncertain
                    record = sources.sources.get(venue.id)
                    source_url = record.source.url if record else f"venue:{venue.id}"
                    if outcome.events is not None:
                        report.sources_ok += 1
                    report.alerts.extend(merge_source(state, source_url, venue.id, outcome.events, self.today,
                                                      outcome.content_hash, outcome.error))
                    for n in outcome.notes:
                        report.alerts.append(Alert(level="warning", venue_id=venue.id, source_url=source_url, message=n))
                if social is not None and self.settings.has_social and venue.social:
                    await self.process_social(venue, sources, social, state, client, report)
                elif not venue.social:
                    gone = forget_events(state, venue.id, {"instagram", "facebook"})
                    if gone:
                        log.info("[%s] social disabled: %d social event(s) forgotten", venue.id, gone)
        state.last_run = self.today
        report.events_published = len(state.events)
        report.events_new = len(set(state.events) - before)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marseille-agenda", description="Update the Marseille venues agenda.")
    parser.add_argument("--venue", action="append", help="only process these venue ids")
    parser.add_argument("--no-verify", action="store_true", help="skip the adversarial verifier")
    parser.add_argument("--no-llm", action="store_true", help="never call a model (schemas must already exist)")
    parser.add_argument("--rediscover", action="append", default=[], help="forget the source of these venue ids")
    parser.add_argument("--regenerate", action="append", default=[], help="forget the schema of these venue ids")
    parser.add_argument("--today", help="override today's date (YYYY-MM-DD)")
    parser.add_argument("--no-images", action="store_true", help="skip downloading event visuals into site/img/")
    parser.add_argument("--no-enrich", action="store_true", help="do not fetch event pages to complete truncated titles")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = load_settings()
    today = date.fromisoformat(args.today) if args.today else datetime.now(TZ).date()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    venues = load_venues(settings)
    if args.venue:
        venues = [v for v in venues if v.id in set(args.venue)]
    sources = load_sources(settings)
    state = load_state(settings.data_dir / "state.json")
    for vid in args.rediscover:
        sources.sources.pop(vid, None)
    for vid in args.regenerate:
        if vid in sources.sources:
            sources.sources[vid].schema_record = None
            sources.sources[vid].next_generation = None
    for vid in {*args.rediscover, *args.regenerate}:
        # Events produced by the old rules are not comparable with the new ones.
        gone = forget_events(state, vid, {"html", "json"})
        if gone:
            log.info("[%s] %d web event(s) forgotten before re-extraction", vid, gone)
    social = load_social(settings.data_dir / "social.json")
    if not settings.has_social:
        log.info("APIFY_API_KEY not set: Instagram / Facebook sources skipped")

    runner = Runner(settings, today, verify=not args.no_verify, allow_llm=not args.no_llm, enrich=not args.no_enrich)
    credits_before = fetch_credits(settings)
    try:
        report = asyncio.run(runner.run(venues, sources, state, social))
        if not args.no_images:
            # Publication-time step, deterministic: cache each published event's visual as a WebP.
            stats = sync_images(published_events(state), state, settings.site_dir, today)
            report.images_published = stats.cached + stats.downloaded
            report.images_failed = stats.failed
            log.info("images: %s", stats.describe())
            for (host, reason), n in stats.failures.most_common():
                log.info("image failures: %s x%d (%s)", host, n, reason)
    finally:
        save_sources(settings, sources)  # never lose a paid-for discovery/schema
        save_social(settings.data_dir / "social.json", social)  # nor a paid-for post analysis
        save_state(settings.data_dir / "state.json", state)
    report.credits_remaining_usd = fetch_credits(settings)
    if credits_before is not None and report.credits_remaining_usd is not None:
        report.run_cost_usd = round(max(0.0, credits_before - report.credits_remaining_usd), 4)

    write_events_json(settings.data_dir / "events.json", state, load_venues(settings), today)
    write_ics(settings.data_dir / "events.ics", state)
    write_report(settings.data_dir / "report.json", report)

    log.info("done: %d/%d sources ok, %d events published (%d new), rejected checks=%d verifier=%d uncertain=%d, "
             "llm calls=%d, posts analysed=%d, apify runs=%d, run cost=%s USD, credits left=%s USD",
             report.sources_ok, report.sources_total, report.events_published, report.events_new,
             report.events_rejected_grounding, report.events_rejected_verifier, report.events_uncertain, report.llm_calls,
             report.posts_analyzed, report.apify_runs, report.run_cost_usd, report.credits_remaining_usd)
    for a in report.alerts:
        log.log(logging.ERROR if a.level == "error" else logging.WARNING, "%s: %s", a.venue_id, a.message)
    return 1 if report.sources_total and report.sources_ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
