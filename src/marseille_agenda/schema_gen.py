"""Schema generator: an agent that writes a declarative ExtractionSchema for one source,
validated on the spot by executing it and comparing with a direct LLM extraction.

Two independent methods agreeing on the same (title, date) pairs is the acceptance test.
The schema never publishes anything by itself before passing it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from .apply import ApplyResult, apply_schema
from .extract import ExtractDeps, ExtractionResult, build_extractor, extract_events
from .extraction_schema import ExtractionSchema, JsonRule
from .fetch import SourceDocument, condense_html, document_from_body, summarize_json
from .validate import check_event, normalize, strip_accents

log = logging.getLogger(__name__)

MIN_AGREEMENT = 0.7

INSTRUCTIONS = """\
You write a declarative extraction schema for a web source listing the upcoming events of one
venue in Marseille. The schema is executed by a fixed engine (no code of yours ever runs), so
you must describe the page structure precisely.

HTML sources (you receive a condensed skeleton of the page: tags, classes, ids, hrefs, text):
- `item_selector` must match exactly one node per event and nothing else.
- Use `container_selector` to restrict to the block of UPCOMING events when the page also lists
  past events, archives or news. Use `exclude_selectors` to remove such blocks otherwise.
- `fields.title`: selector of the title node (text). `fields.date`: node whose text contains the
  date words (French wording is parsed automatically: "mardi 15 septembre à 18 h 30", "du 6 au 7
  juin", "12 Sep 2026", ISO dates). If the date is spread over the item, use selector "" (the item).
  If a `datetime`/`content` attribute holds a machine date, use `attr` plus `date_format`.
- `fields.url`: the event link with attr "href". Add `location`, `price`, `summary` when present.
- Use `regex` (capture group 1) to isolate a value from surrounding text.

JSON sources (you receive a structural summary with sample items):
- `items_path`: dotted path to the list of events. If each item holds several dated sessions,
  set `expand_path` to that sub-list and prefix parent fields with "item.".
- `date_format`/`time_format`: strptime formats matching the raw values exactly ("%Y%m%d", "%H:%M").
- Booleans like free/paid map to the `free` field; lists of names can be joined with `join`.
- If items come in two shapes (e.g. dated sessions vs. a recurrence range), write one rule per shape.

The schema is executed immediately and compared with an independent reading of the same page.
If items fail to parse or the two disagree, you get the details and must fix the schema.
"""


@dataclass
class GenDeps:
    doc: SourceDocument
    ref_doc: SourceDocument
    """Document the independent reader saw (same as `doc`, or a trimmed JSON payload)."""
    today: date
    venue_name: str
    reference: set[tuple[str, date]]
    """(normalized title, start_date) pairs from the direct LLM extraction."""
    reference_events: ExtractionResult
    best: tuple[float, ExtractionSchema, ApplyResult] | None = None
    attempts: list[dict] = field(default_factory=list)


def _key(title: str, d: date) -> tuple[str, date]:
    return (strip_accents(normalize(title)), d)


def _agreement(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def build_generator(model: Model | str) -> Agent[GenDeps, ExtractionSchema]:
    agent: Agent[GenDeps, ExtractionSchema] = Agent(
        model, deps_type=GenDeps, output_type=ExtractionSchema, instructions=INSTRUCTIONS, retries=3, name="schema_generator",
    )

    @agent.output_validator
    async def executes_and_agrees(ctx: RunContext[GenDeps], schema: ExtractionSchema) -> ExtractionSchema:
        deps = ctx.deps
        for rule in schema.rules:
            if (rule.kind == "html") != (deps.doc.kind == "html"):
                raise ModelRetry(f"the source is {deps.doc.kind}; every rule must be of kind {deps.doc.kind!r}")
        try:
            res = apply_schema(schema, deps.doc, deps.today)
            res_ref = res if deps.ref_doc is deps.doc else apply_schema(schema, deps.ref_doc, deps.today)
        except Exception as exc:  # noqa: BLE001 - selector syntax errors etc.
            raise ModelRetry(f"schema could not be executed: {type(exc).__name__}: {exc}") from exc

        # Agreement is measured on the document the independent reader actually saw.
        valid = [ev for ev in res_ref.events if not check_event(ev, deps.ref_doc, deps.today, check_text=ev.grounded_text)]
        got = {_key(ev.title, ev.start_date) for ev in valid}
        agreement = _agreement(got, deps.reference)
        deps.attempts.append({"items": res.items_seen, "events": len(res.events), "valid": len(valid),
                              "failures": len(res.failures), "agreement": round(agreement, 3)})
        if deps.best is None or agreement > deps.best[0]:
            deps.best = (agreement, schema, res)

        problems: list[str] = []
        if res.items_seen == 0:
            problems.append("item_selector/items_path matched nothing")
        if res.items_seen and res.failure_ratio > 0.5 and deps.reference:
            problems.append(f"{len(res.failures)}/{res.items_seen} items failed: " + " | ".join(res.failures[:6]))
        if agreement < MIN_AGREEMENT:
            missing = deps.reference - got
            extra = got - deps.reference
            if missing:
                problems.append("events read by the independent reader but missed by the schema: "
                                + "; ".join(f"{t!r} {d}" for t, d in sorted(missing, key=lambda x: x[1])[:12]))
            if extra:
                problems.append("events produced by the schema but NOT seen by the independent reader "
                                "(often past events or non-events): "
                                + "; ".join(f"{t!r} {d}" for t, d in sorted(extra, key=lambda x: x[1])[:12]))
            if not missing and not extra:
                problems.append("no valid upcoming events produced")

        if problems and ctx.retry < ctx.max_retries:
            raise ModelRetry("Schema executed but failed validation. Fix and return the full schema.\n- " + "\n- ".join(problems))
        return schema

    return agent


def render_source(doc: SourceDocument) -> str:
    if doc.kind == "html":
        return condense_html(doc.raw)
    return summarize_json(doc.raw)


def reference_document(doc: SourceDocument, schema_hint: ExtractionSchema | None = None) -> SourceDocument:
    """Document handed to the independent LLM reader.

    HTML: the page itself (as text). JSON: the same payload, so both readers see identical data;
    large payloads are trimmed to the first items of the list when a schema tells us where it is.
    """
    if doc.kind == "html" or len(doc.raw) < 120_000:
        return doc
    if schema_hint:
        first = schema_hint.rules[0]
        if isinstance(first, JsonRule):
            try:
                data = json.loads(doc.raw)
                cur = data
                parts = [p for p in first.items_path.split(".") if p]
                for p in parts[:-1]:
                    cur = cur[p]
                if parts:
                    cur[parts[-1]] = cur[parts[-1]][:8]
                else:
                    data = data[:8]
                return document_from_body(doc.url, "json", json.dumps(data, ensure_ascii=False))
            except (KeyError, TypeError, IndexError):
                pass
    return document_from_body(doc.url, "json", summarize_json(doc.raw, max_chars=120_000, sample_items=8))


@dataclass
class GenerationOutcome:
    schema: ExtractionSchema
    agreement: float
    validated: bool
    apply_result: ApplyResult
    reference: ExtractionResult
    attempts: list[dict]
    llm_calls: int


async def generate_schema(
    generator: Agent[GenDeps, ExtractionSchema],
    extractor,
    doc: SourceDocument,
    venue_name: str,
    today: date,
) -> GenerationOutcome:
    # 1. Independent reading of the source by the grounded extractor.
    ref_doc = reference_document(doc)
    ref_deps = ExtractDeps(doc=ref_doc, today=today, venue_name=venue_name)
    ref_run = await extract_events(extractor, ref_deps)
    reference = {_key(e.title, e.start_date) for e in ref_run.output.events}
    log.info("reference extraction: %d events (%d rejected by grounding)", len(reference), len(ref_deps.rejected))

    # 2. Generate + validate.
    deps = GenDeps(doc=doc, ref_doc=ref_doc, today=today, venue_name=venue_name, reference=reference,
                   reference_events=ref_run.output)
    prompt = (
        f"Today is {today.isoformat()}. Venue: {venue_name}. Source URL: {doc.url} (kind: {doc.kind}).\n"
        f"The independent reader found {len(reference)} upcoming events, e.g.: "
        + "; ".join(f"{t!r} on {d}" for t, d in sorted(reference, key=lambda x: x[1])[:6])
        + f"\n\n=== SOURCE ({doc.kind}) ===\n{render_source(doc)}\n=== END ==="
    )
    calls = ref_run.usage().requests
    try:
        run = await generator.run(prompt, deps=deps)
        calls += run.usage().requests
        schema = run.output
    except Exception as exc:  # noqa: BLE001 - retries exhausted etc.
        log.warning("generator did not converge: %s", exc)
        if deps.best is None:
            raise
        schema = deps.best[1]
    res = apply_schema(schema, doc, today)
    res_ref = res if ref_doc is doc else apply_schema(schema, ref_doc, today)
    valid = [ev for ev in res_ref.events if not check_event(ev, ref_doc, today, check_text=ev.grounded_text)]
    agreement = _agreement({_key(e.title, e.start_date) for e in valid}, reference)
    return GenerationOutcome(
        schema=schema, agreement=agreement, validated=agreement >= MIN_AGREEMENT and bool(valid),
        apply_result=res, reference=ref_run.output, attempts=deps.attempts, llm_calls=calls,
    )
