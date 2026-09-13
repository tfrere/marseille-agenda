"""Grounded event extraction from free-form web pages with Pydantic AI.

The agent returns structured `ExtractionResult`. An output validator re-checks every
event against the source text (`validate.check_event`). On the first failures the model
is asked to retry with the exact list of problems; if problems persist after retries,
the offending events are dropped rather than published.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from .config import MAX_SOURCE_CHARS
from .fetch import SourceDocument
from .schema import ExtractedEvent, ExtractionResult
from .validate import FRENCH_WEEKDAYS, check_event

INSTRUCTIONS = """\
You extract upcoming public events from the text of ONE web page belonging to a venue or
organisation in Marseille, France. The text is in French. Output structured data only.

Hard rules (violations are detected automatically and the whole answer is sent back to you):
1. COPY, never paraphrase: `title`, `evidence` and `url` must be verbatim substrings of the page.
   Keep the original casing and punctuation. Evidence quotes are 20 to 300 characters each and
   must contain the date/time wording you relied on. When the page groups entries under a day
   heading ("Aujourd'hui, dimanche 13 septembre", "Mardi 15 septembre") and the entry itself
   only shows a time ("14h-19h"), give two quotes: the day heading and the entry's own line.
2. Only UPCOMING events: start date >= today, or multi-day events still running today.
   Ignore sections titled "passés", "archives", "précédents", "retour sur", "revoir", "vidéo",
   and anything clearly reporting on something that already happened.
3. Not events: opening hours, membership offers, generic "every Tuesday" schedules with no dates,
   calls for participation, newsletters, merchandise, past-event videos.
4. Dates without a year: pick the year that makes the date the NEXT occurrence on or after today,
   then double-check the weekday if the page names one (e.g. "mardi 15 septembre" must be a Tuesday).
   If the weekday does not fit any plausible year, do not output the event.
5. `url`: only a link that appears in the page text as `(https://...)`, pointing to this event's
   detail or ticket page. Otherwise null. Never invent or "fix" URLs.
6. `location_name`: the place exactly as written if the event happens somewhere specific
   (another venue, a room). Null if not written.
7. If the page has no upcoming events, return an empty list and explain in `notes`.
8. One entry per distinct event date. Recurring workshops with several listed dates: one entry per date.

Today's date and the page URL are given in the user message.
"""


@dataclass
class ExtractDeps:
    doc: SourceDocument
    today: date
    venue_name: str
    rejected: list[tuple[ExtractedEvent, list[str]]] = field(default_factory=list)


def build_extractor(model: Model | str) -> Agent[ExtractDeps, ExtractionResult]:
    agent: Agent[ExtractDeps, ExtractionResult] = Agent(
        model,
        deps_type=ExtractDeps,
        output_type=ExtractionResult,
        instructions=INSTRUCTIONS,
        retries=2,
        name="extractor",
    )

    @agent.output_validator
    async def grounded(ctx: RunContext[ExtractDeps], output: ExtractionResult) -> ExtractionResult:
        deps = ctx.deps
        good: list[ExtractedEvent] = []
        bad: list[tuple[ExtractedEvent, list[str]]] = []
        seen: set[tuple[str, date]] = set()
        for ev in output.events:
            problems = check_event(ev, deps.doc, deps.today)
            key = (ev.title.strip().lower(), ev.start_date)
            if key in seen:
                problems.append("duplicate of another event in this answer (same title and date)")
            seen.add(key)
            if problems:
                bad.append((ev, problems))
            else:
                good.append(ev)
        if bad and ctx.retry < ctx.max_retries:
            lines = [
                f"- {ev.title!r} ({ev.start_date}): " + "; ".join(p) for ev, p in bad
            ]
            raise ModelRetry(
                "Some events failed verification against the page text. Fix them (copy text verbatim, "
                "re-check the year/weekday) or drop them, then return the full corrected list. "
                "For 'evidence too short': quote a longer span, e.g. the day heading that carries the date "
                "as one quote and the entry's title/time line as another.\n" + "\n".join(lines)
            )
        deps.rejected = bad
        return ExtractionResult(events=good, notes=output.notes)

    return agent


def user_prompt(deps: ExtractDeps) -> str:
    text = deps.doc.text
    if len(text) > MAX_SOURCE_CHARS:
        text = text[:MAX_SOURCE_CHARS] + "\n[... truncated ...]"
    weekday = FRENCH_WEEKDAYS[deps.today.weekday()]
    return (
        f"Today is {weekday} {deps.today.isoformat()}.\n"
        f"Venue: {deps.venue_name}\n"
        f"Page URL: {deps.doc.url}\n\n"
        f"=== PAGE TEXT START ===\n{text}\n=== PAGE TEXT END ==="
    )


async def extract_events(agent: Agent[ExtractDeps, ExtractionResult], deps: ExtractDeps):
    """Run the extractor; returns the AgentRunResult (output + usage)."""
    return await agent.run(user_prompt(deps), deps=deps)
