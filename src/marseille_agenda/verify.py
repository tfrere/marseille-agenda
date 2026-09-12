"""Adversarial verifier: a second, independent model tries to refute each extracted event.

It only sees the source excerpts around the evidence quotes (not the whole page), so it
cannot "find" support elsewhere; its job is to look for contradictions the deterministic
checks cannot see (wrong year inferred, event listed under a past-events heading, opening
hours mistaken for an event, cancellation notes, session vs festival dates, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models import Model

from .fetch import SourceDocument
from .schema import ExtractedEvent, VerifierVerdict
from .validate import FRENCH_WEEKDAYS, evidence_context

INSTRUCTIONS = """\
You are an adversarial fact-checker for a public events calendar in Marseille, France.
Another system extracted ONE event from a web page. Your job is to try to prove it wrong
using only the source excerpts provided. Be sceptical but fair.

Reject when any of these holds:
- The excerpt places this item under a past-events / archive / video section, or the text
  reports on it in the past tense.
- The date, time or weekday in the excerpt contradicts the extracted values (check the weekday
  name against the ISO date; today's date is given so you can judge the year inference).
- It is not a dated public event: opening hours, generic recurring schedule with no concrete date,
  a call for proposals, a membership offer, a product, a newsletter.
- The excerpt says the event is cancelled/postponed but the extracted status says scheduled.
- The extracted URL is not shown next to this event in the excerpt.

Return "uncertain" when the excerpt is genuinely ambiguous (e.g. no year and no weekday to
confirm it, or the heading context is cut off). Return "accept" only when every check passes.
`reason` must quote the decisive words from the excerpt.
"""


@dataclass
class VerifyDeps:
    today: date


def build_verifier(model: Model | str) -> Agent[VerifyDeps, VerifierVerdict]:
    return Agent(
        model,
        deps_type=VerifyDeps,
        output_type=VerifierVerdict,
        instructions=INSTRUCTIONS,
        retries=1,
        name="verifier",
    )


def verifier_prompt(ev: ExtractedEvent, doc: SourceDocument, venue_name: str, today: date) -> str:
    weekday = FRENCH_WEEKDAYS[today.weekday()]
    ev_weekday = FRENCH_WEEKDAYS[ev.start_date.weekday()]
    return (
        f"Today is {weekday} {today.isoformat()}. Venue: {venue_name}. Page: {doc.url}\n\n"
        f"EXTRACTED EVENT:\n{ev.model_dump_json(indent=1, exclude={'evidence'})}\n"
        f"(start_date {ev.start_date} is a {ev_weekday})\n\n"
        f"EVIDENCE QUOTES:\n" + "\n".join(f"- {q}" for q in ev.evidence) + "\n\n"
        f"SOURCE EXCERPTS AROUND THE EVIDENCE:\n{evidence_context(ev, doc)}"
    )


async def verify_event(
    agent: Agent[VerifyDeps, VerifierVerdict],
    ev: ExtractedEvent,
    doc: SourceDocument,
    venue_name: str,
    today: date,
):
    return await agent.run(verifier_prompt(ev, doc, venue_name, today), deps=VerifyDeps(today=today))
