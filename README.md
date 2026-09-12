# marseille-agenda

Give it a list of places in Marseille, get a calendar of their upcoming events.

- **Input**: [`venues.json`](venues.json), a list of venue names (optionally a website hint, category, coordinates).
- **Output**: [`data/events.json`](data/events.json) and [`data/events.ics`](data/events.ics), published as a full-page calendar on GitHub Pages.
- **Schedule**: a GitHub Actions cron runs every morning and commits the updated data.

Companion of the [marseille-tes-lieux](https://github.com/tfrere/marseille-tes-lieux) map: `map_name` links each venue to a `PLACES` entry there.

## How it works

The pipeline is agentic where it has to be and deterministic everywhere else, so a venue whose page did not change costs zero tokens.

```
venues.json ─┬─► 1. discover   (agent + tools, once per venue)   ─► data/sources.json  (url, kind)
             ├─► 2. generate   (agent, once per source)          ─► data/sources.json  (extraction schema)
             └─► 3. daily run  (no LLM)  fetch ─► execute schema ─► deterministic checks
                                                          │
                                   new HTML events only ──► 4. adversarial verifier (cheap model)
                                                          │
                                                          ▼
                                        5. merge with state ─► data/events.json / events.ics / report.json
```

1. **Discovery** (`discover.py`). A tool-using agent finds the URL that lists the venue's upcoming events. The heavy lifting is one deterministic tool, `scan_site`: home page, agenda-like pages with date snippets tagged upcoming/past, API routes found in the JS bundles (including lazily loaded chunks), and the event-looking routes probed right away, also through the proxy prefixes the front-end defines (`/wp-json/<ns>/v1/...` -> `/api/<name>/...`). The agent picks and verifies, under a hard budget (12 requests, tools refuse to work after step 8). It prefers JSON APIs over HTML. Typical cost: $0.03-0.17 per venue.
2. **Schema generation** (`schema_gen.py`). A second agent writes a *declarative* extraction schema (CSS selectors or JSON paths, date formats, exclusions). Nothing generated is executed as code: the schema is data interpreted by `apply.py`. The schema is accepted only if, executed on the same document, it agrees with an independent grounded LLM extraction of that document (`extract.py`): Jaccard >= 0.7 on normalized titles AND every (title, date) pair the reader found is produced by the schema; start times found by the reader must not be lost. Disagreements are fed back to the generator; after three failed attempts the venue is flagged. If the reader finds no upcoming event at all (the venue simply has nothing announced), the source is kept with zero events and generation is retried a week later instead of failing every day.
3. **Daily execution** (`apply.py`, `dates.py`, `validate.py`). Schemas run without any model. French dates without a year are resolved to the next occurrence, constrained by the weekday when one is written ("mardi 15 septembre" must be a Tuesday). Every event must pass: evidence quote verbatim in the page, title verbatim, URL present in the page, date not in the past and not implausibly far, weekday consistent.
4. **Adversarial verification** (`verify.py`). New events coming from free-form HTML are handed to a different model family, which only sees the excerpts around the evidence and tries to refute the event (past-events section, wrong year, opening hours mistaken for an event, cancellation). Only `accept` gets published. Structured JSON sources skip this step.
5. **State** (`merge.py`). Stable UIDs (`sha1(venue|title|date)`), `first_seen`/`last_seen`, past events expire, events missing for 3 runs are dropped, a sudden collapse of a source (0 events, or >50% drop) keeps the last verified state and raises an alert. A broken schema triggers regeneration; three consecutive failures trigger re-discovery. Errors open a GitHub issue.

## Cost

OpenRouter, defaults: `anthropic/claude-sonnet-5` for discovery/generation/reference extraction, `google/gemini-3.8-flash` as verifier. Measured: $0.03-0.17 for discovery and $0.10-0.30 for schema generation per venue, *once*, then a few cents per new HTML event. Unchanged venues cost nothing. Every discovery tool call is logged with its size (`-v`), and the run report carries the run cost and the remaining credits (shown in the calendar header).

## Run locally

```bash
cp .env.example .env            # add OPENROUTER_API_KEY
uv sync
uv run pytest                   # offline tests (fixtures, engine, pipeline behaviour)
uv run pytest -m live -s        # acceptance tests calling real models
uv run marseille-agenda         # full run -> data/
uv run marseille-agenda --no-llm                 # execute known schemas only
uv run marseille-agenda --regenerate mucem       # force a new schema for one venue
uv run marseille-agenda --rediscover mucem       # forget the source and search again
```

Preview the calendar: serve `site/` next to `data/` (the page loads `./events.json` then `../data/events.json`).

## Adding a venue

Add `{"name": "..."}` to `venues.json` and push (or wait for the next cron). The workflow discovers the source, writes the schema into `data/sources.json` and commits the events. Check the run summary for alerts. `data/sources.json` is meant to be read and, if needed, hand-edited: it is the contract between the agents and the daily run.

## Deploy

GitHub Actions + Pages. Requirements: repository secret `OPENROUTER_API_KEY`, Pages source set to "GitHub Actions".
