# marseille-agenda

Give it a list of places in Marseille, get a calendar of their upcoming events.

- **Input**: [`venues.json`](venues.json), a list of venue names (optionally a website hint, category, coordinates).
- **Output**: [`data/events.json`](data/events.json) and [`data/events.ics`](data/events.ics), plus the events' visuals cached in `site/img/`, published as a full-page calendar on GitHub Pages.
- **Schedule**: a GitHub Actions cron runs every morning and commits the updated data.

Companion of the [marseille-tes-lieux](https://github.com/tfrere/marseille-tes-lieux) map: `map_name` links each venue to a `PLACES` entry there.

## How it works

The pipeline is agentic where it has to be and deterministic everywhere else, so a venue whose page did not change costs zero tokens.

```
venues.json ─┬─► 1. discover   (agent + tools, once per venue)   ─► data/sources.json  (url, kind)
             ├─► 2. induce     (no LLM) or generate (agent), once  ─► data/sources.json  (extraction schema)
             └─► 3. daily run  (no LLM)  fetch ─► execute schema ─► deterministic checks
                                                          │
                                   new HTML events only ──► 4. adversarial verifier (cheap model)
                                                          │
                                                          ▼
                                        5. merge with state ─► data/events.json / events.ics / report.json
```

1. **Discovery** (`discover.py`). A tool-using agent finds the URL that lists the venue's upcoming events. The heavy lifting is one deterministic tool, `scan_site`: home page, agenda-like pages with date snippets tagged upcoming/past, API routes found in the JS bundles (including lazily loaded chunks), and the event-looking routes probed right away, also through the proxy prefixes the front-end defines (`/wp-json/<ns>/v1/...` -> `/api/<name>/...`). The agent picks and verifies, under a hard budget (12 requests, tools refuse to work after step 8). It prefers JSON APIs over HTML. Typical cost: $0.03-0.17 per venue.
2. **Schema induction, then generation.** Most agenda pages are a list of repeated cards, each with a date, a title and a link; finding them is counting, not language. `induce.py` takes every element whose short text or `data-date`/`datetime` attribute is an upcoming date, walks up its ancestors and keeps the selector shared by the most of them without nesting; inside the cards the title is the heading or link that varies from card to card, the URL its link, the date the machine attribute when there is one or the dated element otherwise (two `<time>` per card is a run, not two events). Flat listings with no wrapper per event (SPIP and old sites: a date block, then the title block, then the next date) are handled by the schema's `item_mode: "run"`, where each item is a date node plus its following siblings up to the next one; the induction falls back to it when no wrapper is found and derives the fields, `location` included, from those runs (Mille Bâbords). An aggregator listing events held elsewhere can be restricted with `Venue.location_filter`, a case- and accent-insensitive regex the event's location must match (events without a location are kept). The result is an ordinary schema, executed and judged with the same engine and grounding checks as a model-written one; it is kept when it covers the dates found on the page with distinct titles (`generator_model: "induction"`, quality >= 0.8). Six of the eight listing fixtures in `tests/` are induced this way with zero model calls, at zero cost, and a broken schema on such a page heals itself the same way. A discovered agenda page whose server-rendered text holds no upcoming date at all is treated as a client-rendered shell (the cards are injected by JavaScript), not as an empty agenda: the discovered `fallback_url`, then the site home, are fetched and the source switches to the first one that induces a schema (La Fabulerie's `/agenda/` is such a shell; its home lists the same 22 cards). When a page cuts its titles short ("Le Grand Voyage aux Fertiles..."), the trailing marker is removed and the event's own page is read once (`enrich.py`, at most 20 fetches per venue per run, remembered in the state): its `og:title`, `h1` or `<title>` (minus the site name) replaces the prefix only if it starts with it and is longer, so the published title is verbatim on the linked page and verified to extend what the listing showed; otherwise the prefix stays. Either way the event carries `title_truncated: true`. Pages that are not listings (prose, two events, client-rendered shells with no better page) fall through to the generator agent (`schema_gen.py`): a second agent writes a *declarative* extraction schema (CSS selectors or JSON paths, date formats, exclusions). Nothing generated is executed as code: the schema is data interpreted by `apply.py`. The schema is accepted only if, executed on the same document, it agrees with an independent grounded LLM extraction of that document (`extract.py`): Jaccard >= 0.7 on normalized titles AND every (title, date) pair the reader found is produced by the schema; start times found by the reader must not be lost. Disagreements are fed back to the generator; after three failed attempts the venue is flagged. If the reader finds no upcoming event at all (the venue simply has nothing announced), the source is kept with zero events and generation is retried a week later instead of failing every day.
3. **Daily execution** (`apply.py`, `dates.py`, `validate.py`). Schemas run without any model. French dates without a year are resolved to the next occurrence, constrained by the weekday when one is written ("mardi 15 septembre" must be a Tuesday). Every event must pass: evidence quote verbatim in the page, title verbatim, URL present in the page, date not in the past and not implausibly far, weekday consistent. Two things the engine does on its own because generators keep getting them wrong on day-by-day agendas (Friche la Belle de Mai, cinemas): when the selected date text holds only a time ("14h-19h"), it falls back to the whole card, then to `data-date` / `datetime` attributes on the card or its parents; and `data-date` values are surfaced in the page text as `[2026-09-13]` markers so the reader, the schema and the verifier all see the same date. An item repeated under every day it runs (an exhibition) is folded into one entry with a date range, weekly closing days included.
4. **Adversarial verification** (`verify.py`). New events coming from free-form HTML are handed to a different model family, which only sees the excerpts around the evidence and tries to refute the event (past-events section, wrong year, opening hours mistaken for an event, cancellation). Only `accept` gets published. Structured JSON sources skip this step.
5. **State** (`merge.py`). Stable UIDs (`sha1(venue|title|date)`), `first_seen`/`last_seen`, past events expire, events missing for 3 runs are dropped, a sudden collapse of a source (0 events, or >50% drop) keeps the last verified state and raises an alert. A broken schema triggers regeneration; three consecutive failures trigger re-discovery. Errors open a GitHub issue. At publication time (`output.py`) the same real-world event announced on the website, Facebook and Instagram under different titles is folded into one entry: same venue and day, same clock time when both have one, and enough shared title words; the most structured source wins (JSON > HTML > Facebook > Instagram).
6. **Images** (`images.py`). Each published event may carry its own visual (poster, photo of the show), found deterministically first: an optional `image` field in the schema (`img[src]`, lazy-loading attributes and `srcset` handled by the engine), induced when most cards have a picture about as varied as their titles; without the field the engine takes the first picture of the card, or of the JSON item (`main_image`, `thumbnail`...). A picture shared by more than half of a source's events (venue logo, default thumbnail) is dropped for all of them. The generator agent only writes the field as a fallback, under one rule: an image that is not the event's own visual is a mismatch and worthless as a thumbnail, so missing is better than wrong. Social events take the first image of their post. At publication time the visuals are downloaded once, resized to 640 px and cached as `site/img/<uid>.webp` (Instagram and Facebook CDN URLs expire within days); dead URLs are remembered and retried after two weeks, files of unpublished events are removed, and an event with no trustworthy image has `image: null` so the site shows an honest placeholder. Skip the download with `--no-images`.

## Instagram and Facebook

Many small venues only announce on social media, usually as a flyer image with a one-line caption. With an Apify token (`APIFY_API_KEY`) each venue can also have social sources (`social.py`, `social_extract.py`, `apify.py`):

- **Where the handles come from**: `instagram` / `facebook` in `venues.json`, or the accounts the discovery agent saw linked from the venue's website (`scan_site` reports them; the agent never spends a tool call on them). `"web": false` marks a venue that has no web agenda at all.
- **Fetch**: Meta blocks anonymous clients (401/400 even from residential IPs), so posts are read through Apify's maintained actors (`instagram-api-scraper`, `facebook-posts-scraper`, and the structured `facebook-events-scraper` for a page's "upcoming hosted events" tab, which needs no model at all). Posts are requested daily over a short window (7-30 days, adjusted to the last success), the events tab weekly.
- **Reading a post**: a vision model (`qwen/qwen3-vl-32b-instruct`) gets the caption and up to 4 images and lists the events announced *at the venue*. Deterministic checks then apply: caption-grounded fields must be verbatim in the caption; image-grounded fields require an image; the event must be on or after the publication date and less than 300 days after it (a wrong year is always >= 365 days off, so this pins the year and rejects recaps of past events); a weekday written in the evidence must match; URLs not present in the caption are dropped, not trusted.
- **Adversarial verification**: every surviving event is challenged by a second vision model of another family (`deepseek/deepseek-v4.1-flash`) that sees the same post and images. Only `accept` is published.
- **Cache**: a post is analysed exactly once; `data/social.json` stores the verdicts per post, the venue's social events are rebuilt from it every run, and posts older than 180 days are pruned. A failed Apify run keeps the cached events and alerts after two consecutive failures. If the website already published the same event that run, the social copy is skipped (the website is canonical).

Model choice is measured, not guessed: `tests/test_live.py` reads a synthetic flyer whose year is written nowhere and runs the verifier on a recap trap and a wrong-time trap. Qwen3-VL and Mistral Small read the flyer 3/3; DeepSeek V4.1 Flash read it correctly but returned an empty list (nested structured output), while as a verifier it was 8/8 alongside Qwen3-VL and Gemini 3.1 Flash Lite; Gemini 2.5 Flash Lite missed the wrong time and GPT-5.4 nano hallucinated a contradiction.

## Cost

OpenRouter, defaults: `anthropic/claude-sonnet-5` for discovery/generation/reference extraction, `google/gemini-3.8-flash` as verifier. Measured: $0.03-0.17 for discovery and $0.10-0.30 for schema generation per venue, *once*, then a few cents per new HTML event. Unchanged venues cost nothing. Every discovery tool call is logged with its size (`-v`), and the run report carries the run cost and the remaining credits (shown in the calendar header).

Social sources: about 2,000 input tokens per post per model, so roughly $0.002 per post for extraction plus verification. Apify bills per result: ~$1.40 / 1000 Instagram posts, ~$2 / 1000 Facebook posts, ~$13 / 1000 Facebook events, which for a venue posting a few times a week is $0.2-0.5 per month on Instagram and about $1 per month for the weekly events tab; the free Apify plan includes $5 per month.

## Run locally

```bash
cp .env.example .env            # add OPENROUTER_API_KEY (and APIFY_API_KEY for Instagram / Facebook)
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

Optional fields: `"website"` (discovery hint), `"instagram"` (handle or URL), `"facebook"` (page slug or URL), `"web": false` for a venue with no web agenda, `"social": false` to ignore social media even when discovery finds handles (typically an account shared by several venues, like `museesmarseille` for every municipal museum).

When the generator cannot produce a schema for a page (it happened once, on lafriche.org, before the fallbacks above existed), a hand-written schema in `data/sources.json` with `"generator_model": "manual"` is honoured like any other and still goes through the daily health checks and the verifier.

## Deploy

GitHub Actions + Pages. Requirements: repository secrets `OPENROUTER_API_KEY` and, for social sources, `APIFY_API_KEY`; Pages source set to "GitHub Actions".
