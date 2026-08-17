# scrape

*Stage 1 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Requests

Fetches a Project Gutenberg book by ID and writes it to S3 exactly as fetched, in two
steps that advance `pipeline_status`:

1. `scrape_book_metadata` — the bibrec table. Non-English books are marked
   `SCRAPED_SKIPPED_NON_ENGLISH` and go no further.
2. `scrape_book_content` — the book HTML, stored **raw**, license boilerplate and all.

Each step is idempotent on `pipeline_status`, so a re-run skips work already done.

This is the only stage that fetches the corpus over the network, and it derives
nothing. Turning that HTML into readable text is
[standardize-submit](../standardize-submit/) and
[standardize-collect](../standardize-collect/)'s job, so changing how an artifact is
rendered costs a re-run over `html/` rather than a refetch.

| S3 artifact | Contents |
|---|---|
| `metadata/{index}.json` | Title, author, publication metadata |
| `html/{index}.html` | Raw HTML, exactly as fetched |

## One Lambda, one stage per invocation

The handler runs **one** stage per call, chosen by the event:

```json
{ "index": "gutenberg-3300", "stage": "metadata" }
{ "index": "gutenberg-3300", "stage": "content" }
```

Both keys are required; an event missing either is rejected rather than half-run. The
reply carries the status the book ended at, which is what lets the state machine branch:

```json
{ "index": "gutenberg-3300", "status": "SCRAPED_METADATA" }
```

The per-book state machine therefore invokes this function twice, with a `Choice`
between the two calls that ends the execution for a book marked
`SCRAPED_SKIPPED_NON_ENGLISH` instead of sending it on to `tokenize`.

`metadata` and `content` are the only stages the handler accepts; any other value is
rejected rather than run.

## Seeding comes first

Both stages read the book's current `pipeline_status` and refuse to guess: a book with no
pipeline entry raises rather than creating one. Seed the subject before invoking either
stage.

`scrape.py` runs standalone for seeding and bulk backfills:

```bash
python scrape.py list --subject 12345   # seed pipeline entries from a subject
python scrape.py metadata               # CREATED -> SCRAPED_METADATA
python scrape.py content                # SCRAPED_METADATA -> SCRAPED_HTML
```

The `metadata` and `content` subcommands sweep every book sitting at the status that
stage consumes, pausing between books; one book failing does not stop the rest.

The `lambda-scrape` compose service builds the deployed image, so its entrypoint is the
Lambda runtime and a bare `python …` argument would be read as a handler name. Override
the entrypoint to reach the CLI:

```bash
docker compose run --rm --entrypoint python lambda-scrape scrape.py list --subject 12345
```

`src/` and `shared/` are bind-mounted into the image, so edits apply without a rebuild.
