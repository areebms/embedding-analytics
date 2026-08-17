# scrape

*Stage 1 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Requests

Fetches a Project Gutenberg book by ID and writes it to S3 exactly as fetched, in two
steps that advance `pipeline_status`, plus a third that seeds the books to run them on:

1. `scrape_subject_book_list` — every book ID in a Gutenberg subject, seeded at
   `CREATED`. The other two stages refuse to run on a book this has not created.
2. `scrape_book_metadata` — the bibrec table. Non-English books are marked
   `SCRAPED_SKIPPED_NON_ENGLISH` and go no further.
3. `scrape_book_content` — the book HTML, stored **raw**, license boilerplate and all.

Every step is idempotent — the per-book pair on `pipeline_status`, the seeding on a
conditional create — so a re-run skips work already done.

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
{ "subject": "12345",        "stage": "list" }
```

`stage` is validated first, then whatever that stage needs — `metadata` and `content`
take an `index`, `list` takes a `subject`. An event missing its argument is rejected
rather than half-run, and any stage name outside the three is rejected rather than run.

The two per-book stages reply with the status the book ended at, which is what lets a
state machine branch:

```json
{ "index": "gutenberg-3300", "status": "SCRAPED_METADATA" }
```

The per-book state machine therefore invokes this function twice, with a `Choice`
between the two calls that ends the execution for a book marked
`SCRAPED_SKIPPED_*` instead of sending it on to `tokenize`.

A second machine would invoke it once for the subject, then twice per book inside a
`Map` running at `MaxConcurrency: 1` — but that one is designed and not deployed, with
no deploy path yet. Today the `list` stage is driven by the `aws lambda invoke` below,
or by the CLI. See [Operations § Orchestration](../../docs/operations.md#orchestration).

## Seeding is a stage, not a prerequisite step

Both per-book stages read the book's current `pipeline_status` and refuse to guess: a
book with no pipeline entry raises rather than creating one. Seeding is the `list`
stage's job — and the first thing the undeployed scrape machine
(`infra/scrape-pipeline.step-function.template.json`) would run:

```bash
aws lambda invoke --function-name $LAMBDA_PREFIX-scrape \
    --payload '{"stage":"list","subject":"12345"}' out.json
```

```json
{
  "subject": "12345",
  "found": 100,
  "created": 87,
  "indexes": ["gutenberg-3300", "gutenberg-846", "..."]
}
```

`found` is what the walk returned, so it is capped at 100 however large the subject is.
`indexes` is **every** book in that set, not only the newly created ones. Because the
per-book stages are idempotent on `pipeline_status`, re-running over an already-scraped
book costs one status read — which is exactly what lets a re-run resume a subject that
only got part of the way through.

> **Subject size cap.** `get_book_ids` walks the subject sorted by download count and
> stops at `MAX_BOOKS_PER_SUBJECT` (100), so a large subject contributes its 100 most
> read books rather than all of them. At 1s per 25-book page that is four pages, well
> inside the 120s default timeout; a cap raised past roughly 800 books would outlast the
> invocation, and should be seeded with the CLI instead.

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
