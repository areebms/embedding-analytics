# scrape

*Stage 1 of 5. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Requests

Fetches a Project Gutenberg book by ID and writes it to S3 exactly as fetched, in two
steps that advance `status`, plus a third that seeds the books to run them on:

1. `scrape_subject_book_list` — every book ID in a Gutenberg subject, seeded at
   `LISTED`. The other two stages refuse to run on a book this has not created.
2. `scrape_book_metadata` — the bibrec table. Non-English books are marked
   `SCRAPED_SKIPPED_NON_ENGLISH` and go no further.
3. `scrape_book_content` — the book HTML, stored **raw**, license boilerplate and all.

Every step is idempotent — the per-book pair on `status`, the seeding on a
conditional create — so a re-run skips work already done.

This is the only stage that fetches the corpus over the network, and it derives
nothing. Turning that HTML into readable text is
[standardize-html](../standardize-html/)'s job — its `SEND` and `RETRIEVE`
stages — so changing how an artifact is rendered costs a re-run over `html/` rather
than a refetch.

| S3 artifact | Contents |
|---|---|
| `metadata/{index}.json` | Title, author, publication metadata |
| `html/{index}.html` | Raw HTML, exactly as fetched |

## One Lambda, one stage per invocation

The handler runs **one** stage per call, chosen by the event:

```json
{ "index": "gutenberg-3300", "stage": "METADATA" }
{ "index": "gutenberg-3300", "stage": "CONTENT" }
{ "subject": "12345",        "stage": "SUBJECT" }
```

`stage` is validated first, then whatever that stage needs — `METADATA` and `CONTENT`
take an `index`, `SUBJECT` takes a `subject`. An event missing its argument is rejected
rather than half-run, and any stage name outside the three is rejected rather than run.

The two per-book stages reply with the status the book ended at, which is what lets a
state machine branch:

```json
{ "stage": "METADATA", "book_id": "gutenberg-3300", "status": "SCRAPED_METADATA" }
```

The per-book state machine therefore invokes this function twice, with a `Choice`
between the two calls that ends the execution for a book marked
`SCRAPED_SKIPPED_*` instead of sending it on to `tokenize`.

The subject machine (`${ENV_PREFIX}-scrape`, deployed by the
`${ENV_PREFIX}-scrape-pipeline` stack) invokes it once for the subject,
then twice per book inside a `Map` running at `MaxConcurrency: 1`. The `SUBJECT` stage can
also be driven on its own by the `aws lambda invoke` below, or by the CLI. See
[Operations § Orchestration](../../docs/operations.md#orchestration).

## Seeding is a stage, not a prerequisite step

Both per-book stages read the book's current `status` and refuse to guess: a
book with no pipeline entry raises rather than creating one. Seeding is the `SUBJECT`
stage's job — and the first thing the subject machine
(`infra/scrape-pipeline.step-function.template.json`) runs:

```bash
aws lambda invoke --function-name $ENV_PREFIX-scrape \
    --payload '{"stage":"SUBJECT","subject":"12345"}' out.json
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
per-book stages are idempotent on `status`, re-running over an already-scraped
book costs one status read — which is exactly what lets a re-run resume a subject that
only got part of the way through.

The listing stage is idempotent too, at the cap. A subject that already holds
`MAX_BOOKS_PER_SUBJECT` books is served from the pipeline table and never listed again:
the cap is the ceiling on what it contributes, so re-walking could only re-rank the same
books, never add one. On that path `found` is what the table holds, `created` is `0` —
meaning the listing was never walked, not that it was walked and yielded nothing — and
`indexes` is a superset of every earlier run's work list. That last part matters: the
listing is ranked by downloads, so re-walking it used to drop a book that had drifted
past rank 100, stranding it at `SCRAPED_HTML` where no later stage would pick it up.

> **Subject size cap.** `get_book_ids` walks the subject sorted by download count and
> stops at `MAX_BOOKS_PER_SUBJECT` (100), so a large subject contributes its 100 most
> read books rather than all of them. At 1s per 25-book page that is four pages, well
> inside the 120s default timeout; a cap raised past roughly 800 books would outlast the
> invocation, and should be seeded with the CLI instead.

`scrape.py` runs standalone for seeding and bulk backfills:

```bash
python scrape.py SUBJECT --subject 12345   # seed pipeline entries from a subject
python scrape.py METADATA                  # LISTED -> SCRAPED_METADATA
python scrape.py CONTENT                   # SCRAPED_METADATA -> SCRAPED_HTML
```

The `METADATA` and `CONTENT` subcommands sweep every book sitting at the status that
stage consumes, pausing between books; one book failing does not stop the rest.

The `lambda-scrape` compose service builds the deployed image, so its entrypoint is the
Lambda runtime and a bare `python …` argument would be read as a handler name. Override
the entrypoint to reach the CLI:

```bash
docker compose run --rm --entrypoint python lambda-scrape scrape.py SUBJECT --subject 12345
```

`src/` and `shared/` are bind-mounted into the image, so edits apply without a rebuild.
