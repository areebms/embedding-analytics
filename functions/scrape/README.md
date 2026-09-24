# scrape

*Stage 1 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Requests

Fetches Project Gutenberg books and writes them to S3 exactly as fetched. It is the only
stage that touches the network, and it derives nothing — turning the HTML into text is
[standardize-html](../standardize-html/)'s job, so re-rendering never needs a refetch.

| Stage | Function | Does |
|---|---|---|
| `SUBJECT` | `scrape_subject_book_list` | Seeds every book in a subject at `LISTED` |
| `METADATA` | `scrape_book_metadata` | Fetches the bibrec; non-English books stop at `SCRAPED_SKIPPED_NON_ENGLISH` |
| `CONTENT` | `scrape_book_content` | Fetches the book HTML, raw, license boilerplate and all |

Every stage is idempotent, so a re-run skips work already done.

| S3 artifact | Contents |
|---|---|
| `metadata/{index}.json` | Title, author, publication metadata |
| `html/{index}.html` | Raw HTML, exactly as fetched |

## Invoking

One stage per call:

```json
{ "index": "gutenberg-3300", "stage": "METADATA" }
{ "index": "gutenberg-3300", "stage": "CONTENT" }
{ "subject": "12345",        "stage": "SUBJECT" }
```

An unknown stage, or one missing its argument, is rejected. The per-book stages reply with
the book's resulting status:

```json
{ "stage": "METADATA", "book_id": "gutenberg-3300", "status": "SCRAPED_METADATA" }
```

The subject machine (`${ENV_PREFIX}-scrape`) runs `SUBJECT` once, then `METADATA` and
`CONTENT` per book in a `Map` at `MaxConcurrency: 1`, with a 3s pace before every request —
gutenberg.org is a single volunteer-run host. A failed book `Catch`es to a `Succeed`, keeps
its status for the next run, and doesn't fail the subject; `SCRAPED_SKIPPED_*` books are
filtered out of the `Map`'s output (by substring, since statuses carry rank prefixes). See
[infra § Orchestration](../../infra/README.md#orchestration).

## Seeding

The per-book stages raise on a book with no pipeline entry, so `SUBJECT` must run first:

```bash
aws lambda invoke --function-name $ENV_PREFIX-scrape \
    --payload '{"stage":"SUBJECT","subject":"12345"}' out.json
```

```json
{ "subject": "12345", "found": 100, "created": 87, "indexes": ["gutenberg-3300", "..."] }
```

- `found` — books listed, capped at `MAX_BOOKS_PER_SUBJECT` (100, the most downloaded).
- `created` — newly seeded entries.
- `indexes` — every book in the set, not just new ones, so a re-run resumes a partial subject.

A subject already holding 100 books is served from the pipeline table instead of re-listed:
`created` is `0` and `indexes` covers every earlier run. Re-walking a download-ranked
listing could drop a book that slipped past rank 100 and strand it at `SCRAPED_HTML`.

The listing takes ~1s per 25-book page, well inside the 120s timeout; a cap raised past
~800 books should be seeded with the CLI.

## Running it

```bash
python scrape.py SUBJECT --subject 12345   # seed pipeline entries from a subject
python scrape.py METADATA                  # LISTED -> SCRAPED_METADATA
python scrape.py CONTENT                   # SCRAPED_METADATA -> SCRAPED_HTML
```

`METADATA` and `CONTENT` sweep every book at their input status; one failure doesn't stop
the rest.

The `lambda-scrape` compose image's entrypoint is the Lambda runtime, so override it to
reach the CLI. `src/` and `shared/` are bind-mounted, so edits need no rebuild:

```bash
docker compose run --rm --entrypoint python lambda-scrape scrape.py SUBJECT --subject 12345
```

The same image serves the runtime interface emulator on port 9010:

```bash
docker compose up -d lambda-scrape
curl -X POST http://localhost:9010/2015-03-31/functions/function/invocations \
    -d '{"stage":"SUBJECT","subject":"12345"}'
```

Tests run inside the image ([infra § Deploying](../../infra/README.md#deploying)):

```bash
docker build -f functions/scrape/Dockerfile --target test -t scrape-test . && docker run --rm scrape-test
```
