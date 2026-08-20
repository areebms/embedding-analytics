# standardize-headings

*Corpus-wide job, outside the per-book state machine. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Anthropic (Claude Sonnet 5)

Classifies every heading in the corpus into a semantic block and rewrites each book as
`h1`/`h2`/`h3` prose. The work runs in two stages, selected by the `stage` field, because
the Anthropic Batch API is asynchronous: `submit` opens a batch and returns without
waiting, `collect` settles it once it has ended.

## `stage: submit`

Sweeps every book at `SCRAPED_HTML` and submits their headings as one batch.

1. Loads each book's raw `html/{index}.html` and reduces it to `(tag, text)` prose
   blocks, dropping the Project Gutenberg license wrapper
2. Builds one heading detail line per heading: position, original tag, truncated
   excerpt, and the word count before the next heading
3. Marks books with no headings at all `SCRAPED_SKIPPED_NO_HEADINGS`
4. Submits every remaining book as a single batch
5. Writes a manifest, then moves each book to `STANDARDIZE_SUBMITTED`

The manifest is written **before** the status changes: a book marked
`STANDARDIZE_SUBMITTED` with no manifest to render from would be stuck out of reach of
both stages.

`STANDARDIZE_SUBMITTED` is also what keeps a second run from resubmitting — and paying
for — a corpus that is already in flight. Finding any book at that status stops the run
outright, before the sweep: the manifest below sits at one fixed key, so opening a second
batch over an open one would overwrite the index the first batch still needs in order to
be collected.

## `stage: collect`

Settles the batch that `submit` created, if it has finished. Given a `batch_id`:

1. Reads the batch's status. **If it has not ended, returns immediately** — this stage
   never waits on a batch, which is the whole reason it is a separate invocation
2. Streams the results, resolving each `llm_index` to a book through the batch index
3. Loads that one book's manifest, maps the classified semantic blocks back onto its
   headings, and renders both artifacts
4. Uploads both, then advances the book to `STANDARDIZED` in a single atomic update

Step 4 writes the status and nothing else. The artifact keys are derived from the index
by `standardized_html_key` and `text_key` in
[`shared/tables/pipeline_entries.py`](../../shared/tables/pipeline_entries.py), the same
way `metadata_key` and `html_key` serve scrape and publish, so the row records that the
artifacts exist rather than where they are.

Idempotent and safe to call repeatedly — a book already written has left
`STANDARDIZE_SUBMITTED`, and a batch still running costs nothing but the call.

Books are held one at a time, since a whole corpus of flattened text does not fit in
memory at once. This stage never loads or parses HTML: everything needed to render comes
from the manifest `submit` already wrote.

## Artifacts

| S3 artifact | Written by | Contents |
|---|---|---|
| `standardize-headings/batch-details/index.json` | submit | `llm_index` → book index, plus the `llm_batch_id` they belong to |
| `standardize-headings/books/{index}.json` | submit | One book's `(tag, text)` blocks |
| `html-standardized/{index}.html` | collect | `h1`/`h2`/`h3`/`p` only, no attributes and no styling |
| `text/{index}.txt` | collect | Body text, one block per paragraph/heading, blocks separated by a blank line |

One manifest object per book rather than one per batch: the extracted text is most of a
book, so a whole-corpus manifest would be a single object the size of the corpus. Neither
manifest key carries the batch id — the batch is named inside `index.json`, and collect
checks it against the id it was invoked with. Nothing else records it: the batch id
reaches collect through submit's return value, and is otherwise recoverable from
`client.messages.batches.list()` for 29 days after the batch was created.

The blank lines in `text/{index}.txt` are load-bearing: [tokenize](../tokenize/) segments
sentences within each block, so a heading that ends without a period stays off the front
of the paragraph following it.

## Layout

This function's `src/` owns both halves of the wire format. `llm_classify_request/`
builds the prompt and sends the batch; `llm_parse_response/` defines the semantic blocks
a reply may name, the heading level each one renders as, and reads the lines that come
back onto a book's headings — `fetch.py` pulls the batch results,
`standardize.py` is the settle loop, and `save_artifacts.py` writes the two artifacts.

Keeping them side by side is the point: a block added to the `SYSTEM_PROMPT` and not to
`SEMANTIC_BLOCK_TO_LEVEL` is rejected in the same package it was introduced in, rather
than drifting out of step with a validator somewhere else. The two stages are separate
invocations because of *when* they run, not because they own different code.

```bash
aws lambda invoke --function-name $LAMBDA_PREFIX-standardize-headings \
    --payload '{"stage":"submit"}' out.json
# {"batch_id": "msgbatch_...", "book_count": 42}

aws lambda invoke --function-name $LAMBDA_PREFIX-standardize-headings \
    --payload '{"stage":"collect","batch_id":"msgbatch_..."}' out.json
# {"batch_id": "...", "batch_status": "ended", "standardized": 41}
```
