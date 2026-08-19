# standardize-headings

*Corpus-wide job, outside the per-book state machine. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Anthropic (Claude Sonnet 5)

Sweeps every book at `SCRAPED_HTML`, submits their headings to the Anthropic Batch API
as one job, and returns. It does not wait for that job — [standardize-collect](../standardize-collect/)
settles it.

1. Loads each book's raw `html/{index}.html` and reduces it to `(tag, text)` prose
   blocks, dropping the Project Gutenberg license wrapper
2. Builds one heading detail line per heading: position, original tag, truncated
   excerpt, and the word count before the next heading
3. Marks books with no headings at all `SCRAPED_SKIPPED_NO_HEADINGS`
4. Submits every remaining book as a single batch
5. Writes a manifest, then moves each book to `STANDARDIZE_SUBMITTED`

The manifest is written **before** the status changes: a book marked
`STANDARDIZE_SUBMITTED` with no manifest to render from would be stuck out of reach
of both functions.

`STANDARDIZE_SUBMITTED` is also what keeps a second run from resubmitting — and
paying for — a corpus that is already in flight. Finding any book at that status
stops the run outright, before the sweep: the manifest below sits at one fixed key,
so opening a second batch over an open one would overwrite the index the first batch
still needs in order to be collected.

| S3 artifact | Contents |
|---|---|
| `standardize-batches/index.json` | `custom_id` → book index, plus the `batch_id` they belong to |
| `standardize-batches/books/{index}.json` | One book's `(tag, text)` blocks |

One object per book rather than one per batch: the extracted text is most of a book,
so a whole-corpus manifest would be a single object the size of the corpus. Neither key
carries the batch id — the batch is named inside `index.json`, and collect checks it
against the id it was invoked with. Nothing else records it: the batch id reaches
collect through this function's return value, and is otherwise recoverable from
`client.messages.batches.list()` for 29 days after the batch was created.

This function also owns both halves of the wire format, not just the outbound one.
`llm_classify_request/` builds the prompt and sends the batch; `llm_parse_response/`
defines the semantic blocks a reply may name and validates the lines that come back.
collect imports the second package to read its batch. Keeping them side by side is
the point: a block added to the SYSTEM_PROMPT and not to `SEMANTIC_BLOCKS` is
rejected in the same package it was introduced in, rather than drifting out of step
with a validator in another function.

```bash
aws lambda invoke --function-name $LAMBDA_PREFIX-standardize-headings \
    --payload '{}' out.json
# {"batch_id": "msgbatch_...", "book_count": 42}
```
