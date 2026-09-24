# standardize-html

*Stage 2 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Anthropic (Claude Sonnet 5)

Classifies every heading in a subject's books into a semantic block and rewrites each
book as `h2`/`h3`/`p` prose. The Anthropic Batch API is asynchronous, so the work is two
invocations, selected by the payload: `book_ids` or `subject_id` runs `SEND`, which opens
a batch and returns, and `batch_id` runs `RETRIEVE`, which settles it once it has ended.

## `SEND` (`book_ids` or `subject_id`)

Submits the given books that are at `SCRAPED_HTML` or `STANDARDIZE_UNRESOLVED` as one
batch; the rest are passed over.

1. Loads each book's raw `html/{index}.html` and reduces it to `(tag, text)` prose
   blocks, dropping the Project Gutenberg license wrapper and printed page numbers
2. Builds one detail line per heading: position, original tag, truncated excerpt, and
   the word count before the next heading
3. Marks books with no headings `SCRAPED_SKIPPED_NO_HEADINGS`
4. Submits every remaining book as a single batch
5. Writes a manifest, then moves each book to `STANDARDIZE_SUBMITTED`

The manifest is written first so that no book is ever `STANDARDIZE_SUBMITTED` without
one to render from.

### Choosing the input

| | `book_ids` | `subject_id` |
|---|---|---|
| Used by | The scrape machine's event, which passes the books it just took to `SCRAPED_HTML` | Re-running a subject by hand |
| Lookup | One `BatchGetItem` | A table Scan — `subject_ids` is a set, so no index can key on it |
| Empty | Rejected; the scrape machine's `books-to-standardize?` gate never sends one | Returns `batch_id: null`, `batch_status: ended` |
| Book in flight | The whole call is refused (below) | Filtered out; the rest of the subject submits |
| Size | Set by the caller | Capped at `MAX_BOOKS_PER_SUBJECT`, id-sorted; the rest keep `SCRAPED_HTML` for the next run |

Prefer starting the machine with `{ "subject_id": ... }` over invoking the Lambda, so
the poll loop collects the batch
([Re-running a subject](../../infra/README.md#re-running-a-subject)).

### Books already in flight

If any named book is at `STANDARDIZE_SUBMITTED`, `SEND` raises `BooksInFlightError` and
submits nothing, so no book is paid for twice. This also absorbs EventBridge's
at-least-once delivery. The machine catches it to `standardize-blocked`, a `Succeed`,
since nothing was lost. It catches **only** that error: any other raise may come after
the batch was opened, so it ends on `standardize-submit-failed`, where it is visible.
Batches for different subjects can run at once; a run is blocked only when its books
overlap an open batch.

A book stranded at `SCRAPED_HTML` by a run that died before submitting comes back on the
next run's list, because both scrape stages report the status they find. Re-running the
subject is the recovery. The exception is a book on the `book-failed` branch: its output
has no `status`, so it waits for a later run.

### Return value

`batch_id`, `book_count` and `batch_status`, the same `batch_status` field `RETRIEVE`
returns. When nothing was submitted, `batch_status` is `ended` and `batch_id` is `null`.
The machine's `batch-settled?` is written as the negative, so an unfamiliar
`processing_status` keeps polling.

## `RETRIEVE` (`batch_id`)

1. Reads the batch's status and **returns immediately if it has not ended**
2. Streams the results, matching each reply's `custom_id` to a book in the batch index
3. Loads that book's manifest, maps the classified blocks onto its headings, and renders
   both artifacts
4. Uploads both, then advances the book to `STANDARDIZED` in one atomic update

The row records only the status. Artifact keys are derived from the book id
(`PipelineEntry.s3_standardized_html_key` and `.s3_text_key` in
[`shared/tables/pipeline_entries.py`](../../shared/tables/pipeline_entries.py)). The stage
is safe to call repeatedly: a re-run over a settled batch re-renders from the same
manifest, and the forward-only status guard makes the repeated write a no-op. Books are
held one at a time, and no HTML is parsed here.

## Artifacts

| S3 artifact | Written by | Contents |
|---|---|---|
| `standardize-html/batch-details/{batch_id}.json` | `SEND` | The batch's book ids and its `llm_batch_id` |
| `standardize-html/books/{index}.json` | `SEND` | One book's `(tag, text)` blocks |
| `html-standardized/{index}.html` | `RETRIEVE` | `h2`/`h3`/`p` only, each with its `data-block` classification, no styling |
| `text/{index}.txt` | `RETRIEVE` | Body text, one block per paragraph/heading, separated by blank lines |

- Book manifests are per book because a per-batch one would hold every book's text.
- The batch manifest is keyed on the batch id so that concurrent batches don't
  overwrite each other's index. `RETRIEVE` checks the id stored inside it.
- The blank lines in `text/` matter: [tokenize](../tokenize/) segments sentences within
  a block, so an unpunctuated heading stays off the paragraph after it.

Nothing else records the batch id. It travels from `SEND` to `RETRIEVE` inside the
standardize machine, so a failed collect is recovered by redrive
([Recovering a batch](#recovering-a-batch)).

## Text extraction

Gutenberg wraps drop caps, small caps and page numbers in `<span>`, so `get_text(" ")`
split words (`L abour`, `J. M c Creery`). `clean_element_text` separates only at
block-level boundaries. Page numbers have to be stripped in the same change: once inline
elements imply no space, `regulate this iv distribution` becomes `ivdistribution`, long
enough to pass [create-embeddings](../create-embeddings/)'s minimum token length.

## Semantic blocks

A heading level alone is lossy (`section` and `drop` both render as `h3`), so
`standardize_tag_text_pairs` returns `StandardizedBlock(tag, text, block)`, and prose
inherits the block of the heading above it. That is what makes a whole index droppable,
not just its heading.

Nothing is deleted from `html-standardized/`. Every element carries `data-block`, and
each consumer skips what it doesn't want. `text/` feeds only the embedding stage, so it
leaves out `UNTRAINABLE_BLOCKS`. The page is titled from the library record and is always
`lang="en"`, because scrape sends non-English books to `SCRAPED_SKIPPED_NON_ENGLISH`.

```html
<html lang="en">
<title>On the Principles of Political Economy, and Taxation</title>
<h2 data-block="chapter">CHAPTER I.</h2>
<h3 data-block="section">ON VALUE.</h3>
<p data-block="section">The value of a commodity…</p>
<h3 data-block="drop">INDEX.</h3>
</html>
```

- **`drop`** is for paratext: tables of contents, indexes, errata, publishers'
  catalogues and the title page. An index matters most. It is the book's vocabulary in
  alphabetical order, so a `WINDOW = 10` co-occurrence count over it would pair `banks`
  with `agriculture`.
- **The author's own back matter stays `section`**, including appendices, conclusions,
  epilogues and footnotes (Smith's `APPENDIX TO BOOK IV`, the `Footnotes` of
  gutenberg-30107).
- **The title comes only from `metadata/{index}.json`**, which scrape uploads before the
  book can reach `SCRAPED_HTML`. `SEND` puts the title and author into the prompt only so
  the model can find the title page and drop it.

**The prompt's rules interact.** The title-page rule alone scored 22%: it made the model
swallow `CHAPTER I. / ON VALUE. / CHAPTER II. / …` as a contents listing. With the
precedence rule, the four rules together scored 100%. That was measured before `title`
and `imprint` were removed, and the evaluation set is no longer in the repo. Any
replacement needs n≥3, because Sonnet 5 has no `temperature` and one prompt has scored
both 100% and 21%.

## When a reply fails

`RETRIEVE` isolates failures per book. If rendering throws, `standardize_from_batch`
puts the book in the returned `failed` list, and a later collect renders it again. An
errored, cancelled or truncated result is logged and skipped by
`yield_anthropic_content`. Such a book appears only in the log, and it moves to
`STANDARDIZE_UNRESOLVED`.

That status exists because of the forward-only guard. Left at `STANDARDIZE_SUBMITTED`,
the book would be skipped again on every collect and refused by every `SEND`, and it
could never move back to `SCRAPED_HTML`. `STANDARDIZE_UNRESOLVED` ranks after
`STANDARDIZE_SUBMITTED`, so the guard allows the move and the book leaves flight. It is
resubmitted only when an operator names it in `book_ids`. The automated paths select
`SCRAPED_HTML` only, so a book that always fails is not paid for on every run.

Raising instead would end the results stream at the same item on every re-run, so the
batch would never settle. Truncated replies are not salvaged, because truncation falls
at the tail, where the index is, and an index that defaults to `section` stays in
`text/`. Truncation is possible because `max_output_tokens` (`heading_count * 12 + 100`)
is capped at `MAX_OUTPUT_TOKENS`, so the margin shrinks past about 1,300 headings.

## Re-classifying a book

There is no replay path. A book whose extraction or vocabulary changes goes back through
`SEND` ($0.006 per book). The obstacle is the status guard, not the cost: `SEND` takes
books at `SCRAPED_HTML`, and a classified book is at `STANDARDIZED`. Replies are kept under `batch-results/` as an audit trail only,
for telling a bad render from a bad classification.

## Layout

`src/constants.py` belongs to neither stage. It holds what both must agree on: the
semantic blocks, the heading level each renders as, and which tags count as headings.
`SYSTEM_PROMPT` lists `SEMANTIC_BLOCK_TO_LEVEL`'s keys, so a new block reaches the
prompt and the renderer in one edit, and a block the model invents is rejected.

## Recovering a batch

**Nothing watches the standardize machine.** A failed collect strands books at
`STANDARDIZE_SUBMITTED`, where `get_entries` refuses any later run that names them. An
`ExecutionsFailed` alarm on `${ENV_PREFIX}-standardize-html` is the minimum cover and
doesn't exist yet, and the recovery below has a deadline.

The batch is already paid for. Redrive restarts the execution from the failed state with
the `batch_id` it recorded:

```bash
aws stepfunctions redrive-execution \
    --execution-arn "arn:aws:states:${AWS_REGION}:${AWS_ACCOUNT_ID}:execution:${ENV_PREFIX}-standardize-html:<name>"
```

Three things in the machine keep that working:

- **`standardize-collect` has no `Catch`.** Redrive re-enters a `Fail` state and fails
  again, so the execution must fail *at the task*.
- **`standardize-submit` keeps its `Catch`**, so the submit path can't be redriven.
  Re-running it would pay for a second batch.
- **The machine has one entry point**, `{ book_ids }`.

Redrive is available for **14 days** after the execution ends, within a 24,999-event
history (a 36h poll at 300s is a few thousand). After that, or for a batch orphaned by
`standardize-submit-failed`, invoke the Lambda with `{"batch_id": "..."}`. The id can be
found with `client.messages.batches.list()` for 29 days after the batch was created.

## Running it

There is no CLI. Invoke the function, or for `SEND`, start the machine so it also
collects the batch.

```bash
aws lambda invoke --function-name $ENV_PREFIX-standardize-html \
    --payload '{"book_ids":["gutenberg-3300"]}' out.json
# {"batch_id": "msgbatch_...", "book_count": 42, "batch_status": "in_progress"}

aws lambda invoke --function-name $ENV_PREFIX-standardize-html \
    --payload '{"batch_id":"msgbatch_..."}' out.json
# {"batch_id": "...", "batch_status": "ended", "standardized": 41, "failed": []}

# SEND over the subject's pending books. It opens a batch -- there is no dry run.
aws lambda invoke --function-name $ENV_PREFIX-standardize-html \
    --payload '{"subject_id":"12345"}' out.json
```

The `lambda-standardize-html` compose service serves the deployed image on port 9030. It
runs against real Anthropic and real S3, so be careful with `SEND`:

```bash
docker compose up -d lambda-standardize-html
curl -X POST http://localhost:9030/2015-03-31/functions/function/invocations \
    -d '{"batch_id":"msgbatch_..."}'
```

The suite runs inside the image ([infra § Deploying](../../infra/README.md#deploying)):

```bash
docker build -f functions/standardize-html/Dockerfile --target test -t standardize-html-test . && docker run --rm standardize-html-test
```
