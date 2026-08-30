# standardize-html

*Subject-scoped job, outside the per-book state machine. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** BeautifulSoup, Anthropic (Claude Sonnet 5)

Classifies every heading in a subject's books into a semantic block and rewrites each
book as `h2`/`h3` prose. The work runs in two stages, selected by the field the
payload carries, because the Anthropic Batch API is asynchronous: `book_ids` runs `SEND`,
which opens a batch and returns without waiting, and `batch_id` runs `RETRIEVE`, which
settles it once it has ended.

## `SEND` (`book_ids`)

Submits the headings of the books it is handed as one batch. The stage takes a
`book_ids` work list and nothing else; it does not go looking for work of its own. Of
those books, the ones at `SCRAPED_HTML` are submitted and the rest are passed over.

The list is the caller's, because the caller already knows it: the scrape machine's
`Map` puts the books it just took to `SCRAPED_HTML` into an EventBridge event, and the
rule that starts this machine passes them straight through. Reading those rows is one
`BatchGetItem`, where finding them by subject is a table Scan — `subject_ids` is a set,
so no index can key on it. `book_ids` is required, and an empty list is rejected with it:
a caller with no work to hand over is a caller that should not be calling. The scrape
machine gates on that itself, at `books-to-standardize?`, so a subject whose every book
failed to scrape emits no event at all and nothing starts, instead of arriving here to be
refused. It never learns the outcome either way — it announces and ends.

The stage no longer finds strays on its own, but the machine hands them back to it. A
book left at `SCRAPED_HTML` by a run that died before submitting returns in the next
run's list: both scrape stages report the status they find rather than refetching, so
the book comes out of the `Map` still marked `SCRAPED_HTML` and the `book_ids` filter
keeps it. Re-running the subject is the recovery; re-scraping the book is not needed.

What does drop a book out of reach is narrower. A caught error puts it on the
`book-failed` branch, whose output has no `status` for the filter to match, so it waits
for the next run even if the table says `SCRAPED_HTML`. And a book that falls below
`MAX_BOOKS_PER_SUBJECT` in the listing stops being handed over at all.

1. Loads each book's raw `html/{index}.html` and reduces it to `(tag, text)` prose
   blocks, dropping the Project Gutenberg license wrapper and the printed page numbers
2. Builds one heading detail line per heading: position, original tag, truncated
   excerpt, and the word count before the next heading
3. Marks books with no headings at all `SCRAPED_SKIPPED_NO_HEADINGS`
4. Submits every remaining book as a single batch
5. Writes a manifest, then moves each book to `STANDARDIZE_SUBMITTED`

The manifest is written **before** the status changes: a book marked
`STANDARDIZE_SUBMITTED` with no manifest to render from would be stuck out of reach of
both stages.

`STANDARDIZE_SUBMITTED` is also what keeps a second run from resubmitting — and paying
for — a book that is already in flight. That now absorbs a second case: EventBridge
delivers at least once, so the same announcement can start this machine twice, and the
duplicate is refused here rather than paid for. The check is per book but the refusal is not:
**one book at that status raises `BooksInFlightError` and the whole call is refused.**
Nothing is submitted, no status moves, and the caller re-runs once the open batch has
been collected. The standardize machine `Catch`es this to its `standardize-blocked`
state — a `Succeed`, because nothing was lost and this is not a failure to go looking
for later — and a subject that is already mid-batch stops there instead of failing.

Two subjects can still be in flight at once — each batch settles on its own `batch_id`
and its own manifest, and a subject only blocks a run whose `book_ids` overlap it.

The return carries `batch_id`, `book_count`, and `batch_status` — the same
`batch_status` field `RETRIEVE` reports, so one caller-side branch reads both stages.
When no batch was opened, because no book was at `SCRAPED_HTML` or every one of them
turned out to have no headings, `batch_status` is `ended`: there is nothing left to wait
on. `batch_id` is `null` on exactly that path, which is what tells the two cases apart.

## `RETRIEVE` (`batch_id`)

Settles the batch that `SEND` created, if it has finished. Given a `batch_id`:

1. Reads the batch's status. **If it has not ended, returns immediately** — this stage
   never waits on a batch, which is the whole reason it is a separate invocation
2. Streams the results, matching each reply's `custom_id` to a book in the batch index
3. Loads that one book's manifest, maps the classified semantic blocks back onto its
   headings, and renders both artifacts
4. Uploads both, then advances the book to `STANDARDIZED` in a single atomic update

Step 4 writes the status and nothing else. The artifact keys are derived from the index
by `standardized_html_key` and `text_key` in
[`shared/tables/pipeline_entries.py`](../../shared/tables/pipeline_entries.py), the same
way `metadata_key` and `html_key` serve scrape and publish, so the row records that the
artifacts exist rather than where they are.

Safe to call repeatedly — a batch still running costs nothing but the call, and a re-run
over a settled batch renders the same artifacts from the same manifest rather than
skipping the books it already wrote. The repeated status write is the no-op: the guard
only lets a status move forward.

Books are held one at a time, since a whole corpus of flattened text does not fit in
memory at once. This stage never loads or parses HTML: everything needed to render comes
from the manifest `SEND` already wrote.

## Artifacts

| S3 artifact | Written by | Contents |
|---|---|---|
| `standardize-html/batch-details/{batch_id}.json` | `SEND` | The book ids one batch was opened over, plus the `llm_batch_id` it belongs to |
| `standardize-html/books/{index}.json` | `SEND` | One book's `(tag, text)` blocks |
| `html-standardized/{index}.html` | `RETRIEVE` | `h2`/`h3`/`p` only, no attributes and no styling |
| `text/{index}.txt` | `RETRIEVE` | Body text, one block per paragraph/heading, blocks separated by a blank line |

The book manifests are one object per book rather than one per batch: the extracted text
is most of a book, so a per-batch manifest would be one object the size of every book in
it. The batch-details manifest is the opposite — one small object per batch, keyed on the
batch id. It has to be: `SEND` works on the list it is handed, so several batches can be
in flight at once, and a shared key would let the second `SEND` overwrite the index the
first batch still needs in order to be collected. The batch is also named inside the manifest, which
`RETRIEVE` checks against the id it was invoked with.

Nothing else records the batch id: it reaches `RETRIEVE` through `SEND`'s return value,
carried between the two by the standardize machine, which has one entry point and takes
`book_ids` only. A collect that failed is resumed with `aws stepfunctions
redrive-execution` on that execution — Step Functions reschedules the failed poll with the
`batch_id` it had recorded, so the id never has to be carried back by hand
([Recovering a batch](../../docs/operations.md#recovering-a-batch)). Past redrive's 14-day
window, or for a batch orphaned by a submit that died after writing its manifests, the
fallback is invoking this Lambda directly with `{"batch_id": "..."}` as below; that id is
recoverable from `client.messages.batches.list()` for 29 days after the batch was
created.

The blank lines in `text/{index}.txt` are load-bearing: [tokenize](../tokenize/) segments
sentences within each block, so a heading that ends without a period stays off the front
of the paragraph following it.

## Inline elements imply no whitespace

Gutenberg wraps drop caps, small caps and printed page numbers in `<span>`. Separating
every descendant string — what `get_text(" ")` does — therefore splits words rather than
joining them: `Labour, like all other things` came out as `L abour , like all other
things`, and `J. McCreery` as `J. M c Creery`. `element_text` separates only at
block-level boundaries, so inline markup closes up and a nested list still keeps `Outer`
off `Inner`.

Page numbers have to go in the same change, not after it. They sit *between* two words of
a sentence, so once inline elements stop implying a space the number fuses onto the next
word — `regulate this iv distribution` becomes `regulate this ivdistribution`, long
enough to survive [train-kvector](../train-kvector/)'s `len(word) > 3` filter where the
separated form was harmlessly discarded. Stripping them alone, or closing the spaces
alone, each leaves the text worse than doing both.

## The semantic block outlives the heading level

The level is lossy by design — `section` and `drop` both render as `h3` — so a
level cannot tell an index from a chapter. `standardize_tag_text_pairs` returns `StandardizedBlock(tag, text, block)`
instead of a bare pair, and prose inherits the block of the heading above it. That
inheritance is what makes a *whole* index droppable rather than only its heading.

**Nothing is deleted from `html-standardized/`.** Each element carries its
classification as `data-block`, so a consumer skips what it does not want instead of
being handed a different book than the next consumer got. One artifact then serves the
trainer, a passage index, and a plain reader. `text/` is the exception, because it feeds
the trainer and nothing else: it leaves out `UNTRAINABLE_BLOCKS`.

The page also declares `lang="en"` and titles itself from the library record rather
than its index. The language is fixed rather than read off the source
because the corpus is: scrape sends any book whose metadata is not English to the
terminal `SCRAPED_SKIPPED_NON_ENGLISH` (`functions/scrape/src/scrape.py:105`), so nothing
else can reach this stage.

```html
<html lang="en">
<title>On the Principles of Political Economy, and Taxation</title>
<h2 data-block="chapter">CHAPTER I.</h2>
<h3 data-block="section">ON VALUE.</h3>
<p data-block="section">The value of a commodity…</p>
<h3 data-block="drop">INDEX.</h3>
</html>
```

**An author's own back matter is deliberately not one of them.** The classification
prompt sends appendices, conclusions and epilogues to `section`, and those are the
author's own prose — dropping them would have deleted Adam Smith's `APPENDIX TO BOOK IV`
and the whole `Footnotes` section of gutenberg-30107. The paratext that genuinely is not
the book gets its own block instead: `drop`, covering a table of contents, an index,
errata and a publisher's catalogue. A listing is split off from a preface for the same
reason — a table of contents is a list of page numbers, a preface is the author writing.

An index is the case worth naming. It is not merely noise: it is the book's own
vocabulary in alphabetical order, so a `window=10` skip-gram over it manufactures
co-occurrences between exactly the terms the corpus is queried on — `banks` beside
`agriculture` because B follows A.

The title page goes to `drop` with the rest of it — the title, byline, degrees,
translator, publisher, place, date and printer's line, however many headings they are set
across. A title page is set one line per element, so it arrives as a run of consecutive
headings with no prose between them and no structural signal, and deciding where the
title stops within that run was the most delicate judgement the prompt asked for: stop one
line late and the byline welds into the book's name.

## The library record is the only source of a book's title

`SEND` puts the title and author from `metadata/{index}.json` above the heading list, and
`render_html` titles the page from the record alone. Nothing is lost by dropping the
printed title page, because no book can reach this stage without a record:
`scrape_book_metadata` uploads it before it advances the status
(`functions/scrape/src/scrape.py:101-110`), and `RETRIEVE` only selects entries at
`SCRAPED_HTML`, two transitions further on. Classifying a title page duplicated metadata
already in hand.

The record still goes into the prompt, now only so the model can recognise where the
title page is in order to drop it.

The prompt's rules are **not independent**, which is worth knowing before editing one.
Adding the title-page rule on its own scored 22% against the evaluation set: it makes the
model readier to treat a run of headings as one unit, and without the precedence rule to
stop it, it swallows `CHAPTER I. / ON VALUE. / CHAPTER II. / …` as a contents listing.
The four rules together score 100%. That measurement predates the removal of `title` and
`imprint`, so re-run it before leaning on the numbers. See [tests/eval](tests/eval/README.md), and run it at
n≥3 — Sonnet 5 has no `temperature`, and the same prompt has returned both 100% and 21%.

## One bad reply must not strand the batch

Both halves of `RETRIEVE` isolate per book. `yield_anthropic_content` logs and skips an
errored, cancelled or truncated result instead of raising, and `standardize_from_batch`
catches anything the render throws. A skipped book is never yielded, so it keeps
`STANDARDIZE_SUBMITTED` and is named in the returned `failed` list for a later collect.

The alternative is worse than it looks. Raising ends the results iteration, so every book
after it in the stream goes uncollected — and permanently, because a re-run streams the
same results and stops at the same item. The batch could never settle, with a paid-for
batch behind it and redrive's 14-day clock running.

A truncated reply is still never *applied*: it is missing its last lines, and the ones it
did return would land on the wrong headings.

## Re-classifying a book

There is no replay path. A book whose extraction or vocabulary has changed goes back
through `SEND` for a fresh classification, because one costs $0.006 -- about sixty cents
for a hundred-book corpus -- and a second rendering path costs more than that to keep
honest. What stands in the way is the status guard, not the money: `SEND` takes books at
`SCRAPED_HTML` and a classified book is at `STANDARDIZED`.

Replies stay under `batch-results/` all the same. They are the audit trail -- the record
of what the model actually returned for a book, which is how a bad render is told apart
from a bad classification -- and nothing reads them back into the pipeline.

## Layout

This function's `src/` owns both halves of the wire format. `llm_request/` builds the
prompt and sends the batch; `llm_response/` reads the lines that come back onto a book's
headings — `fetch.py` pulls the batch results, `standardize.py` is the settle loop, and
`save_artifacts.py` writes the two artifacts. `book_records/` is what both work on:
`reduce_html.py` turns a scraped page into `(tag, text)` blocks, and `keys.py` names every
S3 object the stage writes.

`constants.py` sits above both, because it is the one thing they have to agree on: the
semantic blocks a reply may name, the heading level each one renders as, and which tags
count as headings at all. It belongs to neither stage on purpose. `SYSTEM_PROMPT` prints
`SEMANTIC_BLOCK_TO_LEVEL`'s keys as the list it offers the model rather than restating
them, so a block added to the vocabulary reaches the prompt and the renderer in the same
edit, and one the model invents anyway is rejected against the same map. The two stages
are separate invocations because of *when* they run, not because they own different code.

```bash
aws lambda invoke --function-name $LAMBDA_PREFIX-standardize-html \
    --payload '{"book_ids":["gutenberg-3300"]}' out.json
# {"batch_id": "msgbatch_...", "book_count": 42, "batch_status": "in_progress"}

aws lambda invoke --function-name $LAMBDA_PREFIX-standardize-html \
    --payload '{"batch_id":"msgbatch_..."}' out.json
# {"batch_id": "...", "batch_status": "ended", "standardized": 41}
```
