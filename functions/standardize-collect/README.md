# standardize-collect

*Corpus-wide job, outside the per-book state machine. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** Anthropic (Claude Sonnet 5)

Settles a batch that [standardize-headings](../standardize-headings/) created, if it
has finished. Given a `batch_id`:

1. Reads the batch's status. **If it has not ended, returns immediately** — this
   function never waits on a batch, which is the whole reason it is separate from
   submit
2. Streams the results, resolving each `custom_id` to a book through the batch index
3. Loads that one book's manifest, maps the classified semantic blocks back onto its
   headings, and renders both artifacts
4. Uploads both, then advances the book to `STANDARDIZED` in a single atomic update
5. Resets every failed book to `SCRAPED`

Step 5 matters: standardize-headings only sweeps `SCRAPED`, so a book left sitting at
`STANDARDIZE_SUBMITTED` would never be retried by anything. Handing failures back
makes the pair self-healing.

Idempotent and safe to call repeatedly — a book already written has left
`STANDARDIZE_SUBMITTED`, and a batch still running costs nothing but the call.

Steps 1-4 are not this function's code. Reading the batch, resolving a `custom_id`
to a book, and validating the reply all belong to
[`llm_parse_response`](../standardize-headings/src/llm_parse_response/), which lives
beside the `llm_classify_request` package that wrote the prompt — the two halves of
one wire format, changed together. What stays here is the rendering: this function
owns `SEMANTIC_BLOCK_TO_LEVEL`, the map from a semantic block to a heading level,
and fails at import if the prompt ever offers a block it has no level for.

**No BeautifulSoup.** Everything needed to render comes from the manifest, so this
function never loads or parses HTML. Books are held one at a time, since a whole
corpus of flattened text does not fit in memory at once.

| S3 artifact | Contents |
|---|---|
| `html-standardized/{index}.html` | `h1`/`h2`/`h3`/`p` only, `data-semantic-block` on each heading, no styling |
| `text/{index}.txt` | Body text, one block per paragraph/heading, blocks separated by a blank line |

Those blank lines are load-bearing: [tokenize](../tokenize/) segments sentences within
each block, so a heading that ends without a period stays off the front of the
paragraph following it.

```bash
aws lambda invoke --function-name $LAMBDA_PREFIX-standardize-collect \
    --payload '{"batch_id":"msgbatch_..."}' out.json
# {"batch_id": "...", "batch_status": "ended", "standardized": 41}
```
