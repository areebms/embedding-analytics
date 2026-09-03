# create-embeddings

*Stage 3 of 5. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** NumPy, SciPy (`scipy.sparse`, `svds`)

Turns one tokenized book into one vector per term: count how often terms occur near each
other, weight those counts with PPMI, and take a truncated SVD of the result. Takes a book
at `TOKENIZED` and leaves it at `EMBEDDED`, with a single `.npz` in S3 in between.

There is no training loop and no seed. The embedding is a deterministic function of the
book's lemma CSV and `src/constants.py` — the same input gives the same vectors, on any
machine, which is what lets a re-run be a no-op rather than a new set of numbers.

## One book per invocation

```json
{ "index": "gutenberg-3300" }
```

`extract_index` reads `index` off the event or out of a JSON `body`, so the same payload
works from a state machine task and from an HTTP-shaped source. An event without one
raises `ValueError` rather than being read as "every tokenized book".

The reply is `{"book_id": "gutenberg-3300"}` when a book was embedded, and the same field
plus `"skipped": true` when it was not. Four things skip, all of them logged, none of them
an error:

| Skipped when | Because |
|---|---|
| the book has no pipeline entry | this stage never creates a row; `scrape` seeds them |
| its status is not `TOKENIZED` | including a book already at `EMBEDDED` — that is the idempotent path, and it costs one status read |
| `token_lemmas/{index}.csv` is missing | the row says tokenized and the artifact disagrees. Only `NoSuchKey` is read this way; any other `ClientError` is re-raised, so a permissions problem does not present as an untokenized book |
| the vocabulary is not larger than `VECTOR_SIZE` | `svds` needs `k` below the matrix dimension, so a book too small for 100 dimensions is skipped rather than embedded in fewer |

The status write is conditional and forward-only (`build_status_guard`), so a book that is
already `EMBEDDED` turns the write down. That returns `False`, which is logged and not
raised — a re-run that finds its own earlier output is normal, not a failure.

## From passages to vectors

A row of `token_lemmas/{index}.csv` is one passage, which is the unit `tokenize`
established and the unit a context window is not allowed to cross.

1. **Filter.** Tokens that are not alphabetic, or shorter than `MIN_TOKEN_SIZE`, are
   dropped — digits, punctuation and stray short forms never reach the vocabulary.
2. **Vocabulary.** Terms occurring at least `MIN_COUNT` times in the whole book are kept,
   ordered by descending count and then alphabetically. The order is total, so term ids
   do not depend on dictionary iteration order.
3. **Co-occurrence.** Each passage becomes a list of vocabulary ids; the ids are
   concatenated and paired at every offset from 1 to `WINDOW`, masked to pairs that came
   from the same passage. Adding the transpose makes the matrix symmetric, so `WINDOW` is
   ten kept terms *on each side*. Because the ids are built from kept terms only, a token
   dropped in step 1 or 2 closes the gap rather than consuming a window slot.
4. **PPMI.** Pointwise mutual information with context distribution smoothing: the context
   marginal is raised to `CDS` before normalizing, `ALPHA` is subtracted as a shift, and
   negatives are dropped. Dropping them is what keeps the matrix sparse — PPMI can only
   have as many stored cells as the counts it came from, never more.
5. **SVD.** `svds` at `k = VECTOR_SIZE` with a fixed start vector (`V0_SEED`) and the
   `SOLVER` factorization, dimensions re-sorted by descending singular value, then scaled
   by `S ** GAMMA`. The start vector is where an iterative solver would
   otherwise vary between runs, and pinning it is what makes the vectors reproducible.

A term that occurs `MIN_COUNT` times but never within a window of any *other* term has an
empty PPMI row and would ship as a zero vector, which has no direction and therefore no
cosine. Those terms are dropped after the SVD — row sum minus diagonal of zero — and the
count of dropped terms is logged.

Vectors ship unnormalized; the API L2-normalizes on load.

| S3 artifact | Contents |
|---|---|
| `embeddings/{index}.npz` | A `KVectors` archive: `terms`, a `float32` matrix of one `VECTOR_SIZE` row per term in the same order, and an `attr_count` column of whole-book occurrences after the step 1 filter |

## Hyperparameters

All of `src/constants.py`, all read at import:

| | | |
|---|---:|---|
| `WINDOW` | 10 | context terms on each side, never across a passage |
| `MIN_COUNT` | 10 | vocabulary threshold, counted over the whole book |
| `MIN_TOKEN_SIZE` | 4 | shorter tokens are dropped before counting |
| `VECTOR_SIZE` | 100 | SVD dimensionality, and the floor a book's vocabulary must clear |
| `GAMMA` | 0.5 | eigenvalue weighting, `w = U * S**gamma` |
| `ALPHA` | 0.0 | PMI shift subtracted before clipping |
| `CDS` | 0.75 | context distribution smoothing exponent |
| `SOLVER` | `arpack` | `propack` is roughly twice as fast and less accurate |
| `V0_SEED` | 0 | Lanczos start vector |

Changing any of them changes every book's vectors, so a change is a re-run of the whole
corpus through this stage and `publish`, not of one book.

## What still points at the old layout

This stage used to write a centroid model and a stack of replicates under
`embeddings/{index}/`, and three call sites have not caught up with the single flat file:

- `publish` loads `embeddings/{index}/centroid.npz` and every other `.npz` under that
  prefix as per-seed models, and reads `variance`, `disparity` and `r_squared` attributes
  off the centroid (`functions/publish/src/publish_utils.py`). None of those exist now, so
  `publish` raises on a book this stage embedded — it needs to read
  `embeddings/{index}.npz` and stop expecting an alignment-quality axis.
- `PipelineEntry.s3_prefix_models` still returns `embeddings/{index}/` and has no caller.
- [Pipeline](../../docs/internals.md) still describes the output as "centroid and
  replicate models", and [Operations](../../docs/operations.md#lambda-resources)
  justifies the 1536 MB as "PPMI/SVD over sentence-bootstrap replicates".

Nothing invokes this stage automatically either. `infra/app.py` deploys `scrape`,
`standardize-html` and `tokenize`; there is no per-book state machine and no rule turning
`tokenize`'s output into an invocation here, so a book reaches this stage only by being
named in a payload or swept up by the CLI below. `infra/services.yaml` already carries the
profile it will deploy with — 1536 MB, 600s, `S3_BUCKET` and `PIPELINE_TABLE`.

## Running it

`create_embeddings.py` runs standalone and sweeps every book the status index reports at
`TOKENIZED`, one at a time:

```bash
docker compose run --rm lambda-create-embeddings python create_embeddings.py
```

The `local` target clears the Lambda entrypoint, so no override is needed, and `src/` is
bind-mounted, so edits apply without a rebuild.

The suite is 43 tests at 100% statement coverage against an 85% floor, run against moto
rather than AWS:

```bash
docker build -f functions/create-embeddings/Dockerfile --target test -t create-embeddings-test . && docker run --rm create-embeddings-test
```

Unlike `scrape` and `standardize-html`, this Dockerfile does its COPYs in `base` and the
`lambda` stage adds only the handler CMD, so the `test` stage still runs against the same
files that ship.
