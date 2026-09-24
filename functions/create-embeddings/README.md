# create-embeddings

*Stage 4 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** NumPy, SciPy (`scipy.sparse`, `svds`)

Turns one tokenized book into one vector per term: count how often terms occur near each
other, weight those counts with PPMI, and take a truncated SVD of the result. Takes books
at `TOKENIZED` and leaves each at `EMBEDDINGS_CREATED`, with a single `.npz` in S3 in
between.

There is no training loop and no seed. The embedding is a deterministic function of the
book's lemma CSV and `src/constants.py` — the same input gives the same vectors, on any
machine, which is what lets a re-run be a no-op rather than a new set of numbers.

## What it skips

Payload and reply are the [standard batch contract](../../infra/README.md#how-a-stage-is-invoked);
`${ENV_PREFIX}-create-embeddings-trigger` supplies the `book_ids` form, carrying the ids
`tokenize` just announced.

Books that reached `EMBEDDINGS_CREATED` are announced as `Books Embedded` before the
reply is returned, which is what invokes `publish`. The announcement is emitted only if
at least one book made it, and a bus that rejects it raises — a run whose work nothing
downstream hears about is a failure, not a quiet success.

Three things take a book out of the run, all of them logged, none of them ending it:

| Skipped when | Because | Left at |
|---|---|---|
| the book has no pipeline entry, or its status is not `TOKENIZED` | this stage never creates a row; `scrape` seeds them. A book already at `EMBEDDINGS_CREATED` is the idempotent path, and it costs one status read | unchanged |
| `token_lemmas/{index}.csv` is missing | the row says tokenized and the artifact disagrees. Only `NoSuchKey` is read this way; any other `ClientError` is re-raised, so a permissions problem does not present as an untokenized book | `TOKENIZED` |
| the vocabulary is not larger than `VECTOR_SIZE` | `svds` needs `k` below the matrix dimension, so a book too small for 100 dimensions is not embedded in fewer | `EMBEDDINGS_CREATION_FAILED` |

Only the last of those moves a status, and it moves it to a terminal one: a book too
small for 100 dimensions will be too small on every re-run, so it is marked rather than
retried forever.

The status write is conditional and forward-only (`build_status_guard`), so a book that
is already `EMBEDDINGS_CREATED` turns the write down. That returns `False`, which is
logged as a warning and not raised — a re-run that finds its own earlier output is
normal, not a failure. It is also what makes a forced re-embedding awkward. Forcing a
re-embed means writing the earlier status by hand, outside `set_status`.

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
| `embeddings/{index}.npz` | Three arrays: `terms`, `vectors` — a `float32` matrix of one `VECTOR_SIZE` row per term in the same order — and `attr_count`, whole-book occurrences after the step 1 filter |

## Hyperparameters

`src/constants.py`, read at import. `MAX_BOOKS_PER_SUBJECT` shares the file and is not
one of these — it bounds a subject re-run, not the embedding:

| | | |
|---|---:|---|
| `WINDOW` | 10 | context terms on each side, never across a passage |
| `MIN_COUNT` | 10 | vocabulary threshold, counted over the whole book |
| `MIN_TOKEN_SIZE` | 3 | shorter tokens are dropped before counting |
| `VECTOR_SIZE` | 100 | SVD dimensionality, and the floor a book's vocabulary must clear |
| `GAMMA` | 0.5 | eigenvalue weighting, `w = U * S**gamma` |
| `ALPHA` | 0.0 | PMI shift subtracted before clipping |
| `CDS` | 0.75 | context distribution smoothing exponent |
| `SOLVER` | `arpack` | `propack` is roughly twice as fast and less accurate |
| `V0_SEED` | 0 | Lanczos start vector |

Changing any of the hyperparameters changes every book's vectors, so a change is a re-run
of the whole corpus through this stage and `publish`, not of one book.

## Running it

`create_embeddings.py` runs standalone and sweeps every book the status index reports at
`TOKENIZED`, one at a time — which needs the `status-index` GSI to exist on the table it
is pointed at:

```bash
docker compose run --rm lambda-create-embeddings python create_embeddings.py
```

The `local` target clears the Lambda entrypoint, so no override is needed, and `src/` and
`shared/` are bind-mounted, so edits apply without a rebuild.

Its suite runs inside the image — 43 tests at 100% statement coverage, against moto
rather than AWS ([infra § Deploying](../../infra/README.md#deploying)):

```bash
docker build -f functions/create-embeddings/Dockerfile --target test -t create-embeddings-test . && docker run --rm create-embeddings-test
```

Unlike `scrape` and `standardize-html`, this Dockerfile does its COPYs in `base` and the
`lambda` stage adds only the handler CMD, so the `test` stage still runs against the same
files that ship.
