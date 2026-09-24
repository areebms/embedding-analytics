# api

*Stage 6 of 6. [Pipeline overview](../../docs/pipeline.md) · [Project README](../../README.md)*
**Libraries:** FastAPI, Mangum, Pydantic, fastapi-cache, Redis, OpenAI

Mangum runs FastAPI inside Lambda, behind an API Gateway REST API at
`api.embedding-analytics.com`. Redis caching is optional — the
API runs without it when `REDIS_URL` is unset. Cached responses have no expiry,
so flush Redis after reprocessing a book.

Local and production run different concurrency models. `docker compose up
lambda-api` serves through `uvicorn --reload` — a single worker, reloading on
file change. The deployed Lambda has no application server at all: the runtime
invokes `main.handler` directly, one request per invocation, with no event loop
shared across concurrent requests the way `--reload` implies. A slowdown under
concurrent load in local dev is an artifact of that single worker, not a signal
about production behavior.

## What it reads

One DynamoDB row per (book, term), written by [publish](../publish/README.md): the
vector, the whole-book count, and the POS tags. `BookTermVectors` loads a whole book at
once, sorts the rows by term, and L2-normalizes the matrix — the sort is load-bearing,
because cross-book term pairing is elementwise and unsorted rows would silently pair the
wrong terms.

The column is `vector`, one per term.

The pipeline row is read the same way `publish` writes it: `book_id`, `status`, and a
nested `metadata` map. `/books` answers from the books carrying that map, so it lists
what has been published rather than what has been embedded, and a book with no
`published_year` is left out of the dropdown it feeds.

**What is left is the deploy — the image and the environment together.** The deployed
function still runs the release-4.0 image and still names `PipelineStatus` and `BookTerms`
— the pre-migration tables — so the live site serves the old code over the old corpus.
Neither half works without the other: this code reads `vector`, the old tables carry
`vectors`.

It is CDK-owned now, and its `services.yaml` entry lists every variable it reads — CDK
declares the whole environment map rather than merging into the deployed one.

## What the score measures

Each book is embedded on its own, in its own frame. Nothing puts two books into a
shared frame, and the API never assumes anything does.

### Why the comparison is rotation-invariant by construction

An SVD frame is arbitrary up to rotation and sign: the axes of one book's
`VECTOR_SIZE` (100) dimensions have no counterpart in another's, and the two were
solved over different vocabularies from different texts. A cosine taken between two
books' vectors would therefore not be a quantity the system could stand behind —
cross-book alignment was tried and abandoned on measurement. The query-time path
avoids cross-book vectors entirely: every cosine is taken inside one book, and what
crosses between books is only a per-book baseline subtracted from it.

### Adjusted cosine

Five names sit on top of one another here. Each layer is built from the one above
it:

| Layer | Name | Shape |
|---|---|---|
| The query's cosine to every term in one book | similarity vectors | `(n_terms,)` |
| The mean of the query's top 75 of those | baseline, `local_mean_similarity`, `r_b` | scalar, per book |
| The mean of `r_b` across the books on the line | average baseline, `cross_book_mean_similarity`, `R` | scalar, per request |
| A term's cosine, less its book's `r_b`, plus `R` | adjusted similarity, `similarity` | field |
| The model carrying it | `BookSimilarity` | model |

```text
similarity_b(t) = cos_b(q, t) − r_b(q) + R
r_b(q)          = mean of the query's NUM_LOCAL_NEAREST_TERMS (75) highest cosines in book b
R               = mean of r_b(q) over the books on the query line
```

The query's own leaf terms are dropped before anything is ranked — a term is
trivially its own nearest term.

**The query line is `R`, flat.** Every book on it reports the same value, so it
carries no per-book information; it is the level a term line is read against. A term
drawn above the query line in some book is closer to the query there than that book's
average top-75 term, and one below it is further.

**Why `r_b` is subtracted.** Each book's model runs at its own cosine level, so a raw
cosine in one book is not on the same scale as a raw cosine in another. `r_b` measures
that level among the query's nearest terms, where the comparative terms live, and
subtracting it turns each value into a contrast inside one model, where the
model's scale cancels. Before the subtraction, `r_b` tracked book vocabulary size at
+0.78 to +0.86 and the term lines moved with it; after it, the term lines' median
correlation with book size is −0.14, a slight overcorrection (live corpus, 2026-09-13). The formula is the
query half of CSLS (Conneau et al., ICLR 2018).

**Why `R` is added back.** Only so the axis still reads as a cosine. It is one constant
for the whole response, so it changes no comparison within it — but it does move with
the roster, so levels are not comparable between two responses over different books;
gaps are. Nothing is comparable between two queries: `R` and every value under it
measure closeness to one query.

**What the baseline cannot do.** It cannot tell model scale from real tightness: a book
that uses the query in a narrower sense really does place its nearest terms closer,
`r_b` rises, and that difference is subtracted as though it were scale. Only the query
side is corrected — a term close to everything in one book (a hub) reads as a strong
associate there.

Implemented in `BookSimilarityVectors` and `get_related_terms`. A book is on the
lines only if it carries every leaf of the query and shares at least 75 non-leaf terms
with another requested book — with a selection, with the selected book. A term's point
is dropped from a book that lacks the term.

## Request-path performance

Three choices in the request path that are not obvious from the code, kept here
because each one was corrected by measurement after it was first written.

### A request costs the slowest book, not the sum

**The situation.** A drift request touches every book in the corpus, and each
book's term matrix is a separate DynamoDB read. Serially, a cold request costs
the sum of all of them.

**The decision.** Load them concurrently through a `ThreadPoolExecutor`, so the
request costs the slowest single book instead. The work is ~99% DynamoDB
round-trip, so threads are the right primitive — there is almost no GIL-bound
compute to serialize. The matrices are then cached process-wide, so warm
containers skip the load entirely.

**Caught by measurement before the "~99%" claim above went unchecked.**
Profiling `load_book` under a simulated per-book round-trip found the opposite
of that paragraph: decoding each term's stored vectors with its own
`np.frombuffer` / `np.stack` calls is ~8,000 tiny NumPy operations per book, and
with several concurrent workers each holding the GIL in slices too short to
overlap with I/O, the pool made a synthetic 16-book cold load *slower* than
loading serially — up to 8x, not faster. The fix collects every term's raw
vector buffer during the DynamoDB scan and decodes the whole book in one
`b"".join` → `frombuffer` → `reshape` → `normalize` pass instead of one per term
(bit-exact against the old per-term path). Once decode is a handful of long
NumPy calls instead of thousands of short ones, the threads stop thrashing and
go back to overlapping I/O the way the paragraph above assumes.

**Re-measured since, and the "~99%" now holds.** That finding was made when each
term stored five seed vectors. With one vector per term, profiling 24 books against
the real tables (2026-09-15, from a dev box) puts the load at 99.8% DynamoDB fetch,
and a per-term decode is within noise of the bulk one under the pool (1.10s against
1.20s). The bulk decode stays, but it no longer buys anything measurable. The pool
does: 5.84s serial against 1.20s with 8 workers. The cache does more: a warm request
is 8.7ms against 1.27s cold.

**The tradeoff.** Memory, though not the binding limit yet: 24 books peak at 131 MB.
The API Lambda is provisioned at 1024 MB rather than the 256 MB default for CPU,
which Lambda allocates in proportion to memory. Corpus growth is still bounded by
the per-request working set rather than by storage.

---

### A cache key that's stable across containers

`BooksMetadataCache.__repr__` is overridden to drop the default
`<...at 0x7f...>` address, because fastapi-cache's default key builder folds a
dependency's `repr()` into the `/books` and `/terms` cache key. Left alone, every
warm container would mint its own cache key for what is otherwise an identical
response.

---

### One matmul per book, cached only for the request

**The situation.** A `/semantic-drift` request draws the query and its five to ten
comparative terms in every requested book. When each line was its own second-order
comparison, that was one expression per line — up to eleven matmuls per book, which
`BooksSimilarityCache.warm_cache` stacked into one.

**The decision.** Every line is now read off the query's own similarity vector: a
comparative term's point in a book is one entry of the vector already computed for
the query, less that book's `r_b`, plus `R`. Each book therefore costs one
`(n_terms, dim) @ (dim,)` matmul whatever the number of comparative terms, and
`BooksSimilarityCache.warm_cache` is deleted along with the measure that needed it —
`BooksTermCache.warm_cache`, the concurrent book load above, is a different method and
still runs on every request. The query's leaf terms
are masked out of the vector after the matmul.

`get_shared_term_indexes` finds shared terms via a sorted-array merge join
(`np.searchsorted`) rather than a set intersection, and caches the result per book
pair on the process-wide term cache. It now serves only the 75-term comparability
check and `n_shared_terms`.

**Why the cache itself is rebuilt every request, unlike the term cache.** The
term-vector cache ([A request costs the slowest book, not the
sum](#a-request-costs-the-slowest-book-not-the-sum)) is process-wide because a
book's vocabulary is the same for every request that touches it. `BooksSimilarityCache` is keyed by
*this request's own query* — there is nothing in it worth keeping
once the response is sent, so it is built fresh and discarded.

## The describe pipeline

`/parse-describe` converts natural language into a validated expression tree in
four steps. The LLM proposes; deterministic code decides whether the result is
structurally valid. No LLM output reaches the evaluation layer without passing
the parser.

```text
message → LLM expression → parser → term resolution → validated expression
```

**Step 1 — LLM generation.** The message goes to `gpt-4o-mini` with a system
prompt tuned for classical economics vocabulary: rules for multi-word concepts
(join components with `+`, prefer adjective forms), contrastive phrasing (shared
concept with subtracted modifiers), and lemmatization conventions.

**Step 2 — Recursive descent parsing.** Tokenized and parsed into `TermNode` /
`OpNode`. Every binary operator takes exactly two arguments; nested operations
must be parenthesized, the outermost need not be. Max depth 5. Malformed syntax
returns a 400.

**Step 3 — Term resolution.** Every parsed term is validated against the DynamoDB
vocabulary, cached per Lambda instance via `lru_cache`, in three tiers that
escalate cost only when needed:

- *Exact match* — the term exists as written. No external call.
- *Fuzzy match* — `difflib.get_close_matches` at a 0.6 cutoff. A single close
  match is used automatically. No external call.
- *LLM fallback* — widen to a 0.3 cutoff, collect up to 20 candidates, ask
  `gpt-4o-mini` to pick the most semantically appropriate. The selection must
  exist in the vocabulary.

Unresolvable terms raise `TermResolutionError` → 404 with up to 5 candidate
suggestions, giving the frontend enough context for manual recovery.

**Step 4 — Rebuild.** Substitutions are applied, the tree is serialized back to
an expression string, and both are returned.

### Expression evaluation

The evaluator normalizes after each sub-expression, not once at the end. For
`labour + (productive - unproductive)`:

1. Fetch the vectors for `productive` and `unproductive`
2. Compute the difference element-wise
3. Normalize the contrast direction to unit length
4. Add it to the `labour` vector
5. Normalize the final query vector
6. Take the query's cosine to every term each book uses
7. Subtract each book's top-75 mean and add back the mean of those across books —
   the [adjusted cosine](#adjusted-cosine)

Per-operation normalization prevents high-frequency or high-norm terms from
dominating combined expressions, and makes a contrast a *direction* that tilts
the base term rather than a raw magnitude that might be too small to matter.

## API contract

Four endpoints. **CORS:** `localhost:5173`, `127.0.0.1:5173`, and
`PRODUCTION_DOMAIN`.

| Endpoint | Returns |
|---|---|
| `GET /books` | Every corpus that completed the pipeline and carries a publication year |
| `GET /terms` | Cross-book vocabulary — terms in ≥2 books, adverb-only excluded |
| `POST /semantic-drift[/{source_book_id}]` | The whole comparison in one round-trip |
| `POST /parse-describe` | Plain English → a validated vector expression |

### `GET /books`

Returns `list[BookResponse]`:

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `label` | string | First author surname + year, e.g. `"Smith (1776)"` — built for a dropdown |
| `author` | string | |
| `title` | string | |
| `published_year` | int | |

Only books that have completed the pipeline *and* carry a `published_year` are
returned — a trained book with no recorded year is invisible to this endpoint.

### `GET /terms`

Returns `list[TermResponse]`:

| Field | Type | Notes |
|---|---|---|
| `term` | string | |
| `books` | int[] | The ids of the books carrying the term, joinable directly against `GET /books`' `id` |

Only terms appearing in at least two books are returned, and a term tagged
modifier-only across every occurrence (`tags <= {"J", "R", "W"}`) is excluded.

### `POST /semantic-drift`

| Field | Type | Notes |
|---|---|---|
| `tree` | object | Recursive `TermNode`/`OpNode`, max depth 5 |
| `book_ids` | int[] | 20–50 entries, unique, must not contain `source_book_id` |

With a `source_book_id` that book is **selected**. It is on no line itself, and it
does not enter any value: every book is adjusted by its own baseline either way. It
decides which books are on the lines — each must share 75 non-leaf terms with the
selected book rather than with any requested peer — and it narrows the candidate
pool: a comparative term must be in the selected book's vocabulary as well as
clearing the corpus-wide count. So the same query can return a different set of
comparative terms, and a different `R`, selected and unselected.

The response is grouped by term: the query's own under `expr`, one per
comparative term under `top_mean` or `top_std`, each carrying a `book_similarities`
list of scores. `book_stats` at the top level is the roster — one row per book on
the query line, in request order. A requested book on no line has no row.

```json
{
  "book_id": 3300,
  "similarity": 0.354,
  "occurrences": 1284
}
```

On the query line `similarity` is `R` and `occurrences` is how often the query's
terms appear in that book, summed across a compound expression's leaf terms, so
`labour + (productive - unproductive)` reports the total for all three. On a
comparative term's line `occurrences` is that term's own count.

Each comparative term's statistics are aggregates of its **adjusted similarity**
`similarity_b(t)`, the value its line draws, not of its raw cosine to the query —
since a mean or spread of cosines measured in frames that share no scale would
be reading each book's scale as much as the term. The statistics are taken over
every requested book that carries the query, including any left off the lines for
thin overlap.

On that footing a term carries `similarity_mean` — its mean adjusted similarity
across the books that carry it, above `R` where it sits on average nearer the query
than a book's baseline, and **not** confined to 0–1 — and `similarity_std`, the
sample standard deviation of the same values. `R` is one constant per response, so
it moves `similarity_mean` by exactly `R` and leaves `similarity_std` untouched.
Beside them sits `n_books_in`, the books carrying the word at all, which is the
field to read absence from.

`top_mean` and `top_std` are two disjoint selections of up to `NUM_COMPARATIVE_TERMS` (6)
terms each — the *consistent* and the *contested* terms of the product README,
under the field names that carry them here. `top_mean` holds the highest
`similarity_mean`; `top_std` holds the highest `similarity_std` among the
terms left over, so a term that tops both lists is returned once, in `top_mean`,
and the next contested term takes its slot in `top_std`. Both draw from one pool of *relevant* terms, and a term
enters it by sitting above the query line — nearer the query than that book's
baseline `r_b` — in at least `ceil(BOOKS_WITH_TERM_ABOVE_EXPR * n)` (0.2) books,
where `n` is the number of requested books carrying the query. The query's own
leaf terms never enter it, because a term is trivially nearest to itself. A
selection adds a second condition, the selected book's vocabulary. Every field is
returned for every term regardless of which selection put it there — nothing
about a term's statistics is conditional on how it qualified. Both selections
always run, and each list comes back ranked by its own statistic, highest first:
`top_mean` by `similarity_mean`, `top_std` by `similarity_std`.

One consequence is worth stating plainly: a term that holds the *same* cosine in
every book does not report a `similarity_std` of `0.0`. If the query's nearest terms
moved and it did not, its adjusted similarity moved, and the books genuinely disagree about where it sits.
Drift here is always relative to the company a term keeps.

### Reading absence

**Absence is expressed by omission.** There is no `unavailable` flag and there are
no null scores — a score object is either complete or not present. A book appears
on a line only if that line measured it.

**The lists are therefore not parallel and not a fixed stride.** Match a term
line's `book_id` against the top-level roster's `id` — two different key names
for the same book — never on position.

Absence is decided at two levels:

- **Off every line** — a requested book with no roster row. It lacks a leaf of the
  query, or it shares fewer than 75 non-leaf terms with every other requested book
  (with a selection, with the selected book). Thin overlap is decided once per book,
  for all its lines together.
- **Off one term's line** — a roster book that lacks that term, which is then in its
  `missing_terms`. This is the only reason a roster book is missing from a line.

`missing_terms` therefore never lists a query leaf. `n_shared_terms` is the most terms
the book shares with any one peer carrying the query (with a selection, with the
selected book), counted before the query's leaves are dropped — an upper bound on the
75-term check, not the count it ran on.

### `POST /parse-describe`

| Field (request) | Type |
|---|---|
| `message` | string |

| Field (response) | Type | Notes |
|---|---|---|
| `expression` | string | The resolved expression, serialized |
| `terms` | string[] | Every term in the resolved tree |
| `substitutions` | object[] | `{original, resolved}` per term the resolver changed |

`substitutions` is empty when every term the LLM proposed already matched the
vocabulary exactly. See [The describe pipeline](#the-describe-pipeline) for how
a term gets substituted.

### Errors

| Status | `reason` | Raised when |
|---|---|---|
| 404 | `expression_absent` | The selected book lacks a leaf of the expression. Carries `book_id`, `terms` |
| 404 | `query_in_too_few_books` | Fewer than `int(BOOKS_WITH_EXPR * len(book_ids))` — a quarter of the requested books, rounded down — carry the query. Carries `book_id`, null when none selected |
| 404 | `term_resolution` | A describe term could not be matched. Carries `message`, `term`, `candidates` |
| 400 | — | LLM output could not be parsed |
| 422 | — | Fewer than 20 or more than 50 `book_ids`, a repeated `book_id`, selected book among its own targets, or tree deeper than 5 |

The three 404s carry a `reason` discriminator so a client can branch without
inspecting the message.

The `query_in_too_few_books` 404 is a **vocabulary shortage only**. Books that all
carry the expression but none of which shares 75 non-leaf terms with another are a
200 with empty `book_similarities` on every line and an empty `book_stats` — nothing
was missing, there was simply nothing to measure across.

**At the API** (implemented client-side, in the frontend), retry exactly once, after
2s, on a network error or a 5xx — which absorbs a cold start without adding load where
load is the problem. Every 4xx (`expression_absent`, `query_in_too_few_books`, a 422)
is a deterministic answer about the expression and is never retried.

---

## Observability

One JSON line per request, emitted by `RequestLoggingMiddleware` in `api`. Nothing else
writes application logs directly; code that wants a field on the line calls
`add_to_log(**fields)`, which mutates a per-request dict held in a `ContextVar`. The
mutation is load-bearing: FastAPI runs sync `def` handlers on a worker thread with a
copied context, so a rebind (`.set()`) would not reach the middleware, while a mutation
of the same dict does.

The line always carries `method`, `path`, `status`, `dur_ms`, `endpoint` and any path
params; handlers add fields such as `query`, `warm_ms`, `nearest_terms_ms`,
`similarities_ms`, `scored_terms`, `vocab_terms` and `error`. An unhandled exception is
caught, recorded as `status=500` with its type and message, re-raised, and only then
emitted, so a 500 always leaves a line.

---

## Running it

```bash
docker compose up lambda-api    # --> http://localhost:8000
```

The `local` target runs `uvicorn --reload` against a bind-mounted `src/` and `shared/`, so an edit
reloads without a rebuild. It answers against whatever `.env` points at: the API is a
reader, so there is no fixture corpus and an empty set of tables answers every query with
an empty vocabulary rather than an error.

Its suite runs inside the image, with `REDIS_URL` unset by the suite itself so the
cache decorator is a no-op ([infra § Deploying](../../infra/README.md#deploying)):

```bash
docker build -f functions/api/Dockerfile --target test -t api-test . && docker run --rm api-test
```
