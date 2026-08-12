# api

*Stage 6 of 6. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** FastAPI, Mangum, Pydantic, fastapi-cache, Redis, OpenAI

Mangum runs FastAPI inside a Lambda Function URL. Redis caching is optional — the
API runs without it when `REDIS_URL` is unset. Cached responses have no expiry,
so flush Redis after reprocessing a book.

Local and production run different concurrency models. `docker compose up
lambda-api` serves through `uvicorn --reload` — a single worker, reloading on
file change. The deployed Lambda has no application server at all: the runtime
invokes `main.handler` directly, one request per invocation, with no event loop
shared across concurrent requests the way `--reload` implies. A slowdown under
concurrent load in local dev is an artifact of that single worker, not a signal
about production behavior.

## What the score measures

Within-book GPA aligns the *seeds* of one book. It does **not** put two different
books in a shared frame, and the API never assumes it does.

### Why the comparison is rotation-invariant by construction

Procrustes recovers a shared frame when two point sets genuinely share geometry —
the case for seeds of the same book, not for two books trained on different
texts. At 200 dimensions the per-term cross-book rotation is underdetermined, so
a cosine between two books' vectors would not be a quantity the system could
stand behind. The query-time path avoids cross-book vectors entirely.

### Mean local similarity

Four names sit on top of one another here, and keeping them apart is what makes
the rest of this document readable. Each layer is the one below it, reduced:

| Layer | Name | Shape |
|---|---|---|
| The query's cosine to every term in one book | similarity vectors | `(n_seeds, n_terms)` |
| Two books compared over the query's 75-term neighbourhood | `local_similarity` | `(n_seeds,)`, per pair |
| Its mean, across seeds and across peers | `mean_local_similarity` | scalar |
| What that scalar is taken to mean | `DefinitionalAgreement` / `…ToCorpus` | model |

**Local** is doing work: the comparison runs over the 75 terms nearest the query
*in the measuring book*, not over the whole shared vocabulary. **Mean** is the
reduction, not part of the scope — it is the mean *of* the local similarity.
And the model is an interpretation of the value rather than a rename of it:
`mean_local_similarity` is what was measured, definitional agreement is what it
is read as.

Implemented as `BookSimilarityVectors.get_local_similarity`. Each book
describes the query by its **second-order embedding** — the query's cosine to
every term that book uses, computed inside that book's own frame, where
alignment *is* valid. Comparing two books means comparing two such profiles:

1. Take the terms both books share, dropping the expression's own leaf terms — a
   term is trivially its own nearest term, and leaving them in would inflate
   every pair identically.
2. Keep the `NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY` (75) scoring highest against the query **in
   the measuring book** — the query's *local neighbourhood*. Below 75 shared
   terms the pair raises `NoLocalNearestTermsError` rather than reporting a thin
   comparison.
3. Read both books' similarity-to-query over those 75 anchors, center each,
   L2-normalize each, and take the dot product — per seed.

The centering is what makes step 3 informative. The 75 anchors are by
construction the terms closest to the query in the measuring book, so their
cosines are uniformly high and sit within a narrow band of one another. Two raw
profiles would agree mostly on that shared height, and every pair of books would
score near the maximum whatever the authors had done. Removing each profile's own
level leaves only the deviations, so what the dot product compares is the *shape*
of the neighbourhood: whether the two books place the same terms nearer and
further within the query's vicinity, not whether the vicinity as a whole sits at
the same absolute distance. Absolute distances are frame-dependent; the shape is
not.

| Score | Reading |
|---|---|
| 1.0 | The expression sits in the same relational position in both books |
| ~0 | The two neighbourhood profiles are unrelated |
| Negative | One book inverts the other's ordering — a real result, not an error |

Negative scores are why the chart's y-axis is bounded to `[-1.05, 1.05]` and not
clamped at zero.

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

**The tradeoff.** Memory. Holding every requested book's matrices at once is why
the API Lambda is provisioned at 1024 MB rather than the 256 MB default, and it
is the reason corpus growth is bounded by the per-request working set rather
than by storage.

---

### A cache key that's stable across containers

`BooksMetadataCache.__repr__` is overridden to drop the default
`<...at 0x7f...>` address, because fastapi-cache's default key builder folds a
dependency's `repr()` into the `/books` and `/terms` cache key. Left alone, every
warm container would mint its own cache key for what is otherwise an identical
response.

---

### Similarity scoring is batched per request, not cached across requests

**The situation.** A `/semantic-drift` request scores up to eleven
expressions — the query plus its five to ten comparative terms — against
every requested book. Scored naively, that is one matmul per (book, expression)
pair, and each pair's shared anchor terms are found by intersecting two term
lists from scratch, repeated for every expression even though the same two
books recur across all of them.

**The decision.** `BooksSimilarityCache.warm_cache` stacks every expression's
query vectors for a book into one matmul — `(n_seeds, n_terms, dim) @ (n_seeds,
dim, n_exprs)` — instead of calling it once per expression. `get_shared_term_indexes`
finds shared terms via a sorted-array merge join (`np.searchsorted`) rather
than a set intersection, and caches the result per book pair, since the same
pair is shared by every expression in the request. Measured ~4x reduction in
per-request compute.

Batching only works because every expression scored against a book produces
the same shape, so an expression's own leaf terms are dropped by masking the
similarity vector *after* the matmul rather than shrinking the term matrix
before it — trimming the input first would give each expression a differently
shaped matrix and break the stack.

**Why the cache itself is rebuilt every request, unlike the term cache.** The
term-vector cache ([A request costs the slowest book, not the
sum](#a-request-costs-the-slowest-book-not-the-sum)) is process-wide because a
book's vocabulary is the same for every request that touches it. `BooksSimilarityCache` is keyed by
*this request's own query expressions* — there is nothing in it worth keeping
once the response is sent, so it is built fresh and discarded.

**The tradeoff.** Batching pays off because these particular expressions share
one book's term matrix; a request that scored many books against a single
expression wouldn't benefit the same way. Floats also drift by about 1e-7
between the batched and unbatched paths, which is why the tests compare with a
numeric tolerance instead of exact equality.

## Confidence intervals

Computed per query, not during alignment. Both intervals are
`mean ± t_crit(df) · (std / √n)`; what differs is **what gets averaged**, and the
two are not comparable in width because they estimate different things.

**Against a nominated book** (`DefinitionalAgreement`) the unit is the seed.
There is one peer, so there is no between-book variation to estimate:

1. Compute the local similarity in each aligned seed model independently
2. Take mean and standard deviation across those seeds
3. `df = n_seeds - 1`

**Against the corpus** (`DefinitionalAgreementToCorpus`) the unit is the peer.
The claim is about books, not seeds, and the spread across peers already
contains the seed noise inside each one:

1. Reduce each peer to its own mean across seeds
2. Take the standard deviation across those per-peer means
3. `df = n_books - 1`

With a single peer the corpus case falls back to the seed formula, since one
book gives nothing to estimate spread from.

Neither `std` is returned; both exist only to size the half-width. The **pinned**
interval remains a lower bound on true uncertainty: it varies the training seed
only, not the choice of source documents, and Antoniak & Mimno (TACL 2018) find
bootstrapping over documents gives a substantially wider one. The **corpus**
interval moves toward that by varying the peer, though it still samples only the
books the caller asked for rather than the population they are drawn from.

Expect corpus intervals to be wide at small `n_books` — three peers means
`df = 2` and `t_crit = 4.303`. That is the estimator being honest about how
little three books say, not a defect.

Which shape comes back follows from the request. With a `source_book_id`, the
score is against that one book, and there is no `n_books` — it could only ever
say `1`:

```json
{
  "book_id": 3300,
  "mean_local_similarity": 0.354,
  "ci": [0.312, 0.396],
  "occurrences": 1284,
  "n_seeds": 5
}
```

Without one, it is the mean of the pairwise local similarities against each peer
in turn — **not** a comparison against a single aggregate corpus profile, which
would be a different quantity — and `n_books` reports how many peers backed it:

```json
{
  "book_id": 3300,
  "mean_local_similarity": 0.21,
  "ci": [0.18, 0.24],
  "occurrences": 1284,
  "n_seeds": 4,
  "n_books": 4
}
```

`occurrences` is how often the query's terms appear in that book, summed across
a compound expression's leaf terms, so `labour + (productive - unproductive)`
reports the total for all three. `n_seeds` is how many aligned models the
comparison had in common — the minimum across peers, so one thinly-trained book
lowers it for the whole row.

Reading a `ci` depends on which one it is. Against a nominated book, tight means
the relationship was stable across training runs and wide means it was sensitive
to model randomness. Against the corpus, tight means the peers placed the term
alike and wide means they disagreed — a claim about the books, not the training.

**Why this matters for vector expressions.** Contrast directions like
`productive - unproductive` produce small raw difference vectors when the two
terms are semantically close, and per-operation normalization in the evaluator
amplifies whatever signal or noise remains. If the amplified direction varies
across seeds, the seed-based interval widens — making the `ci` on
`DefinitionalAgreement` a direct quality signal for contrast queries. The corpus
interval will not show this as reliably, since it varies the peer rather than the
seed.

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

1. Fetch per-seed vectors for `productive` and `unproductive`
2. Compute the difference element-wise across seeds
3. Normalize the contrast direction to unit length
4. Add it to the per-seed `labour` vectors
5. Normalize the final query vectors
6. Take the query's cosine to every term each book uses — the second-order
   embedding
7. Compare books over their top 75 shared terms, per seed

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
adverb-only across every occurrence (`tags == {"R"}`) is excluded.

### `POST /semantic-drift`

| Field | Type | Notes |
|---|---|---|
| `tree` | object | Recursive `TermNode`/`OpNode`, max depth 5 |
| `book_ids` | int[] | 1–16 entries, unique, must not contain `source_book_id` |

With a `source_book_id` that book is **selected**: every score is read relative to
it. Without one, each book is scored against the mean of its peers. The selected
book must not appear in `book_ids` — its agreement with itself is a constant 1.0,
which carries no information and would compress the real variation into half the
range.

A selection also narrows the candidate pool: a comparative term must be in the
selected book's vocabulary as well as clearing the corpus-wide count, since every
line is drawn against that book and a word it never uses has nothing to be drawn
against. So the same query can return a different set of comparative terms
selected and unselected.

The response is grouped by term: the query's own under `expr`, one per
comparative term under `comparative_terms`, each carrying a `books` list of
scores. `books` at the top level is the roster — one row per requested book, in
request order.

Each comparative term's statistics are aggregates of its **position** in each
book's own nearest-term neighbourhood, not of its raw cosine to the query —
since a mean or spread of distances measured in frames that share no scale would
be reading each book's scale as much as the term. Every book's similarity profile
is centred on its own top-`NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING` (100) before anything is
aggregated, so a position is a signed offset from that centre, on a cosine
scale: above zero is nearer the query than that book's neighbourhood generally
is, and **not** confined to 0–1.

On that footing a term carries `stability` — its average position across the
books that carry it — and `instability`, the sample variance of those same
positions. Three counts sit beside them: `n_books_in`, the books carrying the
word at all, and `n_books_as_top50` / `n_books_as_top100`, the subsets that
placed it inside their own top 50 or top 100 nearest terms respectively
(`n_books_as_top50 <= n_books_as_top100 <= n_books_in`, always). The counts are
the sounder pair to read absence from — a position far below a book's
neighbourhood is read against a centre taken above it, but whether the book put
the term there at all still holds either way.

`comparative_terms` is the union of two selections of up to five terms each: the
highest `stability` among terms with `n_books_as_top50 >= 2`, and the highest
`instability` among terms with `n_books_as_top100 >= 2` **and**
`n_books_as_top50 >= 1` that also rank among the 100 highest by `stability`.
Every field is returned for every term regardless of which selection
put it there — nothing about a term's statistics is conditional on how it
qualified. There was once a `sort` request field choosing one ranked list; it is
gone, both selections always run, and the union comes back in ascending
alphabetical order rather than ranked by either statistic — sort client-side on
`stability` or `instability` for a ranked view.

The centring window (100) is a single shared constant now, not one per
selection, so `stability` and `instability` mean the same thing wherever a term
appears in the response — there is no longer a caveat about comparing two
responses that used different sorts, because there is only one response shape.

One consequence is worth stating plainly: a term that holds the *same* cosine in
every book does not report `0.0`. If the neighbourhood around it moved and it did
not, its position moved, and the books genuinely disagree about where it sits.
Drift here is always relative to the company a term keeps.

### Reading absence

**Absence is expressed by omission.** There is no `unavailable` flag and there are
no null scores — a score object is either complete or not present. A book appears
on a line only if that line measured it.

**The lists are therefore not parallel and not a fixed stride.** Match a term
line's `book_id` against the top-level roster's `id` — two different key names
for the same book — never on position.

A book absent from *every* term's `books` could not be compared at all; a book
absent from *one* could not be measured for that term. The roster row says which:

- **Vocabulary gap** — the term is in that book's `missing_terms` (or, for the
  query line, any of `expr.terms` is).
- **Thin overlap** — `missing_terms` is empty and `n_shared_terms` is under 75,
  so even its best comparison could not clear the anchor floor.

`n_shared_terms` is an upper bound on the anchors, not a count of them: the floor
applies *after* the expression's leaves are dropped, so clearing 75 is no promise
that any particular comparison did.

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
| 404 | `query_in_too_few_books` | Fewer than `MIN_BOOKS_WITH_TERM` (4) requested books carry the query. Carries `book_id`, null when none selected |
| 404 | `term_resolution` | A describe term could not be matched. Carries `message`, `term`, `candidates` |
| 400 | — | LLM output could not be parsed |
| 422 | — | Repeated `book_id`, selected book among its own targets, or tree deeper than 5 |

The three 404s carry a `reason` discriminator so a client can branch without
inspecting the message.

The `query_in_too_few_books` 404 is a **vocabulary shortage only**. Books that all
carry the expression but share too few local nearest terms are a 200 whose every
term has an empty `books`— nothing was missing, there was simply nothing to
measure across.

---
