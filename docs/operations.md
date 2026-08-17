# Operations

Content that spans more than one pipeline stage, or applies to the whole
system rather than to a single one. Stage-specific mechanics live in
[the pipeline map](./internals.md).

---

## Orchestration

```text
scrape → tokenize → train-kvector Map(N seeds) → align-kvectors → publish
```

`train-kvector` runs as a Step Functions Map state, so each seed is an
independent Lambda invocation. Earlier steps carry the `seeds` array through
output transforms, keeping orchestration logic out of the handlers.

```json
{ "index": "gutenberg-3300", "seeds": [1, 2, 3, 4, 5] }
```

The state machine template lives at `infra/step-function.template.json`, rendered
with `AWS_REGION`, `AWS_ACCOUNT_ID`, and `LAMBDA_PREFIX`.

### The scrape machine (pending deployment)

`infra/scrape-pipeline.step-function.template.json` takes a whole Gutenberg subject
and scrapes every book in it. **It has no safe deploy path yet.** Until it gets one,
drive the subject flow through the `list` stage directly — see
[scrape](../functions/scrape/README.md). The template is kept as the recorded design.

> **Do not deploy it with the `STEP_FUNCTION_TEMPLATE` override.** `deploy_step_function.sh`
> will happily render any template that variable points at, but the state machine name it
> deploys to is fixed at `${LAMBDA_PREFIX}-pipeline`. That ARN already exists, so the
> script takes the update path and **replaces the training pipeline with the scrape
> machine** — no new machine, no error, no warning. Teaching the script a name per
> template is the prerequisite for deploying this one.

```text
seed-subject (stage=list)          input: { "subject": "12345" }
  |
scrape-books  Map, MaxConcurrency 1, over the seeded indexes
  |
  +-- scrape (stage=metadata) → ready-for-content? → wait 3s → scrape (stage=content)
```

Three things about it are deliberate:

- **`MaxConcurrency: 1`** — gutenberg.org is a single volunteer-run host, so books go
  through one at a time, no ruder than the `scrape.py` CLI.
- **The `Choice` is positive**, advancing only on `SCRAPED_METADATA`, which is
  `scrape_book_content`'s own precondition — so a resume short-circuits books an
  earlier run already took to `SCRAPED_HTML` instead of paying a wait and a no-op.
- **A failed book does not fail the subject.** An uncaught error in an inline `Map`
  discards every remaining iteration, so each task `Catch`es to a `Succeed`; the book
  keeps its `pipeline_status` for the next run and the count surfaces as `failed`.

The `list` stage seeds at most `MAX_BOOKS_PER_SUBJECT` (100) books, taking the most
downloaded first. That is a deliberate cap on how much of a subject enters the corpus,
not a limit of the invocation: four 25-book pages at 1s each finish well inside the
120s timeout. Raising the cap raises the runtime with it — past roughly 800 books the
invocation is killed mid-list, so seed those with `scrape.py list --subject …`, which
has no timeout.

Every state carries its own `Retry`: up to 3 attempts at `Lambda.ServiceException`,
`Lambda.AWSLambdaException`, `Lambda.SdkClientException`, and
`Lambda.TooManyRequestsException` — AWS/Lambda-service-level failures — with 1s
initial backoff, `BackoffRate` 2, and full jitter. An exception the pipeline code
itself raises is not in that list, so it fails the run rather than retrying
silently; see [Retries are scoped to the transient class only](#retries-are-scoped-to-the-transient-class-only)
for the same split applied at the API layer.

---

## Observability

One JSON line per request, emitted by `RequestLoggingMiddleware`. Nothing else
in the app writes application logs directly — any code that wants a field on
the line calls `add_to_log(**fields)`, which mutates a per-request dict held in
a `ContextVar`.

The `ContextVar` is load-bearing: FastAPI dispatches sync `def` handlers to a
worker thread, and `contextvars` copies the context across that hop. A plain
rebind (`.set()`) inside a handler would not be visible back in the middleware
that emits the line; a mutation of the same dict is. Every `add_to_log` call in
the codebase depends on this distinction.

The line always carries `method`, `path`, `status`, and `dur_ms`, plus `route`
and any path params once routing has matched. Handlers add request-specific
fields on top — `query`, `warm_ms`, `nearest_terms_ms`, `similarities_ms`,
`scored_terms`, `vocab_terms`, and similar. An unhandled exception still produces a line: the
middleware catches it, records `status=500` and the exception type and message,
re-raises, and only then emits — so a 500 always leaves a trace instead of the
request silently vanishing from the logs.

This applies only to `lambda-api` in practice — it is the only stage that serves
live requests — but is documented here rather than under `api` in
[the pipeline map](./internals.md) because the pattern (ContextVar-backed log
context, one line per unit of work) is the system's general logging
convention, not an API-specific mechanism.

---

## Configuration and deployment

### Lambda resources

| Function | Memory | Timeout | Rationale |
|---|---:|---:|---|
| `scrape` | 256 MB | 120s | I/O-bound HTTP fetch |
| `tokenize` | 512 MB | 120s | spaCy model needs headroom |
| `train-kvector` | 1536 MB | 600s | CPU-bound Word2Vec training |
| `align-kvectors` | 256 MB | 120s | NumPy/SciPy on pre-loaded vectors |
| `publish` | 512 MB | 300s | Loads all models + batch writes |
| `api` | 1024 MB | 120s | Holds every requested book's term matrices in memory |

Edit `infra/services.yaml` to change these.

### Environment

```bash
AWS_REGION=
AWS_ACCOUNT_ID=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_ECR_REPO=
LAMBDA_ROLE_ARN=
LAMBDA_PREFIX=
STEP_FUNCTION_ROLE_ARN=
S3_BUCKET=
S3_TEST_DATA_PREFIX=    # e.g. test-data/ (integration tests)
PIPELINE_TABLE=         # DynamoDB, pipeline state
BOOK_TERM_TABLE=        # DynamoDB, term vectors
TERM_CORPUS_TABLE=      # DynamoDB, term → books
REDIS_URL=              # optional
REDIS_PREFIX=
PRODUCTION_DOMAIN=      # Frontend URL, for CORS
OPENAI_API_KEY=         # Required for /parse-describe
PINECONE_API_KEY=       # Required by publish
PINECONE_INDEX_NAME=
```

### Prerequisites

Docker + Docker Compose, AWS CLI, [`yq`](https://github.com/mikefarah/yq) (parses
`services.yaml`), [`envsubst`](https://www.gnu.org/software/gettext/) (renders the
state machine template). Redis optional.

> **Apple Silicon:** `deploy_lambdas.sh` forces `--platform linux/amd64` via
> `docker buildx`. Make sure buildx is available.

### Deploying

`deploy_lambdas.sh` takes service names, builds for `linux/amd64`, runs the
service's suite inside a dedicated `test` stage of its Dockerfile against the
production dependency set, pushes to ECR, then creates or updates the function —
skipping the update if image and configuration are unchanged. A failing test
aborts before any image is pushed.

```bash
./infra/deploy_lambdas.sh scrape tokenize train-kvector align-kvectors publish api
./infra/deploy_step_function.sh
```

Services can define a `smoke_cmd` in `services.yaml` that runs against the
production image before it is pushed.

---

## Cross-stage decisions

Rationale that spans more than one stage, split out of
the stage documentation because
neither of these is really about a single stage.

### Retries are scoped to the transient class only

*Implemented client-side, in the frontend, for the API. Documented here because
the policy only works if the backend's error taxonomy makes "transient" a
decidable question — that split is what this decision is really about — and
because the same split is applied independently in the training pipeline's
orchestration.*

**The situation.** The API is a Lambda behind a Function URL doing heavy
synchronous work on a single worker. Two pressures pull opposite ways: retrying
broadly doubles load on an already-busy server and delays the message the reader
needs, while retrying nothing means a cold start — which says nothing about the
query and would succeed on a second attempt — surfaces as a hard error.

**The decision.** Retry exactly once, after 2s, and only where the failure could
plausibly be transient: a network error or a 5xx. Every 4xx is a deterministic
answer about the expression itself (`expression_absent`,
`query_in_too_few_books`, a 422), so retrying one buys nothing and costs the
reader time.

**Why it's scoped this way.** Narrowing to the transient class absorbs cold
starts without adding load in the cases where load is the problem — the retry
budget goes only where a second attempt can actually change the outcome.

The same split is applied independently one layer down: every state in the
training pipeline's Step Function ([Orchestration](#orchestration)) retries
`Lambda.ServiceException` and its siblings — AWS-service-level failures — up to
3 times with exponential backoff and full jitter, and does not retry an
exception raised by the pipeline code itself. Same principle, applied where the
pipeline needed it rather than copied from where the API needed it.

### Deploys are test-gated

`deploy_lambdas.sh` builds a dedicated `test` stage of each service's Dockerfile
and runs the suite inside it, against the production dependency set, before any
image is built or pushed. A failing test aborts the deploy rather than shipping
and alerting. Applies uniformly to every stage in [Deploying](#deploying), not
just one.
