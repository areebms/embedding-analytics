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

The state machine definition lives at `infra/step-functions/pipeline.asl.json`, rendered
with `AWS_REGION`, `AWS_ACCOUNT_ID`, and `LAMBDA_PREFIX`. **Nothing in the repo deploys
it any more.** The scrape/standardize flow moved to CDK and took `deploy_step_function.sh`
with it, and this machine's per-book training stages have not been converted, so no stack
in `infra/app.py` claims it.

That is not the same as absent: `${LAMBDA_PREFIX}-pipeline` is live in the account and
still the path a per-book run takes. It is the *edit* path that is gone. Until the stages
are converted, a change to this file reaches AWS only by hand — substitute the three
placeholders yourself, then:

```bash
aws stepfunctions update-state-machine \
    --state-machine-arn "arn:aws:states:${AWS_REGION}:${AWS_ACCOUNT_ID}:stateMachine:${LAMBDA_PREFIX}-pipeline" \
    --definition file://rendered.json
```

### The scrape machine

`infra/step-functions/scrape-pipeline.asl.json` takes a whole Gutenberg subject and
scrapes every book in it, then announces that its books are ready. It names no other
machine. An EventBridge rule matches that announcement and starts
`infra/step-functions/standardize.asl.json`, which opens the Anthropic batch.

The two pipelines are separate CDK stacks, and what joins them is a third:

| Stack | Owns |
| --- | --- |
| `${LAMBDA_PREFIX}-scrape` | the `scrape` Lambda and the `-scrape-pipeline` machine |
| `${LAMBDA_PREFIX}-standardize` | the `standardize-headings` Lambda and the `-standardize` machine |
| `${LAMBDA_PREFIX}-relay` | the `-standardize-trigger` rule, and nothing else |

Both pipeline stacks are the same `PipelineStack`
(`infra/stacks/pipeline_stack.py`), instantiated twice in `infra/app.py` with a different
service and ASL file. **No stack references a resource in another**, so no template carries
an `Fn::ImportValue` and CloudFormation never holds one stack up on another's export: each
is deployed, rolled back and destroyed on its own, and `cdk destroy
${LAMBDA_PREFIX}-relay` severs the two pipelines without either of them changing.

The price of that is the ordering the old single stack derived for free. The rule names its
target by ARN rather than by construct reference, so nothing verifies the machine exists —
and `PutTargets` accepts an ARN that resolves to nothing and fails only at delivery,
silently. `infra/app.py` therefore declares a stack-level dependency of `-relay` on
`-standardize`, which orders the deploy through the cloud assembly manifest without
creating the export. `cdk deploy --all` and [`infra/deploy.sh`](#deploying) both honour it;
a hand-written `cdk deploy '*-relay' --exclusively` against an account with no standardize
machine does not.

```text
${LAMBDA_PREFIX}-scrape-pipeline        input: { "subject": "12345" }
  |
list-subject-books (stage=SUBJECT)
  |
scrape-books  Map, MaxConcurrency 1, over the seeded indexes
  |
  +-- pace 3s → scrape (stage=METADATA) → pace 3s → scrape (stage=CONTENT)
  |             \ Catch → book-failed          \ Catch → book-failed
  |
books-to-standardize?  no book at SCRAPED_HTML → subject-done
  |
announce-books-scraped  events:putEvents ──┐   Catch → announce-failed (Fail)
  |                                        |
announce-delivered?  FailedEntryCount > 0 ─┘
  |                                        |
subject-done                               v
                            EventBridge default bus
                              source      embedding-analytics.scrape
                              detail-type Subject Books Scraped
                              detail      { subject, book_ids, scrape_execution }
                                           |
                            ${LAMBDA_PREFIX}-standardize-trigger  (rule)
                              InputTransformer → { book_ids, subject }
                                           |
                                           v
${LAMBDA_PREFIX}-standardize   input: { "book_ids": [...] }
  |
standardize-submit ({book_ids})   BooksInFlightError → standardize-blocked (Succeed)
  |                               anything else      → standardize-submit-failed (Fail)
  |
batch-ended? ⇄ wait-for-batch → standardize-collect ({batch_id}), until it ends
  |                                 a raise here fails the execution at this task,
  |                                 uncaught, so redrive can reschedule it
  |
standardize-done   out: { batch_id, book_count, batch_status, standardized }
```

The two halves are one job but not one shape. The scrape half is bounded — at most
`MAX_BOOKS_PER_SUBJECT` books, each a fixed handful of states — while the batch poll
below it has no bound at all: it runs a `Wait` and a `Task` every 300s for as long as
Anthropic takes, which can be hours. A Standard workflow's history caps at 25,000
events, and fused they shared one budget, sized by the half nobody can size. Split,
each gets its own. The split also isolates the failure: a poll that dies now fails a
machine that does nothing but poll, so restarting it restarts only the poll — see
[Recovering a batch](#recovering-a-batch).

### The scrape machine announces, it does not call

`announce-books-scraped` emits one EventBridge event and ends. This machine names no
other machine; `${LAMBDA_PREFIX}-standardize-trigger` is what turns the event into an
execution, and it lives in its own stack.

The call was here twice before, and each move fixed a different problem.
`startExecution.sync:2` made the scrape execution the child's **owner** rather than its
caller — stop or time out the parent and Step Functions stops the child, which on the
poll loop abandons an open, already-paid-for Anthropic batch. Plain `startExecution`
fixed the lifecycle but still hard-coded what happens after a scrape, in the ASL of a
machine that should only be about scraping. An announcement fixes that, and gives a
second consumer of "a subject finished scraping" somewhere to attach.

Four things carry it:

- **The event carries `book_ids`.** The alternative — a bare signal, with `SEND` finding
  its own work through `status-index` — was available and cheap, but it would have cost
  `SEND` its required, non-empty work list and left two concurrent sweeps racing to
  submit the same books. Carrying the list keeps `get_entries` refusing in-flight books,
  and that refusal is what makes at-least-once delivery safe: a duplicate event starts a
  second execution over books already at `STANDARDIZE_SUBMITTED`, which raises
  `BooksInFlightError` and lands on `standardize-blocked`. Nothing is paid for twice.
- **`books-to-standardize?` keeps the event off the empty path**, which is what lets the
  rule match on `detail-type` alone. This matters more than it looks: an EventBridge
  pattern cannot reach into a stringified field, so a rule on
  `Step Functions Execution Status Change` could not have told a subject with books from
  one without — `detail.output` arrives as a JSON *string*. The event only exists when
  there is work, so the question never has to be asked.
- **`announce-delivered?` is not optional.** `PutEvents` reports a rejected entry in
  `FailedEntryCount` and succeeds the task anyway. Without the `Choice` the one failure
  that matters — the announcement never reaching the bus — reads as a finished subject
  that then silently never standardizes.
- **The join got weaker, and that is the price.** An EventBridge target cannot set an
  execution name, so the old `<subject>-<parent execution name>` and its
  `ExecutionAlreadyExists` catch are both gone. `subject` rides in the event detail and
  through the transformer into the standardize machine's *input*, where it shows on the
  execution in the console — but the execution list can no longer be scanned by subject.

**Nothing watches the standardize machine.** A failed collect ends red on its own
machine, strands books at `STANDARDIZE_SUBMITTED`, and no scrape execution turns red with
it. An alarm on `ExecutionsFailed` for `${LAMBDA_PREFIX}-standardize` is the minimum cover
and does not exist yet. It matters more than an alarm usually does, because
[Recovering a batch](#recovering-a-batch) has a clock on it: a failed execution is
redrivable for 14 days after it ends, and nobody is told it failed.

`STEP_FUNCTION_ROLE_ARN` needs `lambda:InvokeFunction` on
`${LAMBDA_PREFIX}-standardize-headings`, which only the standardize machine calls, and
`events:PutEvents` on the default bus, which only the scrape machine uses. It needs no
`states:` permission at all any more — no machine here starts another. `PUT_EVENT_ROLE_ARN`
is the new one: trusted by `events.amazonaws.com`, holding `states:StartExecution` on
`${LAMBDA_PREFIX}-standardize`, and it is the rule that assumes it, not a state machine.
Between them these are much smaller than the `.sync` era needed — that integration is
built on a managed EventBridge rule, so it also required `states:DescribeExecution`,
`states:StopExecution`, and `events:PutRule` / `PutTargets` / `DescribeRule` on
`StepFunctionsGetEventsForStepFunctionsExecutionRule`.

`states:RedriveExecution` on `execution:${LAMBDA_PREFIX}-standardize:*` belongs to whoever
runs [Recovering a batch](#recovering-a-batch) — a human, or whatever runs on their
behalf. It is not a machine permission and does not go on `STEP_FUNCTION_ROLE_ARN`: no
state in either machine redrives anything.

Five things about the scrape half are deliberate:

- **`MaxConcurrency: 1`** — gutenberg.org is a single volunteer-run host, so books go
  through one at a time, no ruder than the `scrape.py` CLI. It is also what makes the
  pace states below a throttle rather than decoration: parallel iterations would overlap
  and the request rate would be whatever concurrency allowed.
- **Every gutenberg request is preceded by a pace**, a book's first one included, so the
  book boundary is throttled like every other request. Pacing *before* the fetch rather
  than after is also what lets `book-failed` stay a `Succeed` — an iteration that ends
  early cannot make the next book fetch immediately, because that book opens with its
  own pace. The cost is that the pace is unconditional: a book needing no network still
  pays 3s and an invocation that does one DynamoDB read, so re-running a fully-scraped
  100-book subject costs about ten minutes of wall clock and no gutenberg traffic at
  all — the listing walk is skipped too, once the subject is at the cap described below.
- **The machine does not re-decide what the handler already decides.**
  `scrape_book_content` guards on its own precondition and returns the book's existing
  status untouched, so `stage=CONTENT` runs unconditionally. The `Choice` that used to
  gate it was a second copy of that guard written against a status literal, and it broke
  silently when `EntryStatus` gained rank prefixes (`0100_SCRAPED_METADATA`). The one
  status comparison left — the `Map`'s `book_ids` filter — matches on a substring for
  the same reason.
- **A failed book does not fail the subject.** An uncaught error in an inline `Map`
  discards every remaining iteration, so each task `Catch`es to a `Succeed`; the book
  keeps its `status` for the next run and the count surfaces as `failed`. It is also
  what lets `announce-books-scraped` run at all — a `Fail` would abort the `Map`,
  leaving the subject scraped and never announced.
- **The `Map` hands the standardize machine its work list.** Its `Output` filters the
  per-book results to `SCRAPED_HTML` and passes them as that machine's whole input, so
  `SEND` reads exactly those rows in one `BatchGetItem` instead of Scanning the table for
  the subject twice. A book an earlier run left unsubmitted still comes back — both
  scrape stages report the status they find rather than refetching, so it leaves the
  `Map` still marked `SCRAPED_HTML` and the filter keeps it. Only one thing drops a book now: a caught error this run, since
  `book-failed` carries no `status` for the filter to match. Re-ranking used to be the
  other — a book that slid past the cap fell out of the listing and so out of
  `indexes` — which is why a subject already holding its cap is served from the table
  instead of re-listed.

`SEND` and `RETRIEVE` report the same `batch_status` field, so one `Choice` covers both
"nothing was submitted" and "the batch is still running" — `SEND` reports `ended` when
it opened no batch, and the poll loop is simply never entered. `batch-ended?` is written
as the negative, so an unfamiliar `processing_status` keeps polling rather than reporting
the batch done.

Two guards sit around `standardize-submit`, and both exist because the interesting
failures happen *after* the scrape has already been paid for:

- **`books-to-standardize?` gates the empty work list**, and it stays in the parent so an
  empty list costs no child execution at all. A subject where no book reached
  `SCRAPED_HTML` — every book non-English, or already standardized — is an ordinary
  outcome, but `SEND` requires `book_ids` and raises without it. Handing it an empty list
  raised `ValueError` in the handler — two falsy fields are neither entry point — and put
  the execution on `standardize-submit-failed`, turning a subject that simply had no work
  red. The gate keeps the handler's required field required and stops the machine being
  started with an empty work list.
- **`standardize-blocked` is scoped to `BooksInFlightError`.** As a `States.ALL` catch it
  also swallowed a half-finished submit: `submit` opens the batch before it writes the
  manifest and before it moves any status, so a raise in that window left a batch that was
  paid for, had no manifest to settle against, and whose books were still at
  `SCRAPED_HTML` — reported as a `Succeed`. Everything that is not the in-flight refusal
  now ends at `standardize-submit-failed`.

`standardize-collect` is the exception: it catches nothing at all, which is what makes
the poll loop restartable — see [Recovering a batch](#recovering-a-batch) at the end of
this section.

The `SUBJECT` stage seeds at most `MAX_BOOKS_PER_SUBJECT` (100) books, taking the most
downloaded first. That is a deliberate cap on how much of a subject enters the corpus,
not a limit of the invocation: four 25-book pages at 1s each finish well inside the
120s timeout. Raising the cap raises the runtime with it — past roughly 800 books the
invocation is killed mid-list, so seed those with `scrape.py SUBJECT --subject …`,
which has no timeout.

Once a subject already holds that many books, the stage stops listing it: it serves
`indexes` from the table and returns without a single gutenberg request. The cap is the
ceiling on what a subject contributes, so a subject sitting at it has nothing left to
discover — a second walk could only re-rank the same books, never add one. Skipping it
is also what keeps a re-run from stranding a book. The listing is sorted by download
count, so re-walking it rebuilds the work list from a *live* ranking: a book scraped by
an earlier run that had since drifted past rank 100 dropped out of `indexes`, never
reached the `Map`, and sat at `SCRAPED_HTML` forever. Reading the table instead makes
`indexes` a superset of every previous run's work list. A re-run reports
`created: 0`, which on this path means the listing was never walked rather than that it
was walked and yielded nothing new.

Every Lambda task carries its own `Retry`: up to 3 attempts at `Lambda.ServiceException`,
`Lambda.AWSLambdaException`, `Lambda.SdkClientException`, and
`Lambda.TooManyRequestsException` — AWS/Lambda-service-level failures — with 1s
initial backoff, `BackoffRate` 2, and full jitter. An exception the pipeline code
itself raises is not in that list, so it fails the run rather than retrying
silently; see [Retries are scoped to the transient class only](#retries-are-scoped-to-the-transient-class-only)
for the same split applied at the API layer. `announce-books-scraped` is the one task with
no `Retry` at all, for the reason given above: a blind re-emit of an event that may already
have been delivered is worse than a `Fail` an operator can see.

### Recovering a batch

A raise inside `RETRIEVE` that survives the retrier strands every book it had not reached
at `STANDARDIZE_SUBMITTED`, and `get_entries` refuses any later run of any subject holding
one of them. The batch is already paid for and nothing else will settle it, so there has to
be a way back in.

That way is `redrive-execution`, which restarts an unsuccessful Standard execution from the
state that failed and reschedules it with the input that state recorded — here, the
`batch_id` the poll loop was carrying. The whole recovery is the execution's own ARN:

```bash
aws stepfunctions redrive-execution \
    --execution-arn "arn:aws:states:${AWS_REGION}:${AWS_ACCOUNT_ID}:execution:${LAMBDA_PREFIX}-standardize:<name>"
```

Three things in the machine exist to keep that command working:

- **`standardize-collect` has no `Catch`.** A caught error would end the execution on a
  `Fail` state, and redrive re-enters a `Fail` and fails again — the execution would be
  redrivable in name only. The raise has to fail the execution *at the task* for redrive to
  have something to reschedule. This is the one place in either machine where an uncaught
  raise is the design rather than an oversight.
- **`standardize-submit` keeps its `Catch`, and that is the same argument inverted.** Its
  `Fail` makes the submit path deliberately un-redrivable, which is correct: `submit` opens
  the batch before it writes the manifest and before it moves any status, so a redrive that
  reran it would find the books still at `SCRAPED_HTML` and open — and pay for — a second
  batch.
- **The machine has one entry point.** `{ book_ids }` and nothing else. The `{ batch_id }`
  entry point that used to open this machine was a hand-rolled second copy of redrive, and
  it could not coexist with it: keeping the `Catch` that spelled the id into a `Cause` is
  exactly what stopped redrive from working.

Two limits are worth knowing, given that nothing yet alarms on the failure. Redrive is
available for **14 days** after the execution ends, and the redriven attempt appends to
the same execution history, which must stay under 24,999 events — a 36h poll at 300s is
a few thousand, so only repeated redrives approach it. Past 14 days, or for a batch
orphaned by `standardize-submit-failed` rather than by the poll, the fallback is invoking
the Lambda directly with `{"batch_id": "..."}`, which settles an ended batch in one call;
the id is recoverable from `client.messages.batches.list()` for 29 days after the batch
was created.

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
| `standardize-headings` | 1024 MB | 600s | Holds every pending book's flattened text while building one batch |
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
PUT_EVENT_ROLE_ARN=    # assumed by EventBridge, starts the standardize machine
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
ANTHROPIC_API_KEY=      # Required by standardize-headings
PINECONE_INDEX_NAME=
```

### Prerequisites

Docker + Docker Compose, AWS CLI, [`yq`](https://github.com/mikefarah/yq) (parses
`services.yaml` for `deploy_lambdas.sh`), and Python 3.13. Redis optional.

The CDK app and the `cdk` CLI both install from one file:

```bash
python3.13 -m pip install -r infra/requirements.txt
```

`aws-cdk-cli` is the CDK CLI published on PyPI, with its own bundled Node runtime — there
is no `package.json`, no `npx`, and no global install, so the CLI version is pinned in the
same file as the library. Use `python3.13` specifically: bare `python3` is 3.8 here and
dies on `StrEnum` in `shared/`.

> **Secrets.** CDK reads `ANTHROPIC_API_KEY` from `.env` at synth time and writes it into
> `infra/cdk.out/*.template.json`, which `cdk deploy` uploads to the CDK staging bucket.
> Under `deploy_lambdas.sh` it never left the machine. Moving it to SSM and referencing it
> with `ssm.StringParameter.value_for_string_parameter` would put only the parameter name
> in the template; see the TODO in `infra/config.py`.

> **Apple Silicon:** `deploy_lambdas.sh` forces `--platform linux/amd64` via
> `docker buildx`. Make sure buildx is available.

### Deploying

Two paths, split by service. `scrape` and `standardize-headings` are CDK; the other five
are still `deploy_lambdas.sh`.

```bash
./infra/deploy.sh                        # all three stacks, standardize before relay
./infra/deploy.sh scrape                 # the scrape pipeline, alone
./infra/deploy.sh standardize            # the standardize pipeline, alone
./infra/deploy.sh relay                  # just the rule
./infra/deploy_lambdas.sh tokenize train-kvector align-kvectors publish api
```

`infra/deploy.sh` runs each service's suite inside the `test` stage of its Dockerfile,
then hands off to `cdk deploy`, which builds the `lambda` stage as a content-hashed image
asset, pushes it, and applies the stack. A failing test aborts before anything is built or
pushed. Arguments go to `cdk deploy`; a leading `--` replaces the subcommand, so
`./infra/deploy.sh -- diff` is a test-gated `cdk diff`.

**The optional first argument is the point of the split.** With no target both suites run
and all three stacks deploy; with one, only that pipeline's suite runs and only its stack
is touched — a `standardize-headings` test that is red cannot hold up a `scrape` release,
and a scrape deploy produces no changeset over the standardize machine. A targeted deploy
passes `--exclusively`, so `deploy.sh relay` does not follow the stack dependency into
`-standardize` and publish an image whose suite this invocation never ran.

That isolation reaches the images too, which is less obvious. The Docker build context has
to be the repo root — every Dockerfile copies `shared/` out of it — and everything left in
the context lands in the asset hash, so by default a change anywhere in the repo
republishes every Lambda image. `.dockerignore` drops `infra/`, `docker-compose.yml` and
the local `.venv/` for that reason, and `PipelineStack` excludes the sibling `functions/`
directories per image, along with that service's own `tests/`, `pytest.ini` and
`requirements-test.txt`, and `shared/tests/` — none of which the `lambda` stage COPYs, and
all of which would otherwise republish an identical image on a test-only edit. The
per-service exclusions cannot move to `.dockerignore`, which is context-wide and shared by
both images and by the `test` build. Excluding the tests from the asset does not weaken the
gate: `deploy.sh` builds `--target test` in its own `docker buildx` invocation, reading the
real tree.

In the scrape and standardize Dockerfiles those copies live in the `lambda` stage, not in
`base`: that stage is the single statement of what ships, and it names `shared/`'s runtime
modules (`shared/*.py` and `shared/tables`) rather than taking the directory wholesale, so
`shared/tests` stays out of the running image as well as out of the hash. Excluding it by
deleting it further down would not work — layers only ever add, and a `RUN rm` leaves the
bytes in the layer beneath. A new *subpackage* under `shared/` needs a line added to both
Dockerfiles; a new top-level module is covered by the glob. An omission cannot reach
production silently: `test` builds `FROM lambda`, so the suite runs against the exact image
that ships and the gate fails first.

What remains in a service's asset context is `shared/`'s runtime modules and its own `src/`
and `Dockerfile`: editing a CDK stack rebuilds nothing, and editing one service rebuilds
only that one. Touching `shared/` still rebuilds both, correctly — the seven services agree
on `shared/tables/pipeline_entries.py` against one DynamoDB table, and pinning them apart
would let two of them disagree about the schema.

`deploy_lambdas.sh` takes service names, builds for `linux/amd64`, runs the
service's suite inside a dedicated `test` stage of its Dockerfile against the
production dependency set, creates the ECR repository if it does not exist yet,
pushes to it, then creates or updates the function — skipping the update if
image and configuration are unchanged. A failing test aborts before any image is
pushed.

The name it takes is the `services.yaml` key, and that key is the service's whole
identity: the function deploys as `${LAMBDA_PREFIX}-<key>`, the Dockerfile is read from
`functions/<key>/`, and the ECR image is `lambda-<key>`. Only the values that vary —
`memory`, `timeout`, `env` — are written per service, and a name the file does not list
is rejected rather than deployed on the defaults. `infra/config.py` derives the same
three names from the same key, so the two paths cannot drift on naming either.

A service may declare an `env` list in `services.yaml`, naming the variables it
needs at runtime; the script reads each value from `.env` and passes the set as
the function's environment, aborting if any is unset. `AWS_REGION` is never listed:
Lambda injects it into every runtime and rejects it as a reserved key.

The CDK app (`infra/config.py`) reads the same `env` lists from the same `services.yaml`,
with the same abort on a missing value, so the two paths cannot drift on sizing or
configuration. They differ in one place: **under `deploy_lambdas.sh`, a service with no
`env` key is left alone**, because `--environment` replaces a function's whole variable
map rather than merging into it, so deploying such a service must not clear variables set
outside the script. CDK always declares the full map, so a converted service must list
every variable it needs. Both in-scope services already did.

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

Both deploy paths build a dedicated `test` stage of each service's Dockerfile and run
the suite inside it, against the production dependency set, before any image is built or
pushed. A failing test aborts the deploy rather than shipping and alerting. Applies
uniformly to every stage in [Deploying](#deploying), not just one.

`deploy_lambdas.sh` does this inline. On the CDK side it is why `infra/deploy.sh` exists
at all: `cdk deploy` builds only the Dockerfile's `lambda` stage and has no notion of a
`test` stage, so a bare `cdk deploy` would silently drop the gate. Run `./infra/deploy.sh`,
not `cdk deploy`.

A targeted deploy runs only that pipeline's suite, so it also passes `--exclusively`: the
gate cannot be bypassed by a stack dependency quietly pulling in a second image that this
invocation never tested.
