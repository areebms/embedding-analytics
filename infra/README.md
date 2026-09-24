# infra

*[Pipeline overview](../docs/pipeline.md) · [Project README](../README.md)*
**Libraries:** AWS CDK (Python), PyYAML, python-dotenv

The CDK app, the deploy gate in front of it, and everything about running the pipeline
that spans more than one stage. Each stage documents its own mechanics in its own
directory. Nothing here ships in a Lambda image: `.dockerignore` drops this directory,
so editing a stack rebuilds no image.

## Layout

| File | Owns |
|---|---|
| `app.py` | The one `build()` that assembles the stack from `get_services()` |
| `config.py` | `services.yaml` + `.env` resolution — `PREFIX`, `get_services()`, `get_service_config()` |
| `resources.py` | Every construct: the functions, the machines, the rules, the roles |
| `pipeline_events.py` | The four announcements, declared once as frozen dataclasses |
| `deploy.py` | The test gate, then `exec` into `cdk` |
| `services.yaml` | Per-service memory, timeout, and environment variable *names* |
| `cdk.json` | `python3.13 app.py`, output to `cdk.out` |
| `step-functions/*.asl.json` | The two state machine definitions, rendered at synth |

`config.py` is imported by all of the others and imports none of them.

One CloudFormation stack, named for `ENV_PREFIX`: six Lambdas, two state machines, four
EventBridge rules. Every physical name is `${ENV_PREFIX}-<key>`, and the `services.yaml`
key also selects `functions/<key>/Dockerfile`. The three IAM roles (`LAMBDA_ROLE_ARN`, `STEP_FUNCTION_ROLE_ARN`,
`PUT_EVENT_ROLE_ARN`) already exist and are imported with `mutable=False`, so CDK never
appends policies to them ([Permissions](#permissions)).

Every `services.yaml` key except `default` is deployed and gated by `deploy.py`, so a
service cannot ship ungated.

## Orchestration

```text
scrape → standardize-html → tokenize → create-embeddings → publish
```

No stage names the next. Each announces on the EventBridge default bus when it
finishes, and one rule per consumer starts the next step:

| Announcement | Emitted by | Rule | Starts |
| --- | --- | --- | --- |
| `Subject Books Scraped` | the scrape machine's `announce-books-scraped` | `${ENV_PREFIX}-standardize-trigger` | the standardize machine |
| `Books Standardized` | the standardize machine's `announce-books-standardized` | `${ENV_PREFIX}-tokenize-trigger` | the `tokenize` Lambda |
| `Books Tokenized` | `tokenize`'s own `announce_tokenized` | `${ENV_PREFIX}-create-embeddings-trigger` | the `create-embeddings` Lambda |
| `Books Embedded` | `create-embeddings`'s own `announce_embeddings_creation` | `${ENV_PREFIX}-publish-trigger` | the `publish` Lambda |

`pipeline_events.py` declares each announcement once (`source`, `detail_type`,
`detail_keys`), and both the emitter's ASL and the consumer's rule are built from it.
Every rule matches on `source` and `detail-type` alone and forwards `$.detail` as the
target's whole input, so each consumer gets the `book_ids` its producer just finished.
CDK resolves each rule's target from its construct, so a rule cannot point at nothing.

### The two machines

Both definitions live in `step-functions/` and are rendered at synth time with
`FUNCTION_ARN` and `ENV_PREFIX` substitutions.

```text
${ENV_PREFIX}-scrape        input: { "subject": "12345" }
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
                            source      embedding-analytics.scrape
                            detail-type Subject Books Scraped
                            detail      { subject, book_ids, scrape_execution }
                                           |
                            ${ENV_PREFIX}-standardize-trigger, input path $.detail
                                           |
                                           v
${ENV_PREFIX}-standardize-html   input: { "book_ids": [...] } or { "subject_id": "12345" }
  |
standardize-submit ({book_ids})   BooksInFlightError → standardize-blocked (Succeed)
  |                               anything else      → standardize-submit-failed (Fail)
  |
batch-settled? ⇄ wait-for-batch → standardize-collect ({batch_id}), until it ends
  |                                 a raise here fails the execution at this task,
  |                                 uncaught, so redrive can reschedule it
  |
announce-books-standardized  events:putEvents ──┐  Catch → announce-failed (Fail)
  |                                             |
announce-delivered?  FailedEntryCount > 0 ──────┘
  |                                             |
standardize-done                                v
                            source      embedding-analytics.standardize
                            detail-type Books Standardized
                            detail      { batch_id, book_ids, standardized,
                                          standardize_execution }
```

The scrape half is bounded (at most `MAX_BOOKS_PER_SUBJECT` books), while the batch poll
runs every 300s for as long as Anthropic takes, possibly hours. Split, each half gets its
own 25,000-event history budget, and a poll that dies fails only the machine that polls.
The last three stages are Lambdas because none of them waits on anything.

- **`announce-delivered?` is required.** `PutEvents` reports a rejected entry in
  `FailedEntryCount` and still succeeds, so without the `Choice` an undelivered
  announcement reads as a finished subject.
- **The join is by event detail.** An EventBridge target cannot set an execution name,
  so `subject` rides in the detail: visible on the execution, not searchable in the list.

## How a stage is invoked

`tokenize`, `create-embeddings` and `publish` accept exactly one of:

```json
{ "book_ids": ["gutenberg-3300", "gutenberg-33310"] }
{ "subject_id": "12345" }
```

`book_ids` is the handover from the previous announcement. `subject_id` is for a hand
re-run: it resolves to the subject's books at the status the stage consumes, sorted by
id and capped at `MAX_BOOKS_PER_SUBJECT` (50), so re-invoking drains a backlog in a
stable order. Neither or both raises `ValueError`. No stage sweeps the status index, so a
book stranded by a failed run rejoins only when something names it again.

One invocation takes the whole list, loading a model or table client once. A book that
raises is logged, counted in `failed`, and left at its incoming status. `found` counts
the named books that were at the consumed status.

| Stage | Consumes | Leaves | Reply |
|---|---|---|---|
| `tokenize` | `STANDARDIZED` | `TOKENIZED` | `{ found, tokenized, failed }` |
| `create-embeddings` | `TOKENIZED` | `EMBEDDINGS_CREATED`, or `EMBEDDINGS_CREATION_FAILED` | `{ found, embedded }` |
| `publish` | `EMBEDDINGS_CREATED` | unchanged — publishing writes no status | `{ found, published, failed }` |

## Re-running a subject

Start the machine, not the Lambda: invoking `standardize-html` directly opens a batch
that nothing will collect.

```bash
aws stepfunctions start-execution \
    --state-machine-arn "arn:aws:states:${AWS_REGION}:${AWS_ACCOUNT_ID}:stateMachine:${ENV_PREFIX}-standardize-html" \
    --input '{"subject_id":"12345"}'
```

That re-runs from standardization onwards. To include scraping, start
`${ENV_PREFIX}-scrape` with `{"subject":"12345"}`; a subject already at its cap makes no
gutenberg requests.

To re-run one stage over named books, invoke its Lambda with `{"book_ids": [...]}`.
`tokenize` and `create-embeddings` announce from inside the function, so the chain
carries on; `publish` ends where it is invoked.

## Permissions

| Role | Assumed by | Needs |
|---|---|---|
| `STEP_FUNCTION_ROLE_ARN` | both machines | `lambda:InvokeFunction` on `${ENV_PREFIX}-scrape` and `${ENV_PREFIX}-standardize-html`; `events:PutEvents` on the default bus. No `states:` — no machine starts another |
| `PUT_EVENT_ROLE_ARN` | the standardize rule (`events.amazonaws.com`) | `states:StartExecution` on `${ENV_PREFIX}-standardize-html` |
| `LAMBDA_ROLE_ARN` | every function | `events:PutEvents`, because `tokenize` and `create-embeddings` announce from inside |

Lambda-target rules need no role: CDK adds a resource policy to the function.
`states:RedriveExecution` on `execution:${ENV_PREFIX}-standardize-html:*` belongs to
whoever runs [Recovering a batch](../functions/standardize-html/README.md#recovering-a-batch).

## Retries

Every Lambda task in the machines retries up to 3 times on `Lambda.ServiceException`,
`Lambda.AWSLambdaException`, `Lambda.SdkClientException` and
`Lambda.TooManyRequestsException` (1s initial backoff, `BackoffRate` 2, full jitter).
Exceptions raised by pipeline code are not retried. `announce-books-scraped` has no
`Retry`: re-emitting a possibly delivered event is worse than a visible `Fail`.

## Configuration and deployment

### Lambda resources

| Function | Memory | Timeout | Rationale |
|---|---:|---:|---|
| `scrape` | 256 MB | 120s | I/O-bound HTTP fetch |
| `standardize-html` | 1024 MB | 600s | Holds every pending book's flattened text while building one batch |
| `tokenize` | 1769 MB | 900s | A full vCPU for spaCy, and a whole backlog in one invocation |
| `create-embeddings` | 1536 MB | 600s | PPMI/SVD over the whole book: 21.7s and ~490 MB peak RSS on a 190,000-token synthetic book |
| `publish` | 512 MB | 300s | Loads a book's embeddings + batch writes |
| `api` | 1024 MB | 120s | CPU for the similarity matmul; 256 MB gave ~0.14 vCPU |

Set in `services.yaml`, whose `default:` block supplies 256 MB, 120s, and `S3_BUCKET` and `PIPELINE_TABLE`; a
service's `env` list adds to those.

### Environment

```bash
AWS_REGION=
AWS_ACCOUNT_ID=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_ECR_REPO=
LAMBDA_ROLE_ARN=
ENV_PREFIX=              # every physical name in the stack, and the stack itself
STEP_FUNCTION_ROLE_ARN=
PUT_EVENT_ROLE_ARN=    # assumed by EventBridge, starts the standardize machine
S3_BUCKET=
S3_TEST_DATA_PREFIX=    # e.g. test-data/ (integration tests)
PIPELINE_TABLE=         # DynamoDB, pipeline state
BOOK_TERM_TABLE=        # DynamoDB, term vectors
TERM_CORPUS_TABLE=      # DynamoDB, term → books
REDIS_URL=              # required to deploy api
REDIS_PREFIX=
PRODUCTION_DOMAIN=      # Frontend URL, for CORS
OPENAI_API_KEY=         # Required for /parse-describe
ANTHROPIC_API_KEY=      # Required by standardize-html
```

### Prerequisites

Docker + Docker Compose, AWS CLI, Python 3.13, and optionally Redis. The CDK app and the
`cdk` CLI (`aws-cdk-cli`, with its own bundled Node) install together:

```bash
python3.13 -m pip install -r infra/requirements.txt
```

- Use `python3.13`: bare `python3` is 3.8 here and dies on `StrEnum` in `shared/`.
- Every image is built `--platform linux/amd64`, so Apple Silicon needs `docker buildx`.
- **Secrets:** CDK writes `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `REDIS_URL` from
  `.env` into `cdk.out/*.template.json`, which is uploaded to the CDK staging bucket.
  See the SSM TODO in `resources.py`.

### Deploying

```bash
python3.13 infra/deploy.py             # test-gated cdk deploy
```

`deploy.py` runs `app.build()`, the infra suite, then each service's suite
inside the `test` stage of its own Dockerfile (production dependencies, 85% coverage
floor), and only then `exec`s into `cdk`. Run it rather than `cdk deploy`, which builds
only the `lambda` stage and skips the gate. Everything after a leading `--` is passed on
to `cdk deploy`; any other argument is refused. Nothing is built or pushed on a failure:

| Exit | Means |
|---|---|
| `EX_USAGE` | An argument that was not `--` |
| `EX_CONFIG` | A service the gate cannot run — no Dockerfile, no `tests/`, no `test` stage |
| `EX_UNAVAILABLE` | A test image that did not build, or `cdk` that could not be run — the suite never ran |
| `EX_SOFTWARE` | A suite that ran and failed |

**Each image rebuilds only when its own files change.** The build context is the repo
root, because every Dockerfile copies `shared/`, so the context is trimmed per image:

- `.dockerignore` drops `infra/`, `docker-compose.yml` and `.venv/`.
- `get_test_files` excludes the sibling `functions/` directories, the service's own
  `tests/`, `pytest.ini`, `requirements-test.txt` and `shared/tests/`. These cannot move
  to `.dockerignore`, which every image and the `test` build share.
- Each `lambda` stage copies `shared/*.py` and `shared/tables` rather than all of
  `shared/`, so a new subpackage needs a line in each Dockerfile. `test` builds
  `FROM lambda`, so an omission fails the gate before it ships.

Editing a stack rebuilds nothing, editing one service rebuilds that one, and touching
`shared/` rebuilds all of them.

## Running the tests

```bash
python3.13 -m pytest infra
```

The suite synthesizes the app in-process against the real `functions/` tree and reaches
no AWS. `conftest.py` sets every synth variable before importing `app`, and `config.py`
loads `.env` with `override=False`, so tests run against `test-prefix`, not your `.env`.

| File | Asserts |
|---|---|
| `test_stack.py` | Resource counts, per-service sizing and environment against `services.yaml`, machine naming, each trigger matching its stage's announcement, and the two target kinds authorising differently |
| `test_gate.py` | The gate's preflight against the real tree, each refusal reported as itself, and that only a real `test` stage arms the gate |
| `test_asl_seam.py` | ASL permissions match the roles, invoked functions keep `:$LATEST`, rules forward only `$.detail`, and announced payloads match their `detail_keys` |
