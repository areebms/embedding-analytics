# Pipeline Documentation

Six containerized Lambda functions, each a single stage with its own resource profile. The per-book pipeline is orchestrated by AWS Step Functions. Artifacts flow through S3 during training; the `publish` stage flattens everything into DynamoDB for sub-second API reads.

---

## Engineering overview

The pipeline demonstrates several production backend patterns:

- **Step Functions orchestration** across independent, containerized Lambda stages
- **Parallel fan-out** via Map state for seeded model training
- **S3 as durable intermediate storage** with per-stage artifact handoff
- **DynamoDB as a denormalized serving layer** for term vectors and metadata
- **Idempotent stages** that skip completed work on retry or rerun
- **Per-service resource tuning** with stage-specific memory and timeout configuration
- **Separate batch job** for corpus-level alignment outside the per-book pipeline

High-level flow:

```text
scrape --> tokenize --> train-kvector Map(N seeds) --> align-kvectors --> publish --> api
```

A separate corpus job builds a shared cross-book alignment frame once two or more books have completed the per-book pipeline.

---

## Stages

| Stage | Input | Output |
|---|---|---|
| `scrape` | Gutenberg book ID | HTML, text, and metadata in S3 |
| `tokenize` | Raw text | Token, lemma, and POS-tag CSVs in S3 |
| `train-kvector` | Token lemmas + seed | One trained Word2Vec model in S3 |
| `align-kvectors` | N raw models | Procrustes-aligned models and centroid in S3 |
| `publish` | Aligned models + token metadata | Term vectors, counts, POS tags, stability metrics in DynamoDB |
| `api` | HTTP request | Books, terms, similarity, and parse responses as JSON |

---

## Step Function orchestration

The per-book pipeline runs as an AWS Step Function state machine:

```text
scrape
  |
tokenize
  |
train-kvector Map(N seeds in parallel)
  |
align-kvectors
  |
publish
```

Example input:

```json
{
  "index": "gutenberg-3300",
  "seeds": [1, 2, 3, 4, 5]
}
```

The `train-kvector` step uses a Map state so each seed runs as an independent Lambda invocation. Earlier steps carry the `seeds` array through Step Function output transforms, keeping orchestration logic out of individual Lambda handlers.

The state machine template lives at `infra/step-function.template.json` and is rendered with `AWS_REGION`, `AWS_ACCOUNT_ID`, and `LAMBDA_PREFIX`.

---

## `lambda-scrape`

**Location:** `functions/scrape/`
**Libraries:** BeautifulSoup, Requests

Fetches a Project Gutenberg book by ID, strips the standard header and footer, and writes clean artifacts to S3. Idempotent: skips re-scraping if the pipeline table already has an `s3_text_key` for the book.

| S3 artifact | Contents |
|---|---|
| `html/{index}.html` | Raw HTML |
| `text/{index}.txt` | Extracted body text |
| `metadata/{index}.json` | Title, author, publication metadata |

---

## `lambda-tokenize`

**Location:** `functions/tokenize/`
**Libraries:** spaCy (`en_core_web_sm`), NLTK, WordNet

Turns raw text into model-ready training data:

1. Segments into sentences with NLTK
2. Lemmatizes with spaCy (NER disabled for speed)
3. Smart-chunks large documents without splitting mid-sentence
4. Normalizes British/American spelling (e.g. "labor" to "labour")
5. Applies domain-specific aggressive lemmatization: nouns with derivationally related verbs collapse to the verb form (e.g. "production" to "produce"), controlled by a curated `ignored_nouns.txt` override list

Idempotent: skips when all three output keys already exist.

| S3 artifact | Contents |
|---|---|
| `token_texts/{index}.csv` | Original tokens, one sentence per row |
| `token_lemmas/{index}.csv` | Lowercased lemmas (Word2Vec training input) |
| `token_tags/{index}.csv` | POS tags |

---

## `lambda-train-kvector`

**Location:** `functions/train-kvector/`
**Library:** Gensim

Trains one Word2Vec model per invocation with an explicit seed for reproducibility. Training multiple seeds creates the ensemble that powers confidence intervals downstream.

Output path: `kvectors/{index}/collected/{seed}-{timestamp}-{randint}.model`

Filters lemmas to alphabetic tokens longer than 3 characters.

**Word2Vec configuration:**

| Parameter | Value |
|---|---|
| Vector size | 200 |
| Window | 10 tokens |
| Min count | 10 |
| Algorithm | Skip-gram (`sg=1`) |
| Training | Hierarchical softmax (`hs=1`) |
| Subsampling | `5e-4` |
| Negative sampling | Off |
| Epochs | 30 |

---

## `lambda-align-kvectors`

**Location:** `functions/align-kvectors/`
**Libraries:** NumPy, SciPy

Alignment solves a core Word2Vec reliability problem: independently trained models can learn equivalent internal geometry but represent it in arbitrarily rotated coordinate systems. Generalized Procrustes Analysis finds orthogonal rotations that bring models into a shared orientation without distorting cosine relationships.

### Per-book alignment

`create_book_centroid.py` aligns all seeded models for a single book into a shared vector space, builds a per-book centroid with per-term stability metrics (disparity, variance, R-squared), and optionally rotates into the corpus frame if one exists.

### Cross-book alignment

`create_corpus_centroid.py` aligns completed book centroids into a shared corpus frame for cross-author comparison. Runs outside the per-book Step Function and requires 2+ completed books. Uses unit-normalized vectors and uniform weights, with single-book terms excluded from the rotation via zero weight but retained in the vocabulary for downstream lookup.

Shared alignment primitives live in `procrustes_utils.py`.

Output location: `kvectors/{index}/aligned/`

More detail: [`alignment.md`](alignment.md)

---

## `lambda-publish`

**Location:** `functions/publish/`
**Libraries:** Gensim, NumPy

Flattens S3 training artifacts into DynamoDB rows optimized for fast API reads. For each term present in the centroid, POS tag set, and aligned model stack, writes a single row containing:

- Centroid vector (float16)
- Per-seed aligned vectors (float16)
- Token occurrence positions (`ilocs`)
- POS tags
- Word count
- Disparity, variance, and R-squared
- Author/title metadata backfilled from Gutenberg metadata

This converts model artifacts into a denormalized serving representation for the FastAPI layer.

---

## `lambda-api`

**Location:** `functions/api/`
**Libraries:** FastAPI, Mangum, Pydantic, fastapi-cache, Redis, OpenAI

The read layer over DynamoDB and the entry point for the frontend. Mangum runs FastAPI inside a Lambda Function URL. Redis caching is optional: the API runs without it when `REDIS_URL` is not set. Cached responses have no expiry.

### `GET /books`

All corpora that have completed the full pipeline.

```json
[{
  "id": 3300,
  "label": "Smith (1776)",
  "author": "Smith, Adam",
  "title": "An Inquiry into...",
  "published_year": 1776
}]
```

### `GET /terms`

Cross-book term vocabulary. Returns every term in at least 2 books, excluding adverb-only terms.

```json
[{ "term": "labour", "books": ["gutenberg-3300", "gutenberg-846"] }]
```

### `POST /similarity/{book_id}`

Accepts a recursive expression tree (Pydantic-validated). Each `+` averages and re-normalizes the two operand vectors; `-` subtracts and re-normalizes. Operations nest arbitrarily. Returns every term in the book ranked by cosine similarity, with 95% confidence intervals via t-distribution across the ensemble.

```json
// Request
{
  "tree": {
    "op": "+",
    "args": [{ "term": "market" }, { "term": "price" }]
  }
}

// Response
[{
  "term": "labour",
  "pos": ["N", "V"],
  "count": 1337,
  "similarity": 0.354,
  "similarity_ci": [0.312, 0.396]
}]
```

### `POST /parse-describe`

Natural language to validated expression tree. Uses gpt-4o-mini for expression generation, recursive descent parsing for validation, and three-tier term resolution (exact, fuzzy at 0.6, LLM fallback at 0.3). Returns 422 with candidate suggestions if a term cannot be resolved.

```json
// Request
{ "message": "productive labour vs unproductive" }

// Response
{
  "expression": "labour + (productive - unproductive)",
  "terms": ["labour", "productive", "unproductive"],
  "substitutions": []
}
```

**CORS:** `localhost:5173`, `127.0.0.1:5173`, + `PRODUCTION_DOMAIN`.

---

## Lambda resource configuration

| Function | Memory | Timeout | Rationale |
|---|---:|---:|---|
| `scrape` | 256 MB | 120s | I/O-bound HTTP fetch |
| `tokenize` | 512 MB | 120s | spaCy model needs memory headroom |
| `train-kvector` | 1536 MB | 600s | CPU-bound Word2Vec training |
| `align-kvectors` | 256 MB | 120s | NumPy/SciPy on pre-loaded vectors |
| `publish` | 512 MB | 300s | Loads all models + batch writes to DynamoDB |
| `api` | 256 MB | 120s | Reads from DynamoDB + optional Redis |

---

## Getting started

### Prerequisites

- Docker + Docker Compose
- AWS CLI (Lambda, S3, ECR, DynamoDB, Step Functions permissions)
- [`yq`](https://github.com/mikefarah/yq): `deploy_lambdas.sh` uses it to parse `services.yaml`
- [`envsubst`](https://www.gnu.org/software/gettext/): `deploy_step_function.sh` uses it to render the state machine template
- Redis: optional, used by the API for response caching

> **Apple Silicon:** `deploy_lambdas.sh` forces `--platform linux/amd64` via `docker buildx`. Make sure buildx is available in your Docker install.

### Environment variables

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
S3_TEST_DATA_PREFIX=    # e.g. test-data/ (used by integration tests)
PIPELINE_TABLE=         # DynamoDB table for pipeline state
TERM_TABLE=             # DynamoDB table for term vectors
REDIS_URL=              # e.g. redis://localhost:6379 (optional)
REDIS_PREFIX=
PRODUCTION_DOMAIN=      # Frontend URL, for CORS
OPENAI_API_KEY=         # Required for /parse-describe endpoint
```

### Running locally

The pipeline containers have a `local` Dockerfile target that drops the Lambda entrypoint. Source files are volume-mounted so you can edit without rebuilding.

```bash
docker compose up lambda-api    # --> http://localhost:8000
```

> Redis responses are cached indefinitely. If you reprocess a book, flush Redis or you will get stale results.

### Deploying

**Lambdas:** `deploy_lambdas.sh` takes service names as arguments. Builds for `linux/amd64`, runs tests if found, pushes to ECR, then creates or updates the Lambda function. Skips the update if the image and configuration have not changed.

```bash
./infra/deploy_lambdas.sh scrape tokenize train-kvector align-kvectors publish api
```

**Step Function:** `deploy_step_function.sh` renders the template with environment variables and creates or updates the state machine.

```bash
./infra/deploy_step_function.sh
```

To change memory or timeouts, edit `services.yaml` before deploying.

---

## What this pipeline demonstrates

- Designing cloud workflows as independent, resumable, idempotent stages
- Orchestrating parallel fan-out with Step Functions Map state
- Balancing batch ML processing with low-latency API serving via a denormalized DynamoDB layer
- Tuning per-service memory and timeout to match workload characteristics
- Combining deterministic parsing with LLM-assisted natural-language input
- Running a zero-idle-cost serverless architecture suitable for portfolio-scale traffic
