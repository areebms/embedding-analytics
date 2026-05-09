# Pipeline Documentation

Six containerized Lambda functions, each a single stage. The per-book pipeline is orchestrated by AWS Step Functions. Artifacts flow through S3 during training; the `publish` stage flattens everything into DynamoDB for fast API reads.

| Stage | Input | Output |
|---|---|---|
| `scrape` | Gutenberg book ID | HTML + text + metadata → S3 |
| `tokenize` | Raw text | Lemmatized token CSVs → S3 |
| `train-kvector` | Token lemmas + seed | One trained `.model` → S3 |
| `align-kvectors` | N raw models | Procrustes-aligned models + centroid → S3 |
| `publish` | Aligned models + tokens | Term vectors, POS tags, counts → DynamoDB |
| `api` | HTTP request | Similarity + confidence intervals as JSON |

A separate **corpus pipeline** (`create_corpus_centroid.py`) runs outside the per-book Step Function to build a cross-book alignment frame from 2+ aligned books.

---

## Step Function orchestration

The per-book pipeline runs as an AWS Step Function state machine:

```
scrape → tokenize → train-kvectors (Map, N seeds in parallel) → align-kvectors → publish
```

Input:
```json
{ "index": "gutenberg-3300", "seeds": [1, 2, 3, 4, 5] }
```

The `train-kvectors` step fans out via a Map state, running one `train-kvector` Lambda per seed in parallel. The `seeds` array is carried through earlier steps via Step Function output transforms (the Lambda handlers don't need to know about seeds).

The state machine definition lives at `infra/step-function.template.json` and uses `${AWS_REGION}`, `${AWS_ACCOUNT_ID}`, and `${LAMBDA_PREFIX}` for environment-specific values.

---

## lambda-scrape
`functions/scrape/` — BeautifulSoup, Requests

Pulls books from Project Gutenberg by ID, strips the standard header/footer, and stores HTML, clean text, and bibliographic metadata to S3. Skips re-scraping if the index already has an `s3_text_key` in DynamoDB.

| S3 artifact | Contents |
|---|---|
| `html/{index}.html` | Raw HTML |
| `text/{index}.txt` | Extracted body text |
| `metadata/{index}.json` | Title, author, publication info |

---

## lambda-tokenize
`functions/tokenize/` — spaCy (`en_core_web_sm`), NLTK

Segments text into sentences (NLTK), lemmatizes with spaCy (NER disabled for speed), and outputs three parallel CSVs. Smart-chunks large documents to stay under spaCy's max-length limit without splitting mid-sentence.

Uses aggressive lemmatization — nouns with derivationally related verbs (via WordNet) are collapsed to the verb form (e.g. "production" → "produce") to reduce semantic fragmentation. A curated `ignored_nouns.txt` list prevents over-lemmatization of nouns that shouldn't collapse. Also normalizes British/American spelling (e.g. "labor" → "labour").

Skips re-tokenizing if all three output keys already exist.

| S3 artifact | Contents |
|---|---|
| `token_texts/{index}.csv` | Original tokens, one sentence per row |
| `token_lemmas/{index}.csv` | Lowercased lemmas — what Word2Vec trains on |
| `token_tags/{index}.csv` | POS tags |

---

## lambda-train-kvector
`functions/train-kvector/` -- Gensim

Trains one Word2Vec model with an explicit seed for reproducibility. Filters lemmas to alphabetic tokens longer than 3 characters. Each invocation produces one model file named `{seed}-{timestamp}-{randint}.model` under `kvectors/{index}/collected/` in S3.

**Word2Vec config:**
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

## lambda-align-kvectors
`functions/align-kvectors/` — NumPy, SciPy

Implements Generalized Procrustes Analysis at two levels.

**Per-book alignment** (`create_book_centroid.py`): Aligns all seed models into a shared vector space, builds a centroid with per-term stability metrics, and optionally rotates to the corpus frame if a corpus centroid exists.

**Cross-book alignment** (`create_corpus_centroid.py`): Aligns book centroids into a shared corpus frame for cross-book comparison. Runs as a CLI command outside the Step Function, requires 2+ aligned books. Uses unit-normalized vectors and uniform weights. Terms are filtered to those appearing in at least 2 books.

Shared alignment primitives (Procrustes rotation, S3 I/O helpers) live in `procrustes_utils.py`.

**→ [Full alignment math](alignment.md)** — Procrustes rotation, convergence, disparity metrics, R².

Output lands at `kvectors/{index}/aligned/` (rotated models) and `kvectors/{index}/centroid.model`.

---

## lambda-publish
`functions/publish/` — Gensim, NumPy

Flattens S3 artifacts into DynamoDB's Term Table for fast API reads. For each term present in the centroid, POS tag set, and aligned model stack, writes a single row containing: centroid vector (float16), per-seed aligned vectors (float16), token occurrence positions (`ilocs`), POS tags, word count, disparity, variance, and R². Also backfills author/title in the Pipeline Table from Gutenberg metadata.

---

## lambda-api
`functions/api/` — FastAPI, Mangum, fastapi-cache, Redis

A thin read layer over the DynamoDB Term Table. [Mangum](https://github.com/jordaneremieff/mangum) makes FastAPI work inside a Lambda Function URL. Responses are cached in Redis via `fastapi-cache` with no expiry. Redis is optional — the API runs without it (caching is simply disabled).

**`GET /books`**  
All corpora that have completed the full pipeline.
```json
[{ "id": 3300, "label": "Smith (1776)", "author": "Smith, Adam", "title": "An Inquiry into...", "published_year": 1776 }]
```

**`POST /similarity/{book_id}`**  
Accepts `{ "primary_term": "market", "secondary_term": "price" }`. When a secondary term is provided, the two term vectors are averaged and re-normalized before computing similarities. Returns every term in the corpus ranked by cosine similarity, with 95% confidence intervals computed via t-distribution across the ensemble.
```json
[{ "term": "price", "pos": ["N"], "count": 1337, "similarity": 0.354, "similarity_ci": [0.312, 0.396] }]
```

The confidence interval width reflects ensemble agreement — tight intervals mean the relationship held up consistently across training runs.

Terms where the only POS tag is adverb (`R`) are excluded from results.

**CORS:** `localhost:5173`, `127.0.0.1:5173`, + whatever's in `PRODUCTION_DOMAIN`.

---

## Lambda Resource Config

| Function | Memory | Timeout | Why |
|---|---|---|---|
| scrape | 256 MB | 120s | I/O bound, no heavy compute |
| tokenize | 512 MB | 120s | spaCy needs headroom |
| train-kvector | 1536 MB | 600s | CPU-bound Word2Vec training |
| align-kvectors | 256 MB | 120s | NumPy/SciPy on pre-loaded vectors |
| publish | 512 MB | 300s | Loads all models + writes to DynamoDB |
| api | 256 MB | 120s | Reads from DynamoDB/Redis |

---

## Getting started

### Prerequisites
- Docker + Docker Compose
- AWS CLI (Lambda, S3, ECR, DynamoDB, Step Functions permissions)
- [`yq`](https://github.com/mikefarah/yq) -- `deploy_lambdas.sh` uses it to parse `services.yaml`
- [`envsubst`](https://www.gnu.org/software/gettext/) -- `deploy_step_function.sh` uses it to render the state machine template
- Redis -- optional, used by the API for response caching

> **Apple Silicon:** `deploy_lambdas.sh` forces `--platform linux/amd64` via `docker buildx`. Make sure buildx is available in your Docker install.

### Environment variables

```
AWS_REGION=
AWS_ACCOUNT_ID=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_ECR_REPO=
LAMBDA_ROLE_ARN=
LAMBDA_PREFIX=
STEP_FUNCTION_ROLE_ARN=
S3_BUCKET=
PIPELINE_TABLE=        # DynamoDB table for pipeline state
TERM_TABLE=            # DynamoDB table for term vectors
REDIS_URL=             # e.g. redis://localhost:6379 (optional)
REDIS_PREFIX=
PRODUCTION_DOMAIN=     # Your frontend URL, for CORS
```

### Running locally

The pipeline containers have a `local` Dockerfile target that drops the Lambda entrypoint. Source files are volume-mounted, so you can edit without rebuilding.

```bash
docker compose up lambda-api    # → http://localhost:8000
```

> Redis responses are cached indefinitely. If you reprocess a book, flush Redis or you'll get stale results.

### Deploying

**Lambdas:** `deploy_lambdas.sh` takes service names as arguments. It builds for `linux/amd64`, runs tests if found, pushes to ECR, then creates or updates the Lambda function. Skips the update if the image and configuration haven't changed.

```bash
./infra/deploy_lambdas.sh scrape tokenize train-kvector align-kvectors publish api
```

**Step Function:** `deploy_step_function.sh` renders the template with environment variables and creates or updates the state machine.

```bash
./infra/deploy_step_function.sh
```

To change memory or timeouts, edit `services.yaml` before deploying.
