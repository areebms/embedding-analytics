# Pipeline Documentation

Six containerized Lambda functions, each a single stage. Artifacts flow through S3 during training; the `publish` stage flattens everything into DynamoDB for fast API reads.

| Stage | Input | Output |
|---|---|---|
| `scrape` | Gutenberg book ID | HTML + text + metadata → S3 |
| `tokenize` | Raw text | Lemmatized token CSVs → S3 |
| `train-kvector` | Token lemmas + seed | One trained `.model` → S3 |
| `align-kvectors` | N raw models | Procrustes-aligned models + centroid → S3 |
| `publish` | Aligned models + tokens | Term vectors, POS tags, counts → DynamoDB |
| `api` | HTTP request | Similarity + confidence intervals as JSON |

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
`functions/train-kvector/` — Gensim

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

Implements Generalized Procrustes Analysis to align all trained models into a shared vector space and compute per-term stability metrics.

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
- AWS CLI (Lambda, S3, ECR, DynamoDB permissions)
- [`yq`](https://github.com/mikefarah/yq) — `push_to_ecr.sh` uses it to parse `services.yaml`
- Redis — optional, used by the API for response caching

> **Apple Silicon:** `push_to_ecr.sh` forces `--platform linux/amd64` via `docker buildx`. Make sure buildx is available in your Docker install.

### Environment variables

```
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_URI_PREFIX=        # e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com
AWS_ECR_REPO=
LAMBDA_ROLE_ARN=
LAMBDA_PREFIX=
S3_BUCKET=
PIPELINE_TABLE=        # DynamoDB table for pipeline state
TERM_TABLE=            # DynamoDB table for term vectors
REDIS_URL=             # e.g. redis://localhost:6379 (optional)
REDIS_PREFIX=
PRODUCTION_DOMAIN=     # Your frontend URL — for CORS
```

### Running locally

The API container has a `local` Dockerfile target that runs uvicorn with hot reload. `app.py` and `main.py` are volume-mounted — edit without rebuilding.

```bash
docker-compose up lambda-api
# → http://localhost:8000
```

> Redis responses are cached indefinitely. If you reprocess a book, flush Redis or you'll get stale results.

### Deploying

`push_to_ecr.sh` takes service names as arguments. It builds for `linux/amd64`, runs a smoke test, pushes to ECR, then creates or updates the Lambda function automatically.

```bash
cd infra
./push_to_ecr.sh scrape tokenize train-kvector align-kvectors publish api
```

To change memory or timeouts, edit `services.yaml` before deploying.
