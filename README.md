# Embedding Analytics — Backend

Most NLP similarity tools give you a single number and call it a day. This one gives you a number *and* tells you how much to trust it.

The backend trains **ensembles of Word2Vec models** on Project Gutenberg texts, aligns them via **Orthogonal Procrustes analysis**, and exposes the results through a FastAPI endpoint — all serverless, all containerized, zero idle cost.

**→ [Live Demo](https://www.embedding-analytics.com)** &nbsp;|&nbsp; **→ [Frontend Repo](https://github.com/areebms/embedding-analytics-frontend)**

---

## Why ensembles?

Word2Vec is stochastic. Two models trained on identical text will learn slightly different vector spaces. That's usually treated as a nuisance. Here it's the whole point.

Train enough independent models on the same corpus, align their vector spaces, and the variance becomes meaningful:

| Signal | Interpretation |
|---|---|
| Low variance across runs | Stable, trustworthy semantic relationship |
| High variance across runs | The model is uncertain — treat with skepticism |
| Systematic drift between corpora | Real semantic shift worth investigating |

You get similarity scores *with built-in confidence* — not just "these words are related" but "these words are reliably related."

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   scrape     │───▶│   tokenize   │───▶│train-kvectors│───▶│aggregate-data│
│  (Lambda)    │    │   (Lambda)   │    │   (Lambda)   │    │   (Lambda)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                    │                    │                    │
       └────────────────────┴────────────────────┴────────────────────┘
                                    │
                            [S3 Artifact Storage]
                            [DynamoDB Pipeline State]
                                    │
                            ┌───────▼────────┐
                            │   api (Lambda) │
                            └───────┬────────┘
                                    │
                            [React Frontend]
```

Six Docker containers, each a single Lambda function, all on Python 3.13. DynamoDB tracks pipeline state so every stage is idempotent. Artifacts flow through S3.

| Stage | Input | Output |
|---|---|---|
| `scrape` | Gutenberg book ID | HTML + text + metadata → S3 |
| `tokenize` | Raw text | Lemmatized token CSVs → S3 |
| `train-kvectors` | Corpus index | Fires N parallel `train-kvector` invocations |
| `train-kvector` | Token lemmas | One trained `.model` → S3 |
| `aggregate-data` | N raw models | Aligned models + centroid + variance CSV → S3 |
| `api` | HTTP request | Similarity + coherence scores as JSON |

---

## The Pipeline

### lambda-scrape
`functions/scrape/` — BeautifulSoup, Requests

Pulls books from Project Gutenberg by ID, strips the standard header/footer, and stores HTML, clean text, and bibliographic metadata to S3. Skips re-scraping if the index already has an `s3_text_key` in DynamoDB.

| S3 artifact | Contents |
|---|---|
| `html/{index}.html` | Raw HTML |
| `text/{index}.txt` | Extracted body text |
| `metadata/{index}.json` | Title, author, publication info |

---

### lambda-tokenize
`functions/tokenize/` — spaCy (`en_core_web_sm`), NLTK

Segments text into sentences (NLTK), lemmatizes with spaCy (parser and NER disabled for speed), and outputs three parallel CSVs. Smart-chunks large documents to stay under spaCy's max-length limit without splitting mid-sentence. Skips re-tokenizing if all three output keys already exist.

| S3 artifact | Contents |
|---|---|
| `token_texts/{index}.csv` | Original tokens, one sentence per row |
| `token_lemmas/{index}.csv` | Lowercased lemmas — what Word2Vec trains on |
| `token_tags/{index}.csv` | POS tags |

---

### lambda-train-kvectors (Orchestrator)
`functions/train-kvectors/`

Fans out to N parallel `train-kvector` workers via Lambda's async `InvocationType='Event'` — fire and forget, no blocking. Returns `{ initiated, attempted }`. Ensemble size is set by the `KVECTORS_TRAINED` Lambda env var (defaults to `2`).

---

### lambda-train-kvector (Worker)
`functions/train-kvector/` — Gensim

Trains one Word2Vec model. Filters lemmas to alphabetic tokens longer than 3 chars. Calculates worker count from Lambda's allocated memory (`vcpu = memory_mb / 1769.0`, capped at 6). Saves with a `{timestamp}-{randint}.model` name to avoid S3 collisions across parallel invocations.

**Word2Vec config:**
| Parameter | Value |
|---|---|
| Vector size | 200 |
| Window | 10 tokens |
| Min count | 2 |
| Algorithm | Skip-gram (`sg=1`) |
| Training | Hierarchical softmax (`hs=1`) |
| Negative sampling | Off |
| Epochs | 30 |

> This function runs at 6144 MB (~3.5 vCPUs). Word2Vec training is CPU-bound, and Lambda's vCPU allocation scales linearly with memory — so more memory means faster training, not just more RAM.

---

### lambda-aggregate-data
`functions/aggregate-data/` — NumPy, SciPy, scikit-learn

The heavy math step. Loads all raw models, aligns them, computes centroids, calculates variance, and writes everything back to S3. Also backfills author/title in DynamoDB from Gutenberg metadata.

**Procrustes alignment:** For each model, finds orthogonal rotation `R` minimizing `||A·R - B||²`, solved via SVD (`R = U·Vᵀ` where `U·Σ·Vᵀ = BᵀA`). Applied in-place with norms recomputed. Preserves cosine similarity while putting all models in the same coordinate frame.

**Variance metrics per term:**
| Field | What it measures |
|---|---|
| `overall` | Mean Euclidean distance from centroid — total spread |
| `semantic` | Mean cosine distance from centroid — directional variance |
| `norm` | Mean absolute deviation in vector magnitude |

Output lands at `keyed_vector_group_data/{index}/` — aligned models, `centroid.model`, and `term_stability.csv`.

---

### lambda-api
`functions/api/` — FastAPI, Mangum, Redis

A thin read layer over the S3 artifacts. [Mangum](https://github.com/jordaneremieff/mangum) makes FastAPI work inside a Lambda Function URL. Responses are cached in Redis with no expiry (`expire=None`) — cold S3 loads happen exactly once per corpus/term.

**`GET /books`**  
All corpora that have completed the full pipeline.
```json
[{ "id": 3300, "label": "Smith (1776)", "author": "Smith, Adam", "title": "An Inquiry into the Nature and Causes of the Wealth of Nations" }]
```

**`GET /similarity/{book_id}/{primary_term}`**  
Every term in the corpus, ranked by cosine similarity to `primary_term`, with coherence and frequency.
```json
[{ "term": "price", "similarity": 0.354, "coherence": 0.938, "count": 1337 }]
```

`coherence = 1 - semantic_variance`. A score of 0.938 means the model is very confident that *price* reliably associates with *market* — it held up consistently across training runs.

**CORS:** `localhost:5173` + whatever's in `PRODUCTION_DOMAIN`.

---

## Lambda Resource Config

| Function | Memory | Timeout | Why |
|---|---|---|---|
| scrape | 256 MB | 120s | I/O bound, no heavy compute |
| tokenize | 512 MB | 120s | spaCy needs a bit more headroom |
| train-kvector | 6144 MB | 600s | CPU-bound — RAM = vCPUs on Lambda |
| train-kvectors | 256 MB | 120s | Just fires async invocations |
| aggregate-data | 256 MB | 120s | NumPy ops on pre-loaded vectors |
| api | 256 MB | 120s | Reads from Redis/S3 |

---

## Getting Started

### Prerequisites
- Docker + Docker Compose
- AWS CLI (Lambda, S3, ECR, DynamoDB permissions)
- [`yq`](https://github.com/mikefarah/yq) — `push_to_ecr.sh` uses it to parse `services.yaml`
- Redis — required by the API at startup

> **Apple Silicon:** `push_to_ecr.sh` forces `--platform linux/amd64` via `docker buildx`. Make sure buildx is available in your Docker install.

### Setup

```bash
git clone https://github.com/areebms/embedding-analytics.git
cd embedding-analytics
cp .env.example .env
```

```
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_URI_PREFIX=        # e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com
AWS_ECR_REPO=
LAMBDA_ROLE_ARN=
S3_BUCKET=
PIPELINE_TABLE=
REDIS_URL=             # e.g. redis://localhost:6379
REDIS_PREFIX=
PRODUCTION_DOMAIN=     # Your frontend URL — for CORS
```

```bash
docker-compose build
```

### Run the pipeline

`--platform-name` must be `gutenberg` (the only supported platform). Indexes follow `gutenberg-{id}`.

```bash
# Book 3300 = Wealth of Nations
docker-compose run lambda-scrape python main.py --platform-name gutenberg --platform-id 3300
docker-compose run lambda-tokenize python main.py --platform-name gutenberg --platform-id 3300
docker-compose run lambda-train-kvectors python main.py --platform-name gutenberg --platform-id 3300
docker-compose run lambda-train-kvector python main.py --platform-name gutenberg --platform-id 3300
docker-compose run lambda-aggregate-data python main.py --platform-name gutenberg --platform-id 3300
```

Every stage checks DynamoDB first and skips work that's already done.

### Run the API locally

The API container has a `local` Dockerfile target that runs uvicorn with hot reload. `app.py` and `main.py` are volume-mounted — edit without rebuilding.

```bash
docker-compose up lambda-api
# → http://localhost:8000
```

> Redis responses are cached indefinitely. If you reprocess a book, flush Redis or you'll get stale results.

### Deploy

`push_to_ecr.sh` takes service names as arguments. It builds for `linux/amd64`, runs a smoke test against book 60411 with your `.env` credentials, pushes to ECR, then creates or updates the Lambda function automatically.

```bash
cd infra

# One or many at a time
./push_to_ecr.sh scrape tokenize train-kvector train-kvectors aggregate-data api
```

To change ensemble size, set `KVECTORS_TRAINED` on the `train-kvectors` Lambda function. To change memory or timeouts, edit `services.yaml` before deploying.

---

## Repo Layout

```
embedding-analytics/
├── functions/
│   ├── scrape/             # Gutenberg scraper
│   ├── tokenize/           # spaCy lemmatization
│   ├── train-kvector/      # Word2Vec worker
│   ├── train-kvectors/     # Ensemble orchestrator
│   ├── aggregate-data/     # Procrustes + variance
│   └── api/                # FastAPI (2 endpoints)
├── shared/
│   ├── aws.py              # S3, DynamoDB, Lambda helpers
│   └── commons.py          # CLI arg parsing
├── infra/
│   ├── push_to_ecr.sh      # Build + deploy script
│   └── services.yaml       # Lambda config
├── docker-compose.yml
└── .env.example
```

---

## What's next

- [ ] Stream sentence iteration — avoid loading full corpora into RAM
- [ ] Step Functions orchestration for multi-corpus pipelines
- [ ] Transformer ensemble support

---

## License

Apache-2.0 — see [LICENSE](./LICENSE)

---

**Areeb Siddiqi** — [LinkedIn](https://www.linkedin.com/in/areeb-siddiqi/) · [GitHub](https://github.com/areebms)
