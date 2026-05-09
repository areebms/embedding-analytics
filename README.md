# Embedding Analytics — Backend

Embedding similarity with confidence intervals.

Instead of:
```
similarity("market", "price") = 0.35
```
You get:
```
similarity = 0.35, 95% CI [0.31, 0.40]
```

Trains **ensembles of Word2Vec models** on Project Gutenberg texts, aligns them via **Generalized Procrustes Analysis**, and serves similarity with confidence intervals through a FastAPI endpoint. Serverless, containerized, zero idle cost.

**→ [Live Demo](https://www.embedding-analytics.com)** &nbsp;|&nbsp; **→ [Frontend Repo](https://github.com/areebms/embedding-analytics-frontend)**


---

## Architecture

```mermaid
graph TD
    subgraph Orchestration
        SF[Step Functions]
    end

    subgraph API
        direction LR
        UI[React] <--> FAPI[FastAPI]
    end

    subgraph Storage
        direction LR
        DDB[(DynamoDB)]
        RED[(Redis)]
        S3[(S3)]
    end

    subgraph DP [Per-Book Pipeline]
        direction LR
        SCR[Scrape] --> TOK[Tokenize] --> TRN["Train (×N seeds)"] --> ALN[Align] --> PUB[Publish]
    end

    subgraph CP [Corpus Pipeline]
        direction LR
        BCC[Build Corpus Centroid]
    end

    SF --> DP
    FAPI -.-> RED
    FAPI <--> DDB
    PUB --> DDB
    DP --> S3
    CP --> S3
```

Six containerized Lambda functions on Python 3.13, orchestrated by AWS Step Functions for the per-book pipeline. S3 stores intermediate artifacts during training. DynamoDB stores final term-level vectors (float16) and metadata for fast API reads. Redis caching is optional.

The pipeline has two levels of alignment: **within-book** GPA aligns seed models into a per-book centroid, and **cross-book** GPA aligns book centroids into a shared corpus frame for cross-book comparison.

**→ [Detailed pipeline documentation](docs/pipeline.md)** — per-stage inputs, outputs, configs, and design decisions.

**→ [Alignment math](docs/alignment.md)** — Generalized Procrustes Analysis, convergence, per-term metrics.

---

## API

**`GET /books`** — all corpora with completed pipelines.
```json
[
  {
    "id": 3300,
    "label": "Smith (1776)",
    "author": "Smith, Adam",
    "title": "An Inquiry into...",
    "published_year": 1776
  }
]
```

**`POST /similarity/{book_id}`** — similarity with 95% confidence intervals via t-distribution. Supports dual-term queries (vectors averaged and re-normalized).
```json
// Request
{ "primary_term": "market", "secondary_term": "price" }

// Response
[
  {
    "term": "labour",
    "pos": ["N", "V"],
    "count": 1337,
    "similarity": 0.354,
    "similarity_ci": [0.312, 0.396]
  }
]
```

Tight CI = ensemble agreed. Wide CI = treat with skepticism.

---

## Quick start

```bash
git clone https://github.com/areebms/embedding-analytics.git
cd embedding-analytics
cp .env.example .env   # fill in AWS creds, S3 bucket, DynamoDB tables
docker compose build
```

Run the per-book pipeline:
```bash
docker compose run lambda-scrape python main.py --platform gutenberg --id 3300
docker compose run lambda-tokenize python main.py --platform gutenberg --id 3300
docker compose run lambda-train-kvector python main.py --platform gutenberg --id 3300 --seed 1
docker compose run lambda-train-kvector python main.py --platform gutenberg --id 3300 --seed 2
docker compose run lambda-align-kvectors python create_book_centroid.py
docker compose run lambda-publish python main.py --platform gutenberg --id 3300
```

Build the corpus centroid:
```bash
docker compose run lambda-align-kvectors python create_corpus_centroid.py
```

More seeds = tighter confidence intervals. Every stage is idempotent.

Start the API:
```bash
docker compose up lambda-api    # → http://localhost:8000
```

Deploy:
```bash
./infra/deploy_lambdas.sh scrape tokenize train-kvector align-kvectors publish api
./infra/deploy_step_function.sh
```

**→ [Full setup guide](docs/pipeline.md#getting-started)** -- env vars, prerequisites, Apple Silicon notes, Redis config.

---

## Repo layout

```
embedding-analytics/
├── functions/
│   ├── scrape/                # Gutenberg scraper
│   ├── tokenize/              # spaCy + NLTK lemmatization
│   ├── train-kvector/         # Word2Vec worker (one model per seed)
│   ├── align-kvectors/        # Generalized Procrustes alignment
│   │   ├── create_book_centroid.py    # Within-book GPA
│   │   ├── create_corpus_centroid.py  # Cross-book GPA
│   │   └── procrustes_utils.py        # Shared alignment primitives
│   ├── publish/               # Flatten S3 artifacts → DynamoDB Term Table
│   └── api/                   # FastAPI + Mangum (2 endpoints)
├── shared/
│   ├── aws.py                 # S3, DynamoDB (Pipeline + Term tables), helpers
│   └── commons.py             # CLI arg parsing
├── infra/
│   ├── deploy_lambdas.sh      # Build + deploy Lambda functions
│   ├── deploy_step_function.sh # Deploy Step Function state machine
│   ├── services.yaml          # Lambda config (memory, timeouts)
│   └── step-function.template.json  # State machine definition
├── docker-compose.yml
└── .env.example
```

---

## What's next

- [ ] Pinecone vector database for top-K similarity queries and lower API latency
- [ ] Diachronic similarity chart (semantic drift of a term across books)
- [ ] Cross-book comparison endpoints using the corpus-aligned frame
- [ ] Expose the training pipeline as callable endpoints if traffic warrants it

---

## License

Apache-2.0 — see [LICENSE](./LICENSE)

---

**Areeb Siddiqi** — [LinkedIn](https://www.linkedin.com/in/areeb-siddiqi/) · [GitHub](https://github.com/areebms)
