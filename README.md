# Embedding Analytics: Backend

Six containerized Python microservices on AWS Lambda, orchestrated by Step Functions, serving a FastAPI API backed by DynamoDB and Redis.

The system trains Word2Vec model ensembles, aligns them via Generalized Procrustes Analysis, and scores queries across the ensemble rather than a single model, reporting every value with a 95% confidence interval. Natural-language queries are translated into validated expression trees through an LLM pipeline with deterministic guardrails.

**Live demo:** https://www.embedding-analytics.com
**Frontend repo:** https://github.com/areebms/embedding-analytics-frontend

---

## Engineering highlights

- **Microservices pipeline:** Six independent, containerized Lambda stages for scraping, tokenization, model training, vector alignment, publishing, and API serving
- **Fan-out orchestration:** Step Functions Map state trains N seeded models in parallel, converging into a single alignment stage
- **Dual serving layer:** S3 holds intermediate artifacts during training; a publish stage fans results out to DynamoDB (per-seed ensemble vectors, term metadata) and Pinecone (indexed centroid vectors), pruning deprecated terms on republish
- **LLM integration with guardrails:** OpenAI-powered natural-language parsing bounded by recursive descent validation, tiered vocabulary resolution, and structured error recovery
- **Ensemble confidence scoring:** Queries are evaluated across every aligned model independently, producing 95% confidence intervals instead of single-point similarity scores
- **Test-gated deploys:** Per-service pytest suites run inside a dedicated Docker test stage before any image is pushed; the API suite enforces an 85% coverage floor
- **Zero idle cost:** Fully serverless, containerized, no always-on infrastructure

The domain is classical political economy, but the architecture maps directly to production AI infrastructure: data ingestion, model generation, vector indexing and serving, LLM parsing, validation, caching, and low-cost cloud deployment.

---

## What the product does

A standard Word2Vec interface returns nearest neighbours for a single term. This system supports compositional queries with vector arithmetic.

A single term like `capital` can return results from multiple meanings. Adding a second term narrows the context:

```text
capital + profit
```

This pulls "capital" toward its economic sense and away from the geographical one.

Subtraction creates contrast directions. The query `labour + (productive - unproductive)` starts with labour and tilts it toward the productive side of the productive/unproductive distinction:

```text
labour + (productive - unproductive)
```

The backend evaluates that expression against each aligned model in the ensemble and returns ranked terms with 95% confidence intervals. Results reflect contextual proximity, not dictionary meaning: two terms score highly because the authors discuss them in similar contexts, not because they are synonyms. Tight intervals indicate the relationship was consistent across training runs; wide intervals flag instability.

Users can also type plain English:

```text
productive vs unproductive labour
```

The backend converts that into a validated expression tree, resolves every term against the corpus vocabulary, and returns the structured expression for the frontend to render and edit.

More detail: [`docs/guide.md`](docs/guide.md)

---

## Architecture

```mermaid
graph TD
    subgraph Orchestration
        SF[Step Functions]
    end

    subgraph API
        direction LR
        UI[React Frontend] <--> FAPI[FastAPI + Mangum]
    end

    subgraph Storage
        direction LR
        DDB[(DynamoDB)]
        PC[(Pinecone)]
        RED[(Redis Cache)]
        S3[(S3 Artifacts)]
    end

    subgraph Pipeline [Per-Book Pipeline]
        direction LR
        SCR[Scrape] --> TOK[Tokenize] --> TRN["Train Word2Vec x N seeds"] --> ALN[Align Models] --> PUB[Publish Vectors]
    end

    subgraph Corpus [Corpus Alignment]
        direction LR
        BCC[Build Corpus Centroid]
    end

    SF --> Pipeline
    Pipeline --> S3
    PUB --> DDB
    PUB --> PC
    FAPI <--> DDB
    FAPI -. optional .-> RED
    Corpus --> S3
```

---

## Tech stack

| Area | Tools |
|---|---|
| API | Python 3.13, FastAPI, Mangum, Pydantic |
| AI/ML | Gensim Word2Vec, NumPy, SciPy, OpenAI API |
| Vector index | Pinecone (centroid index, written by `publish`) |
| Cloud | AWS Lambda, Step Functions, S3, DynamoDB, ECR |
| Caching | Redis, fastapi-cache |
| NLP/Data | spaCy, NLTK, WordNet, BeautifulSoup |
| Testing | pytest, coverage gating, Docker test stages |
| Infrastructure | Docker, Docker Compose, Bash |

---

## Pipeline

Six containerized Lambda services, each a single stage with its own memory and timeout configuration:

| Stage | What it does | Output |
|---|---|---|
| `scrape` | Pulls source text and metadata from Project Gutenberg | HTML, clean text, metadata in S3 |
| `tokenize` | Segments, lemmatizes, normalizes spelling, tags POS | Token CSVs in S3 |
| `train-kvector` | Trains one seeded Word2Vec model | `.model` artifact in S3 |
| `align-kvectors` | Aligns seed models into a shared vector space via GPA | Rotated models + centroid in S3 |
| `publish` | Fans out to the serving layer | Per-seed vectors + metadata in DynamoDB, centroid vectors in Pinecone |
| `api` | Serves books, terms, and NL parse endpoints | JSON responses via Lambda Function URL |

Step Function flow:

```text
scrape --> tokenize --> train-kvector Map(N seeds) --> align-kvectors --> publish
```

Every stage is idempotent: failed or repeated runs skip already-completed work and resume safely. Republishing a book also removes terms that no longer exist in the new model from Pinecone and the corpus vocabulary.

More detail: [`docs/pipeline.md`](docs/pipeline.md)

---

## Storage model

| Store | Role |
|---|---|
| `PipelineTable` (DynamoDB) | Per-book pipeline state, S3 artifact keys, author/title/year metadata |
| `BookTermTable` (DynamoDB) | One row per (term, book): per-seed aligned vectors (float16), alignment stats, token positions, POS tags, counts |
| `CorpusTermTable` (DynamoDB) | Single-partition table mapping each term to the books that contain it, for fast cross-book vocabulary queries |
| Pinecone index | One centroid vector per (book, term), keyed `{book_id}::{term}`, with metadata for book and POS filtering at query time |
| S3 | Intermediate training artifacts: raw text, token CSVs, seed models, aligned models, centroids |

---

## API endpoints

### `GET /books`

Returns every corpus that has completed the full pipeline.

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

### `GET /terms`

Returns the cross-book vocabulary. Terms must appear in at least two books. Adverb-only terms are excluded.

```json
[
  {
    "term": "labour",
    "books": ["gutenberg-3300", "gutenberg-846"]
  }
]
```

### `POST /parse-describe`

Converts plain English into a validated vector expression. Uses an LLM for expression generation, then validates via recursive descent parsing and resolves every term against the DynamoDB vocabulary through a three-tier fallback (exact match, fuzzy match, LLM-assisted resolution). Returns 422 with candidate suggestions when a term cannot be resolved.

Request:

```json
{ "message": "productive vs unproductive labour" }
```

Response:

```json
{
  "expression": "labour + (productive - unproductive)",
  "terms": ["labour", "productive", "unproductive"],
  "substitutions": []
}
```

More detail: [`docs/describe.md`](docs/describe.md)

---

## Alignment and reliability

A single Word2Vec model trained on a small corpus is unreliable. This backend trains multiple seeded models, aligns them with Generalized Procrustes Analysis, and evaluates every query across the full ensemble to produce 95% confidence intervals.

Within-book GPA aligns the seed models for each book into a per-book centroid with per-term stability metrics.

The result is a reliability signal that a standard embedding tool does not provide: tight CI means the relationship was stable across training runs, wide CI means treat with skepticism.

More detail: [`docs/alignment.md`](docs/alignment.md)

---

## Testing

- Per-service pytest suites cover the API routers and services, publish, alignment, and shared table clients
- Deploys are test-gated: `deploy_lambdas.sh` builds a dedicated `test` stage of each multi-stage Dockerfile and runs the suite against the production dependency set; a failure aborts before any image is built or pushed
- The API suite enforces an 85% line-coverage floor (`--cov-fail-under=85`)
- Services can define a `smoke_cmd` in `services.yaml` that runs against the production image before it is pushed to ECR

---

## Running locally

```bash
git clone https://github.com/areebms/embedding-analytics.git
cd embedding-analytics
cp .env.example .env
docker compose build
docker compose up lambda-api    # --> http://localhost:8000
```

Run the per-book pipeline:

```bash
docker compose run lambda-scrape python main.py --platform gutenberg --id 3300
docker compose run lambda-tokenize python main.py --platform gutenberg --id 3300
docker compose run lambda-train-kvector python main.py --platform gutenberg --id 3300 --seed 1
docker compose run lambda-train-kvector python main.py --platform gutenberg --id 3300 --seed 2
docker compose run lambda-align-kvectors python create_book_centroid.py
docker compose run lambda-publish python publish_to_api.py
```

`create_book_centroid.py` and `publish_to_api.py` iterate over every book in the pipeline table; both are idempotent, and publish prunes terms that no longer exist after retraining.

Per-service VS Code devcontainers for the API, alignment, and publish services live in `.devcontainer/`.

---

## Deployment

```bash
./infra/deploy_lambdas.sh scrape tokenize train-kvector align-kvectors publish api
./infra/deploy_step_function.sh
```

Deployment scripts build Docker images for `linux/amd64`, run the service's test suite inside the image, push to ECR, and create or update Lambda services only when the image or configuration has changed.

---

## Repo layout

```text
embedding-analytics/
├── functions/
│   ├── scrape/
│   ├── tokenize/
│   ├── train-kvector/
│   ├── align-kvectors/
│   │   ├── src/
│   │   │   ├── create_book_centroid.py
│   │   │   └── procrustes_utils.py
│   │   └── tests/
│   ├── publish/
│   │   ├── src/
│   │   │   ├── publish_to_api.py
│   │   │   └── publish_utils.py
│   │   └── tests/
│   └── api/
│       ├── src/
│       │   ├── main.py             # FastAPI app, CORS, request logging, Mangum
│       │   └── app/
│       │       ├── core/           # shared dependencies (cache)
│       │       ├── list/           # /books, /terms
│       │       └── search/         # /parse-describe
│       └── tests/                  # pytest, 85% coverage gate
├── shared/
│   ├── tables/                     # storage clients
│   │   ├── base.py
│   │   ├── pipeline.py             # PipelineTable (DynamoDB)
│   │   ├── book_terms.py           # BookTermTable (DynamoDB)
│   │   ├── corpus_terms.py         # CorpusTermTable (DynamoDB)
│   │   └── vectors.py              # Pinecone index client
│   ├── s3.py
│   ├── session.py
│   ├── schemas.py
│   ├── lambda_event.py
│   └── commons.py
├── infra/
│   ├── deploy_lambdas.sh
│   ├── deploy_step_function.sh
│   ├── services.yaml
│   └── step-function.template.json
├── docs/
│   ├── pipeline.md
│   ├── alignment.md
│   ├── describe.md
│   └── guide.md
├── .devcontainer/                  # per-service devcontainers (api, align, publish)
├── docker-compose.yml
└── .env.example
```

---

## What this demonstrates

- Designing and deploying a multi-stage serverless data pipeline on AWS (Lambda, Step Functions, S3, DynamoDB)
- Python API development with FastAPI, Pydantic schemas, recursive evaluation, and structured error handling
- LLM integration with deterministic validation boundaries and tiered fallback resolution
- Ensemble-based confidence scoring as a production reliability pattern for vector search
- Containerized microservice deployment with per-service resource tuning and test-gated releases
- Caching, idempotency, and zero-idle-cost serverless architecture

---

## What's next

- [ ] Diachronic similarity chart (semantic drift of a term across books)
- [ ] Expose the training pipeline as callable endpoints if traffic warrants it

---

## License

Apache-2.0

---

**Areeb Siddiqi** · [LinkedIn](https://www.linkedin.com/in/areeb-siddiqi/) · [GitHub](https://github.com/areebms)
