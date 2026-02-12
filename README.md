# Embedding Analytics Platform

A full-stack platform for **cross-corpus semantic analysis** using **ensembles of Word2Vec embeddings**.  
Captures **model variance** from stochastic training, aligns independently trained vector spaces, and surfaces **semantic similarity + drift** through an API and dashboard.

**Live Demo:** https://www.embedding-analytics.com/

---

## Why This Matters

Embedding training is inherently stochastic (random initialization + sampling/optimization). Even on identical corpora, independent training runs produce slightly different vector spaces.

**This project treats that variability as signal rather than noise:**
- Relationships that persist across runs → **reliable semantic associations**
- Relationships that fluctuate → **semantic uncertainty / instability**
- Systematic drift between corpora → **meaningful semantic shifts**

This approach provides **confidence metrics** for embedding-based analysis, going beyond single-model point estimates.

---

## Architecture Overview

### Serverless Microservices on AWS Lambda

The platform runs as a set of containerized Lambda functions, each handling a specific stage of the pipeline. All containers are built with Docker and deployed to AWS ECR.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   scrape     │───▶│   tokenize   │───▶│train-kvectors│───▶│aggregate-data│
│  (Lambda)    │    │   (Lambda)   │    │   (Lambda)   │    │   (Lambda)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                    │                    │                    │
       └────────────────────┴────────────────────┴────────────────────┘
                                    │
                            [S3 Artifact Storage]
                                    │
                            ┌───────▼────────┐
                            │   api (Lambda) │
                            └───────┬────────┘
                                    │
                            [React Frontend]
```

---

## End-to-End Pipeline (by Container)

### 1. **lambda-scrape** 
**Container:** `functions/scrape/`  
**Technology:** Python, BeautifulSoup  
**AWS Services:** Lambda, S3, DynamoDB

**Responsibilities:**
- Fetch HTML from target URLs (news articles, documents, etc.)
- Extract raw text content from HTML
- Extract metadata (title, author, publication date)
- Store artifacts in S3:
  - `html/{index}.html` - Raw HTML
  - `text/{index}.txt` - Extracted text
  - `metadata/{index}.json` - Document metadata
- Update pipeline tracking table (DynamoDB)

**Input:** Document URL or identifier  
**Output:** Raw text + metadata in S3

**Key Functions:**
```python
get_html(url)        # Fetch HTML from source
get_text(html)       # Extract clean text
get_metadata(url)    # Extract document metadata
```

---

### 2. **lambda-tokenize**
**Container:** `functions/tokenize/`  
**Technology:** Python, spaCy, NLTK  
**AWS Services:** Lambda, S3, DynamoDB

**Responsibilities:**
- Load scraped text from S3
- Perform sentence segmentation using NLTK
- Chunk large documents to handle spaCy's max length limits
- Tokenize sentences using spaCy pipeline (lemmatization, POS tagging)
- Generate three parallel CSV outputs:
  - Token texts (original forms)
  - Token lemmas (normalized forms for embedding training)
  - Token POS tags (for potential filtering)
- Upload tokenized data to S3

**Input:** `text/{index}.txt` from S3  
**Output:** CSV files in S3:
- `token_texts/{index}.csv`
- `token_lemmas/{index}.csv`
- `token_tags/{index}.csv`

**Processing Details:**
- Uses spaCy's `en_core_web_sm` model with sentencizer
- Disables parser/NER for performance (only need lemmas + tags)
- Filters tokens to keep only alphabetic words > 3 characters
- Smart chunking preserves sentence boundaries

---

### 3. **lambda-train-kvectors** (Orchestrator)
**Container:** `functions/train-kvectors/`  
**Technology:** Python, AWS Lambda SDK  
**AWS Services:** Lambda (async invocation)

**Responsibilities:**
- **Orchestrates ensemble training** by invoking multiple `lambda-train-kvector` instances
- Spawns N parallel training jobs (configurable via `KVECTORS_TRAINED` env var)
- Each invocation trains an independent Word2Vec model with different random seed
- Uses Lambda's async invocation (`InvocationType='Event'`) for parallel execution

**Input:** Index pointing to tokenized data  
**Output:** N async Lambda invocations → N independent Word2Vec models

**Why separate orchestrator?**
- Allows configurable ensemble size without code changes
- Enables parallel training at scale (Lambda auto-scales)
- Decouples orchestration logic from training logic

---

### 4. **lambda-train-kvector** (Worker)
**Container:** `functions/train-kvector/`  
**Technology:** Python, Gensim (Word2Vec)  
**AWS Services:** Lambda, S3

**Responsibilities:**
- Load tokenized lemmas from S3
- Filter tokens (alphabetic only, min 3 characters)
- Train a single Word2Vec model on the corpus
- Calculate optimal worker count based on Lambda memory allocation:
  ```python
  vcpu = memory_mb / 1769.0  # AWS proportional vCPU allocation
  workers = min(available_cpu, calculated_workers)
  ```
- Save trained model with timestamp + random ID to avoid collisions
- Upload model to S3: `word_vectors/{index}/{timestamp}-{rand}.model`

**Input:** `token_lemmas/{index}.csv` from S3  
**Output:** Single Word2Vec model in S3

**Word2Vec Configuration:**
- **Vector size:** 200 dimensions
- **Window:** 10 tokens (context size)
- **Min count:** 2 (ignore rare words)
- **Algorithm:** Skip-gram (`sg=1`)
- **Training method:** Hierarchical softmax (`hs=1`)
- **Epochs:** 30
- **Sampling/Negative:** Disabled (pure hierarchical softmax)

**Key Insight:** Each invocation uses a different random seed (via timestamp + randint), producing independent stochastic realizations of the embedding space.

---

### 5. **lambda-aggregate-data**
**Container:** `functions/aggregate-data/`  
**Technology:** Python, NumPy, SciPy, scikit-learn  
**AWS Services:** Lambda, S3

**Responsibilities:**
- Load all trained Word2Vec models for a given index from S3
- **Align embedding spaces** using Orthogonal Procrustes analysis:
  - First model becomes reference frame
  - Compute optimal rotation matrix for each subsequent model
  - Apply rotation to align all models to common space
- **Compute centroid embeddings** (mean vector across aligned ensemble)
- **Calculate variance metrics** for each term:
  - `overall`: Mean Euclidean distance from centroid (total dispersion)
  - `semantic`: Mean cosine distance from centroid (directional variance)
  - `norm`: Variance in vector magnitudes (length instability)
- Upload artifacts to S3:
  - Aligned models: `{index}/aligned_models/{i}.model`
  - Centroid model: `{index}/centroid.model`
  - Term stability data: `{index}/term_stability.csv`

**Input:** Ensemble of raw Word2Vec models from S3  
**Output:** Aligned models + centroid + variance metrics

**Mathematical Details:**
- **Procrustes Alignment:** Finds optimal orthogonal transformation `R` minimizing `||A·R - B||²`
  - Solved via SVD: if `U·Σ·Vᵀ = BᵀA`, then `R = U·Vᵀ`
  - Preserves angles (cosine similarity) while enabling cross-model comparison
- **Variance Metrics:**
  - Higher variance → less reliable semantic relationship
  - Persistent low-variance terms → stable, trustworthy embeddings

---

### 6. **lambda-api**
**Container:** `functions/api/`  
**Technology:** Python, FastAPI (inferred from structure), Gensim  
**AWS Services:** Lambda (Function URL), S3

**Responsibilities:**
- Serve aggregated embedding data via HTTP API
- Load centroid embeddings + variance data from S3 on demand
- Provide endpoints for:
  - **Similarity queries:** Find most similar terms to a query term
  - **Cross-corpus comparison:** Compare term relationships across corpora
  - **Variance lookups:** Return stability metrics for terms
  - **Drift detection:** Identify semantic shifts between corpora

**Key Class: `KeyedVectorGroup`**
- Represents a single corpus's aggregated embeddings
- Lazy-loads centroid model and stability data from S3
- Supports similarity computations and metadata queries

**API Design:**
- Lightweight Lambda function (no persistent state)
- On-demand loading of artifacts from S3
- CORS-enabled for frontend access

---

## Core Capabilities

### Cross-Corpus Semantic Similarity
Compare how terms relate across different corpora in a unified embedding space. Centroid embeddings enable direct comparison even when models were trained independently.

### Semantic Drift Detection
Identify where meaning and association shift between corpora (e.g., domain-specific usage, temporal evolution, political framing differences).

### Variance-Aware Confidence Signals
Quantify stability of similarity relationships across stochastic training runs. High-variance associations indicate semantic uncertainty; low-variance indicates robust relationships.

### Interactive Exploration
React dashboard for inspecting terms, similarity rankings, drift patterns, and variance signals with interactive visualizations.

---

## Example Use Cases

### Research Questions This Platform Answers:

**Semantic Analysis:**
- Which words drift most in meaning between corpus A and corpus B?
- For a target term, which associations are **consistent** vs. **unstable** across training runs?
- Where do two corpora **agree semantically**, and where do they **diverge**?
- Which high-similarity pairs have **high variance** (indicating low confidence)?
- How does domain-specific jargon shift when comparing technical vs. general text?

**Content Verification:**
- How can we detect potential plagiarism by identifying unusually high similarity between documents with high confidence (low variance)?
- Can we distinguish between human-written and LLM-generated text by analyzing variance patterns in semantic relationships?
- Which terms show different stability characteristics in human vs. AI-generated content?
- Has this author actually discussed these concepts before, or is this content fabricated/deepfaked?
- Which terms in a suspicious document align with the author's historical vocabulary, and which show unexpected semantic drift?
- Does a video transcript match an individual's established speaking patterns and topic associations?

---

### Real-World Applications:

**Comparative Analysis:**
- **Content moderation:** Detect semantic drift in problematic language over time
- **Domain adaptation:** Understand how terminology shifts between industries
- **Model monitoring:** Track embedding stability in production systems
- **Research validation:** Assess confidence in semantic similarity claims
- **Comparative discourse analysis:** Examine how different communities use language

**Content Integrity & Verification:**
- **Plagiarism detection:** Identify copied or paraphrased content through cross-corpus similarity with confidence scoring—high similarity + low variance indicates likely plagiarism rather than coincidental overlap
- **AI-generated content detection:** Detect LLM-generated text by analyzing semantic variance patterns—AI models often produce more uniform, lower-variance term associations compared to natural human writing
- **Academic integrity:** Compare student submissions against reference corpora to flag suspicious similarity patterns while accounting for legitimate domain terminology overlap
- **Deepfake/fabricated content verification:** Detect fake videos or fabricated quotes by comparing transcripts against an author's historical corpus—semantic drift on key terms reveals content inconsistent with their established positions
- **Attribution verification:** Confirm whether contested content (leaked documents, anonymous posts, disputed statements) matches an author's historical semantic fingerprint
- **Misinformation detection:** Flag viral content attributed to public figures by measuring semantic distance from their verified past statements—high drift + low historical similarity indicates likely fabrication

---

## Why Variance-Aware Analysis Works for These Use Cases

### Plagiarism Detection
Traditional similarity metrics give false positives when documents share legitimate domain vocabulary. By measuring variance across stochastic runs, we can distinguish between:
- **High similarity + low variance** → likely copied/paraphrased content (stable, unusual overlap)
- **High similarity + high variance** → shared domain terminology (unstable, expected overlap)

### LLM-Generated Content Detection
AI-generated text often exhibits different statistical properties in embedding space:
- **Lower semantic variance** → AI models produce more predictable word associations
- **Uniform stability patterns** → Human writing shows more variability in term usage and context
- **Drift from human corpora** → LLM text may cluster differently when compared to human-written content in the same domain

### Deepfake & Attribution Verification

**Cross-corpus verification approach:**

When verifying attributed content, the platform compares:
1. **Suspicious corpus** (e.g., deepfake video transcript, contested statement)
2. **Author's historical corpus** (verified speeches, writings, interviews)

**Detection signals:**
- **Semantic drift on key terms:** If the author has never associated certain concepts before, high similarity to those topics in the suspicious content flags fabrication
- **Vocabulary mismatch:** Terms that appear in suspicious content but show high variance when aligned with author's historical embeddings indicate foreign influence
- **Topic consistency:** Legitimate content clusters near author's established semantic space; fabricated content shows unexpected drift

**Example workflow:**
```
1. Build ensemble embeddings for Author X's verified corpus (speeches, articles)
2. Build ensemble embeddings for suspicious video transcript
3. Align both spaces using Procrustes
4. Measure:
   - Which terms in suspicious content are absent/rare in historical corpus
   - Semantic drift on key claims
   - Variance in term associations (high variance = likely fabricated)
5. Flag content with high drift + low historical overlap as suspicious
```

**Why this works:**
- **Semantic fingerprinting:** Each author has distinctive patterns in how they connect concepts
- **Variance as verification:** Real content shows consistent term associations across the author's corpus; fabricated content introduces unstable new relationships
- **Context-aware:** Unlike simple keyword matching, embeddings capture *how* terms relate, not just their presence

The variance-aware approach provides **confidence metrics** that traditional binary detection methods lack.

---

## Technology Stack

### Backend (Serverless Functions)
- **Language:** Python 3.10+
- **ML/NLP:** Gensim (Word2Vec), NumPy, SciPy, scikit-learn, spaCy, NLTK
- **Cloud:** AWS Lambda, S3, DynamoDB, ECR
- **Deployment:** Docker containers via ECR, Lambda Function URLs
- **IaC:** `services.yaml` + deployment scripts

### Frontend ([embedding-analytics-frontend](https://github.com/areebms/embedding-analytics-frontend))
- **Framework:** React 18, Vite
- **Visualization:** Chart.js for interactive analytics
- **Deployment:** AWS Amplify
- **UI/UX:** Responsive design for exploring high-dimensional data

### Infrastructure
- **Containerization:** Docker, Docker Compose (local dev)
- **Registry:** AWS ECR for container images
- **Orchestration:** Lambda async invocations for parallel training
- **Storage:** S3 for all artifacts (models, CSVs, metadata)
- **State Tracking:** DynamoDB for pipeline orchestration

---

## Key Technical Decisions

### Why Ensemble Word2Vec?
- **Lightweight + fast:** Scales to large corpora without GPU requirements
- **Interpretable:** Direct word-to-word relationships (vs. contextual embeddings)
- **Variance analysis:** Enables quantifying model uncertainty through ensemble methods
- **Parallelizable:** Each model trains independently (perfect for serverless)

### Why Procrustes Alignment?
- Standard method for comparing independently trained embedding spaces
- Preserves relative geometric relationships while enabling cross-model comparison
- Computationally efficient (closed-form solution via SVD)
- Maintains cosine similarity structure (critical for semantic analysis)

### Why Serverless Lambda?
- **Auto-scaling:** Parallel training of ensemble models
- **Cost-efficient:** Pay only for actual compute time
- **Stateless:** Each function focuses on single responsibility
- **Fault-tolerant:** Failures isolated to individual invocations

### Design Tradeoffs
- **Model choice:** Word2Vec over transformers for speed + interpretability (future: add transformer ensemble support)
- **Alignment:** Procrustes over more complex methods (e.g., optimal transport) for computational efficiency
- **Storage:** S3 artifact storage for scalability + versioning
- **Lambda vs ECS:** Lambda for sporadic workloads; potential ECS migration for continuous high-volume processing

---

## Local Development

> **Note:** The live demo at [embedding-analytics.com](https://www.embedding-analytics.com/) is fully functional. Local setup is optional for running the full pipeline or extending the codebase.

### Backend Setup

```bash
# Clone repository
git clone https://github.com/areebms/embedding-analytics.git
cd embedding-analytics

# Set up environment
cp .env.example .env
# Edit .env with AWS credentials and configuration:
# - AWS_REGION
# - AWS_PROFILE
# - S3_BUCKET
# - AWS_URI_PREFIX (ECR registry)
# - KVECTORS_TRAINED (ensemble size, e.g., 10)

# Build containers locally
docker-compose build

# Test individual functions locally
docker-compose run lambda-scrape python src/main.py
docker-compose run lambda-tokenize python src/main.py
# etc.

# Deploy to AWS (requires AWS CLI + ECR setup)
cd infra
./push_to_ecr.sh
```

### Frontend Setup

```bash
# Clone repository
git clone https://github.com/areebms/embedding-analytics-frontend.git
cd embedding-analytics-frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

---

## Repository Structure

### Backend
```
embedding-analytics/
├── functions/              # Serverless function containers
│   ├── scrape/            # HTML fetching + text extraction
│   ├── tokenize/          # Sentence segmentation + lemmatization
│   ├── train-kvector/     # Single model training (worker)
│   ├── train-kvectors/    # Ensemble training orchestrator
│   ├── aggregate-data/    # Procrustes alignment + variance analysis
│   └── api/               # Query API for frontend
├── shared/                # Shared utilities (AWS helpers, common functions)
├── infra/                 # Deployment scripts + service configuration
│   ├── push_to_ecr.sh    # Deploy containers to ECR
│   └── services.yaml     # Lambda function definitions
├── docker-compose.yml    # Local development environment
└── .env.example          # Environment template
```

### Frontend
```
embedding-analytics-frontend/
├── src/                  # React components, hooks, services
├── public/               # Static assets
├── amplify.yml          # AWS Amplify deployment config
└── vite.config.js       # Build configuration
```

---

## Performance & Scalability

- **Parallel Training:** Lambda auto-scales to train N models simultaneously
- **Efficient Storage:** Compressed Gensim models stored in S3 with versioning
- **On-Demand Loading:** API lazy-loads artifacts only when needed
- **Async Processing:** Orchestrator uses Lambda async invocations (no blocking)
- **Memory Optimization:** Dynamic worker calculation based on Lambda memory allocation
- **Cost Optimization:** Serverless = pay-per-use (no idle infrastructure costs)

---

## Project Evolution & Roadmap

### Current Implementation
- Containerized serverless architecture on AWS Lambda  
- Ensemble Word2Vec training with variance tracking  
- Procrustes alignment for cross-model comparison  
- Variance metrics (overall, semantic, norm)  
- RESTful API for querying embeddings  
- Interactive React dashboard  
- Production deployment on AWS  

### Planned Enhancements
- [ ] **Memory optimization:** Stream sentence iteration to avoid loading full corpora into RAM
- [ ] **Batch processing:** Step Functions orchestration for complex multi-corpus pipelines
- [ ] **Export functionality:** CSV/JSON exports for offline analysis
- [ ] **Author fingerprinting:** Build verified corpus profiles for attribution verification

---

## Technical Highlights for Employers

**What this project demonstrates:**

🔹 **Serverless Architecture:** AWS Lambda + S3 + DynamoDB microservices  
🔹 **Machine Learning Engineering:** Ensemble methods, embedding alignment, variance quantification  
🔹 **Containerization:** Docker + ECR for reproducible deployments  
🔹 **API Design:** RESTful API with lazy-loading and CORS support  
🔹 **Full-Stack Development:** Python backend + React frontend integration  
🔹 **Data Pipeline Engineering:** Multi-stage ETL with state tracking  
🔹 **Parallel Computing:** Orchestration of distributed training jobs  
🔹 **Production Deployment:** Live system with monitoring and logging  
🔹 **Research-to-Production:** Translating NLP research concepts into scalable tools  
🔹 **Cloud-Native Development:** Stateless functions, object storage, managed services  
🔹 **Security Applications:** Content verification, deepfake detection, plagiarism analysis  

---

## License

- **Backend:** Apache-2.0 License
- **Frontend:** MIT License

See respective LICENSE files in each repository for details.

---

## Author

**Areeb Siddiqi**  
 [LinkedIn](https://www.linkedin.com/in/areeb-siddiqi/)  
 [GitHub](https://github.com/areebms)

---

## Acknowledgments

Inspired by research on embedding stability, variance-aware NLP, and cross-corpus semantic analysis. Built with best practices for production ML systems and serverless architectures.

---

**Looking for opportunities in ML Engineering, NLP, or Full-Stack Development.**  
This project demonstrates end-to-end capability: research concepts → serverless implementation → production deployment → real-world security applications.
