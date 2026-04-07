# Vector Space Alignment

The `align-kvectors` stage implements **Generalized Procrustes Analysis** (Gower, 1975) to align independently trained Word2Vec models into a shared coordinate system.

## Why alignment is necessary

Each Word2Vec model learns an arbitrary rotation of the vector space — the relationships between terms are preserved, but the axes are meaningless. Two models trained on identical data will produce vectors that encode the same similarities but point in completely different directions. Procrustes alignment finds the optimal rotation to put them in the same frame so vectors can be directly compared.

## Algorithm

### Step 1: Initial alignment

All models are aligned to the first model via Orthogonal Procrustes. This gives gradient descent a stable starting point and ensures reproducibility (models are sorted by filename before alignment begins).

### Step 2: Iterative convergence

Generalized Procrustes Analysis via gradient descent:

1. Compute centroid vectors (element-wise mean across all models for each term)
2. Align each model to the centroid via Orthogonal Procrustes
3. Compute normalized disparity
4. Repeat until gradient drops below threshold

**Procrustes rotation:** For each model, find orthogonal matrix `R` minimizing `||A·R - B||²`, solved via SVD:

```
U·Σ·Vᵀ = BᵀA
R = U·Vᵀ
```

Applied in-place. Norms recomputed after each rotation. Preserves cosine similarity while putting all models in the same coordinate frame.

**Convergence:** Stops when `|prev_disparity - current_disparity| ≤ 0.0001` or after 40 iterations. In practice, convergence typically occurs within 5–10 iterations.

### Step 3: Centroid construction

After convergence, a final centroid `KeyedVectors` model is built with per-term stability metrics attached as vector attributes.

## Metrics

### Normalized disparity (global)

```
normalized_disparity = ||centroid - aligned||²_F / ||centroid - mean(centroid)||²_F
```

Equivalent to `1 - R²` (coefficient of determination). Measures how well the alignment explains the variance in the data. Lower = better fit.

### Per-term metrics (stored on centroid)

| Attribute | Formula | Interpretation |
|---|---|---|
| `count` | Token frequency | How often the term appears in the corpus |
| `disparity` | Mean SSE from centroid | Total spread across models — higher = less stable |
| `variance` | SSD of centroid from global mean | How far this term's centroid is from the average term |
| `r_squared` | `1 - (disparity / variance)` | Goodness of fit — higher = more reliably positioned |

### Confidence intervals (at query time)

The API computes confidence intervals per query, not during alignment. For each term pair:

1. Compute cosine similarity between the query vector and the target vector *in each aligned model independently*
2. Take the mean and standard deviation across models
3. Apply t-distribution critical value for 95% CI: `CI = mean ± t_crit * (std / √n)`

This captures how consistently the models agree on a relationship. A tight CI means the similarity score is robust across training runs.

## Design decisions

**Why Generalized Procrustes over pairwise?** Pairwise alignment (align model 2 to model 1, model 3 to model 1, etc.) accumulates error and is biased toward the reference model. Generalized Procrustes iteratively finds the centroid that minimizes total disparity across all models simultaneously.

**Why orthogonal rotation only?** The Procrustes solution is constrained to orthogonal matrices (rotation + reflection, no scaling or shearing). This preserves vector norms and cosine similarities — the alignment changes coordinate frame without distorting the geometry of the learned space.

**Why float16 storage?** Aligned vectors are stored as float16 in DynamoDB to reduce storage footprint. The precision loss is negligible for cosine similarity computation — float16 has ~3 decimal digits of precision, and similarity scores are displayed to 3 decimal places.
