# Vector Space Alignment and Confidence Scoring

The alignment pipeline trains multiple Word2Vec models per book, aligns them into shared vector spaces via Generalized Procrustes Analysis, and uses the aligned ensemble to compute confidence intervals for every similarity result the API serves.

This document covers the alignment math, per-term metrics, and how confidence intervals surface model instability at query time. For where alignment sits in the wider system, see [pipeline.md](pipeline.md).

Data flow through alignment:

```text
seed models ─▶ within-book GPA ─▶ book centroid + aligned seed models
                                            │
                                            ▼
                          API confidence intervals (per query)
```

## Contents

- [Why alignment exists](#why-alignment-exists)
- [Within-book alignment](#within-book-alignment)
- [Orthogonal Procrustes rotation](#orthogonal-procrustes-rotation)
- [Metrics](#metrics)
- [Variable glossary](#variable-glossary)
- [Confidence intervals at query time](#confidence-intervals-at-query-time)
- [Design decisions](#design-decisions)
- [Engineering value](#engineering-value)

---

## Why alignment exists

A single Word2Vec model trained on a small corpus is unreliable. Different random seeds produce different vector spaces, and a result that looks meaningful in one run may not replicate in another.

Training multiple models addresses this, but creates a coordination problem: independently trained embeddings can preserve the same internal geometry while being arbitrarily rotated. Two models may agree that `labour` is close to `wage`, but their coordinate axes do not line up. Vectors cannot be averaged or compared directly.

Generalized Procrustes Analysis solves this by finding orthogonal rotations that bring all models into a shared orientation without distorting their internal cosine relationships. After alignment, the backend can compare vectors across seeds, build stable centroids, and compute confidence intervals for similarity scores.

---

## Within-book alignment

**Location:** `create_book_centroid.py`

Takes all seeded models for one book and aligns them into a common frame.

### Why raw vectors are used

Seed models train on identical data with identical hyperparameters, so their norm distributions are comparable. Keeping raw vectors preserves Word2Vec's learned norm structure, where content-bearing terms naturally carry more influence than low-signal words. Unit-normalizing at this stage would flatten that useful information.

### Iterative GPA

1. Compute centroid (element-wise mean across seed models)
2. Align each seed model to the centroid via Orthogonal Procrustes
3. Compute normalized `book_disparity` (mean across seeds — see [Variable glossary](#variable-glossary))
4. Repeat until `|prev_book_disparity - current_book_disparity| <= MIN_GRADIENT (0.0001)` or `MAX_ITERATIONS` (40)

In practice, convergence typically occurs within 5-10 iterations.

---

## Orthogonal Procrustes rotation

For each model, the pipeline solves:

```text
minimize ‖A·R − B‖²
```

Via SVD:

```text
U Σ Vᵀ = Bᵀ A
R = U Vᵀ
```

The constraint is that `R` is orthogonal: it can rotate or reflect the space, but cannot scale or shear it. This preserves vector norms and cosine similarities. The alignment changes coordinate frame without distorting the geometry of the learned space.

---

## Metrics

### Disparity (global)

```text
disparity = sum(||target_i - rotated_i||^2) / sum(||target_i - target_mean||^2)
```

Equivalent to `1 - R-squared`. Lower is better; 0 is perfect alignment. The overall disparity reported by GPA is the mean of per-model disparities.

### Per-term metrics (stored on centroid)

| Metric | Meaning |
|---|---|
| `count` | Token frequency (within-book) |
| `disparity` | Mean per-anchor squared error from centroid (total spread, higher = less stable) |
| `variance` | Squared distance of term vector from global mean vector (how far from center of space) |
| `r_squared` | `1 - (disparity / variance)` (goodness of fit, higher = more reliably positioned) |

Within-book metrics are in raw Euclidean space.

---

## Variable glossary

Reference for every constant and intermediate variable in the alignment pipeline: what it means, and why it has the value or name it does. The in-code identifiers `book_disparity` and `term_disparity` exist to keep `disparity` from meaning two different things — a whole-point-set fit score vs. a single term's leftover error — depending on where you read it. The persisted field name stays `disparity` for schema stability (see the per-term table below); only the in-code identifiers differ.

### Tuning constants (`procrustes_utils.py`)

| Name | Value | Meaning | Why this value |
|---|---|---|---|
| `VECTOR_SIZE` | 200 | Embedding dimensionality for every `KeyedVectors` object in the pipeline | Fixed by the upstream Word2Vec training config — not tuned here, just has to match it |
| `MAX_ITERATIONS` | 40 | Hard cap on GPA loop iterations before raising `"Kvectors not aligned"` | Convergence typically happens in 5–10 iterations; 40 is generous headroom, not a tuned bound |
| `MIN_GRADIENT` | 0.0001 | GPA stops when `prev_book_disparity - current_book_disparity <= MIN_GRADIENT` | Empirical stopping tolerance — small enough that further iterations wouldn't meaningfully improve the fit |
| `EPS` | 1e-6 | Floor for denominators that could be exactly zero (`term_variances`, `target_var`) | Sized to the float16 quantization noise floor of a ~200-dim vector: below this, two vectors are indistinguishable at storage precision anyway, so flooring can't mask real signal — it only guards literal zero |

### Per-alignment (whole point-set) quantities

One value per Procrustes solve — i.e. per seed or per book — not per term.

| Name | Meaning | Why named/chosen this way |
|---|---|---|
| `R` | Orthogonal rotation matrix solving `argmin‖A@R − B‖` | Direct output of `scipy.linalg.orthogonal_procrustes` |
| `book_disparity` (`ProcrustesResult.book_disparity`) | Normalized fit-quality score for one Procrustes solve: `sse / target_var`. 0 = perfect alignment. Analogous to `1 − R²` over the whole anchor set | Named `book_disparity` (not bare `disparity`) so the whole-alignment score is never confused with the per-term `disparity` field — this one is always scoped to a whole alignment, never a single term |
| `mean_book_disparity` | Mean of `book_disparity` across every seed in the GPA stack for one book | The value logged and used as `mean_disparity` in the pipeline table (persisted key unchanged, see below) |
| `prev_book_disparity` / `current_book_disparity` | GPA loop-local convergence tracking | Loop stops once the improvement between these two drops below `MIN_GRADIENT` |

### Per-term quantities

One value per term, stored as `KeyedVectors` vecattrs on the centroid and surfaced via `BookTermTable.alignment_stats`.

| Name | Meaning | Why chosen |
|---|---|---|
| `residuals` (raw, from `orthogonal_procrustes_alignment`) | Leftover squared Euclidean error for one term in one seed, after the best-fit rotation | Numerator that `book_disparity` sums over; also a standalone diagnostic ("which anchors did the rotation struggle with") |
| `combined_residuals` (returned by `gradient_descent_alignment`) | Seed-averaged `residuals`, via `np.nanmean` across the whole seed/book stack | Measures how *reproducible* a term's position was across independent training runs — not error against one reference, but disagreement between runs |
| `"disparity"` (persisted vecattr key, `alignment_stats.disparity`) | Same value as `combined_residuals[i]` for that term | The code concept is "per-term residual", but the persisted field name stays `disparity`: `functions/publish/src/publish_utils.py`'s `alignment_quality_attr` list and its tests read this key by name, so renaming it would silently break that service. Don't confuse it with `book_disparity` above — same word, different scope, different code identifier |
| `term_variances` / `safe_variances` | Squared distance of a term's final vector from the mean of *every* term vector in that centroid (`centroid.vectors.mean(axis=0)`) — how far this term sits from the vocabulary's "center of mass," not variance across the term's own repeated measurements | Chosen as the `r_squared` denominator because it's exactly the quantity queries read downstream — a term's distinctiveness *is* the payload later queries read off, so `r_squared` is really asking "is this term's distinctiveness bigger than its own measurement noise?" `safe_variances = max(term_variances, EPS)` |
| `r_squared` | `1 − (residual / safe_variance)` per term. 1.0 = perfect seed agreement, ≤ 0 = residual exceeds the term's own variance | Standard R²-style reliability score — normalizes an absolute residual by the term's own scale so terms of very different natural magnitude are comparable on one axis |

---

## Confidence intervals at query time

The API computes confidence intervals per query, not during alignment.

For each query-target term pair:

1. Compute cosine similarity in each aligned seed model independently
2. Take mean and standard deviation across models
3. Apply t-distribution critical value for 95% CI: `CI = mean +/- t_crit * (std / sqrt(n))`

```json
{
  "term": "capital",
  "similarity": 0.354,
  "similarity_ci": [0.312, 0.396]
}
```

Tight interval: the relationship was stable across training runs. Wide interval: the result was sensitive to model randomness.

### Why CIs matter for vector expressions

Contrast directions like `productive - unproductive` can produce small raw difference vectors when the two terms are semantically close. Per-operation normalization in the expression evaluator amplifies whatever signal (or noise) remains. If the amplified direction varies across seeds, the confidence interval widens.

This makes the CI a practical quality signal for contrast queries, telling the user whether the ensemble consistently learned the relationship being probed. A standard Word2Vec tool does not provide this information.

---


## Design decisions

**Generalized Procrustes over pairwise:** Pairwise alignment rotates every model toward a single reference, accumulating error and biasing toward that reference. GPA iteratively finds a centroid that minimizes total disparity across all models simultaneously.

**Orthogonal rotations only:** The rotation matrix is constrained to be orthogonal (rotation + reflection, no scaling or shearing). This preserves vector norms and cosine similarities, changing the coordinate frame without distorting learned geometry.

**Float16 storage:** Aligned vectors are stored as float16 in DynamoDB to reduce storage and payload size. The precision loss (~3 decimal digits) is negligible for cosine similarity displayed to 3 decimal places.

---

## Engineering value

The alignment system converts model variability into an explicit, queryable reliability signal. Instead of returning a single nearest-neighbor list from one embedding model, the backend serves similarity results with uncertainty quantified.

The core production idea: do not hide model instability. Measure it, expose it through the API, and let the frontend render it as a first-class part of the user experience.
