# Vector Space Alignment and Confidence Scoring

The alignment pipeline trains multiple Word2Vec models per book, aligns them into shared vector spaces via Generalized Procrustes Analysis, and uses the aligned ensemble to compute confidence intervals for every similarity result the API serves.

This document covers the alignment math, the two alignment modes, per-term metrics, and how confidence intervals surface model instability at query time.

---

## Why alignment exists

A single Word2Vec model trained on a small corpus is unreliable. Different random seeds produce different vector spaces, and a result that looks meaningful in one run may not replicate in another.

Training multiple models addresses this, but creates a coordination problem: independently trained embeddings can preserve the same internal geometry while being arbitrarily rotated. Two models may agree that `labour` is close to `wage`, but their coordinate axes do not line up. Vectors cannot be averaged or compared directly.

Generalized Procrustes Analysis solves this by finding orthogonal rotations that bring all models into a shared orientation without distorting their internal cosine relationships. After alignment, the backend can compare vectors across seeds, build stable centroids, align books into a shared corpus frame, and compute confidence intervals for similarity scores.

---

## Alignment levels

| Level | Purpose | Output |
|---|---|---|
| Within-book | Align seeded models trained on the same text | Per-book centroid + aligned seed models |
| Cross-book | Align book centroids into a shared corpus frame | Corpus centroid for cross-author comparison |

Both modes use the same alignment primitive, `weighted_orthogonal_procrustes`, which dispatches on the `counts` parameter:

| | Within-book | Cross-book |
|---|---|---|
| Counts | `None` | Uniform (1 for terms in 2+ books, 0 otherwise) |
| Normalization | None (raw vectors) | Unit-normalized before SVD |
| Weights | Uniform (implicit norm-based influence) | Uniform among shared terms |
| Disparity space | Raw Euclidean | Normalized, bounded ~[0, 1] |

---

## Within-book alignment

**Location:** `create_book_centroid.py`

Takes all seeded models for one book and aligns them into a common frame.

### Why raw vectors are used

Seed models train on identical data with identical hyperparameters, so their norm distributions are comparable. Keeping raw vectors preserves Word2Vec's learned norm structure, where content-bearing terms naturally carry more influence than low-signal words. Unit-normalizing at this stage would flatten that useful information.

### Iterative GPA

1. Compute centroid (element-wise mean across seed models)
2. Align each seed model to the centroid via Orthogonal Procrustes
3. Compute normalized disparity
4. Repeat until `|prev_disparity - current_disparity| <= 0.0001` or 40 iterations

In practice, convergence typically occurs within 5-10 iterations.

### Opportunistic corpus rotation

If a corpus centroid already exists in S3, the book centroid is rotated into the corpus frame after within-book GPA completes. This allows incrementally adding books without rebuilding the corpus alignment.

---

## Orthogonal Procrustes rotation

For each model, the pipeline solves:

```text
minimize ||A * R - B||^2
```

Via SVD:

```text
U * Sigma * V^T = B^T * A
R = U * V^T
```

The constraint is that `R` is orthogonal: it can rotate or reflect the space, but cannot scale or shear it. This preserves vector norms and cosine similarities. The alignment changes coordinate frame without distorting the geometry of the learned space.

---

## Cross-book alignment

**Location:** `create_corpus_centroid.py`

Rotates completed book centroids into a shared corpus frame for cross-author comparison. Runs as a separate CLI command outside the Step Function, requires 2+ aligned books.

### Why normalization is used across books

Different books produce systematically different norm distributions. Orthogonal rotations preserve norms and cannot compensate for scale heterogeneity. Unit-normalizing vectors before SVD removes that mismatch and lets the rotation focus on angular relationships.

### Term inclusion and weighting

Terms in 2+ books receive count 1. Single-book terms receive count 0: they remain in the corpus vocabulary for downstream lookup but do not influence the rotation, since they have no cross-book signal.

Uniform weighting was chosen after experimentation. Coverage-based weighting (scaling by actual book counts) produced lower R-squared than uniform weights at the current corpus size (5 books, count range 2-5). The count range is too narrow for coverage weighting to differentiate meaningfully.

### Post-GPA normalization

After GPA converges, the centroid vectors are explicitly unit-normalized before storage. This ensures the stored corpus centroid and its diagnostic metrics (R-squared, variance) are in the same normalized space as the alignment residuals.

### Post-rebuild re-alignment

After rebuilding the corpus centroid, per-book centroids need to be re-aligned to the new frame by re-running `create_book_centroid.py` for each book. Within-book GPA is deterministic, so re-running the full pipeline produces the same centroid before applying the new corpus rotation.

---

## Metrics

### Disparity (global)

```text
disparity = sum(w_i * ||target_i - rotated_i||^2) / sum(w_i * ||target_i - target_mean||^2)
```

Equivalent to `1 - R-squared`. Lower is better; 0 is perfect alignment. The overall disparity reported by GPA is the mean of per-model disparities.

### Per-term metrics (stored on centroid)

| Metric | Meaning |
|---|---|
| `count` | Token frequency (within-book) or cross-book coverage flag (corpus) |
| `disparity` | Mean per-anchor squared error from centroid (total spread, higher = less stable) |
| `variance` | Squared distance of term vector from global mean vector (how far from center of space) |
| `r_squared` | `1 - (disparity / variance)` (goodness of fit, higher = more reliably positioned) |

Within-book metrics are in raw Euclidean space. Corpus metrics are in normalized space. Do not compare R-squared values between the two.

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

## Diachronic comparison

The corpus spans the late 18th to mid-19th century: Steuart, Smith, Ricardo, Mill, Bastiat. Once book centroids are aligned into a shared corpus frame, the same query can be evaluated across authors. If a term's neighborhood shifts systematically from earlier to later books, that is evidence of semantic drift.

This connects to the diachronic embeddings literature, particularly Hamilton, Leskovec, and Jurafsky's HistWords project, which applies the same train-align-compare methodology to historical corpora. This project applies that methodology to a specific intellectual tradition, making drift findings historically interpretable.

The cross-book alignment is what makes the comparison valid. Without it, per-author vector spaces would share no common coordinate system.

---

## Design decisions

**Generalized Procrustes over pairwise:** Pairwise alignment rotates every model toward a single reference, accumulating error and biasing toward that reference. GPA iteratively finds a centroid that minimizes total disparity across all models simultaneously.

**Orthogonal rotations only:** The rotation matrix is constrained to be orthogonal (rotation + reflection, no scaling or shearing). This preserves vector norms and cosine similarities, changing the coordinate frame without distorting learned geometry.

**Float16 storage:** Aligned vectors are stored as float16 in DynamoDB to reduce storage and payload size. The precision loss (~3 decimal digits) is negligible for cosine similarity displayed to 3 decimal places.

**Uniform weights across books:** At the current corpus size (5 books, count range 2-5), coverage-weighted GPA produces lower R-squared than uniform weights. As the corpus grows and the count range widens, coverage weighting may become beneficial.

**Single-book terms retained with zero weight:** They have no cross-book signal (centroid is their own vector, residual is always zero), so they must not influence the rotation. Setting their count to 0 keeps them in the corpus vocabulary for downstream lookup while excluding them from SVD. No separate filtering step needed.

---

## Engineering value

The alignment system converts model variability into an explicit, queryable reliability signal. Instead of returning a single nearest-neighbor list from one embedding model, the backend serves similarity results with uncertainty quantified.

The core production idea: do not hide model instability. Measure it, expose it through the API, and let the frontend render it as a first-class part of the user experience.
