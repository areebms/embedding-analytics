# Vector Space Alignment

The alignment pipeline implements **Generalized Procrustes Analysis** (Gower, 1975) at two levels: within-book (aligning seed models into a per-book centroid) and cross-book (aligning book centroids into a shared corpus frame).

## Why alignment is necessary

Each Word2Vec model learns an arbitrary rotation of the vector space. The relationships between terms are preserved, but the axes are meaningless. Two models trained on identical data will produce vectors that encode the same similarities but point in completely different directions. Procrustes alignment finds the optimal rotation to put them in the same frame so vectors can be directly compared.

## The two alignment modes

The pipeline uses a single primitive, `weighted_orthogonal_procrustes`, which dispatches on the `counts` parameter:

| | Within-book | Cross-book |
|---|---|---|
| **Purpose** | Align ~30 seed models per book | Align book centroids into a corpus frame |
| **Counts** | `None` | Uniform (1 for terms in 2+ books, 0 for single-book terms) |
| **Normalization** | None (raw vectors) | Unit-normalized before SVD |
| **Weights** | Uniform (implicit norm-based bias) | Uniform (single-book terms excluded) |
| **Disparity space** | Raw Euclidean | Normalized, bounded ~[0, 1] |

**Why no normalization within-book:** Seeds are trained on identical data with identical hyperparameters, so norm distributions are comparable. Word2Vec's natural norm structure (where rare-but-meaningful terms have larger norms) acts as useful implicit weighting toward content-bearing terms. Discarding it via unit-normalization would equalize stop-words with content words in the rotation loss.

**Why normalization cross-book:** Different books trained on different texts produce systematically different norm distributions. Orthogonal rotations preserve norms and cannot compensate for scale heterogeneity. Pre-normalizing removes the gap before SVD sees it.

**Do not compare disparity values across modes.** Within-book disparity is in raw Euclidean space; cross-book disparity is in normalized space.

## Within-book alignment

**Location:** `create_book_centroid.py`

### Iterative convergence

Generalized Procrustes Analysis via gradient descent:

1. Compute centroid vectors (element-wise mean across all seed models for each term)
2. Align each seed model to the centroid via Orthogonal Procrustes
3. Compute normalized disparity
4. Repeat until gradient drops below threshold

**Procrustes rotation:** For each model, find orthogonal matrix `R` minimizing `||A·R - B||²`, solved via SVD:

```
U·Σ·Vᵀ = BᵀA
R = U·Vᵀ
```

Applied in-place. Norms recomputed after each rotation.

**Convergence:** Stops when `|prev_disparity - current_disparity| <= 0.0001` or after 40 iterations. In practice, convergence typically occurs within 5-10 iterations.

### Centroid construction

After convergence, a centroid `KeyedVectors` model is built with per-term stability metrics attached as vector attributes.

### Opportunistic corpus rotation

If a corpus centroid already exists in S3, the book centroid is rotated into the corpus frame via `align_to_corpus_centroid`. This allows incrementally adding books without rebuilding the corpus. The rotation uses unit-normalized vectors and reads per-term counts from the corpus centroid's vecattrs.

## Cross-book alignment

**Location:** `create_corpus_centroid.py`

Aligns book centroids into a shared corpus frame for cross-book comparison. Runs as a separate CLI command outside the Step Function, requires 2+ aligned books.

### Term selection and weighting

All terms from all books are included. Single-book terms are assigned count 0 (excluded from the rotation via zero weight). All other terms receive count 1 (uniform weighting). Inside `weighted_orthogonal_procrustes`, the `sqrt` of the counts is applied as a row multiplier, but since the non-zero counts are all 1, the effective weighting is uniform.

Single-book terms are zeroed rather than removed from the term list: they still appear in the corpus centroid vocabulary with their vectors, but they do not influence the rotation.

This weighting scheme was chosen after experimentation. Coverage-based weighting (scaling by actual book counts) produced lower r-squared than uniform weights at the current scale (5 books, count range 2-5). The count range is too narrow for coverage weighting to differentiate meaningfully.

### Centroid normalization

After GPA converges, the centroid vectors are explicitly unit-normalized before being passed to `build_centroid_kvector`:

```python
centroid_vectors /= np.linalg.norm(centroid_vectors, axis=1, keepdims=True)
```

This normalization happens after GPA, not during. GPA itself uses unit-normalization internally within `weighted_orthogonal_procrustes` (because counts are provided), but `compute_centroid_vectors` returns raw means of the rotated vectors. The post-GPA normalization ensures the stored corpus centroid and its diagnostic metrics (r-squared, variance) are in the same normalized space as the alignment residuals.

### Post-rebuild re-alignment

After rebuilding the corpus centroid, per-book centroids are aligned to the old corpus frame. Re-running `create_book_centroid.py` for each book will re-align them to the new frame. The within-book GPA is deterministic, so re-running the full pipeline produces the same centroid before applying the new corpus rotation.

## Metrics

### Disparity (global)

Disparity is computed as weighted sum of squared errors divided by weighted target variance:

```
disparity = sum(w_i * ||target_i - rotated_i||^2) / sum(w_i * ||target_i - target_mean||^2)
```

where `w_i` is the per-anchor weight and `target_mean` is the weighted mean of the target matrix. Equivalent to `1 - R²`. Lower is better; 0 is perfect alignment. The overall disparity reported by GPA is the mean of per-model disparities across all models/books in the alignment.

### Per-term metrics (stored on centroid)

| Attribute | Formula | Interpretation |
|---|---|---|
| `count` | Token frequency (within-book) or book coverage flag (corpus) | How often the term appears, or whether it participates in corpus alignment |
| `disparity` | Mean per-anchor squared error from centroid | Total spread across models, higher = less stable |
| `variance` | Squared distance of term's vector from global mean vector | How far this term sits from the center of the space |
| `r_squared` | `1 - (disparity / variance)` | Goodness of fit, higher = more reliably positioned |

For the corpus centroid, `variance` and `r_squared` are computed on the unit-normalized centroid vectors, so they are in the same space as the normalized residuals from cross-book Procrustes.

For within-book centroids, these metrics are in raw Euclidean space. Do not compare r-squared values between within-book and corpus centroids.

### Confidence intervals (at query time)

The API computes confidence intervals per query, not during alignment. For each term pair:

1. Compute cosine similarity between the query vector and the target vector *in each aligned model independently*
2. Take the mean and standard deviation across models
3. Apply t-distribution critical value for 95% CI: `CI = mean ± t_crit * (std / √n)`

This captures how consistently the models agree on a relationship. A tight CI means the similarity score is robust across training runs.

## Design decisions

**Why Generalized Procrustes over pairwise?** Pairwise alignment (align model 2 to model 1, model 3 to model 1, etc.) accumulates error and is biased toward the reference model. Generalized Procrustes iteratively finds the centroid that minimizes total disparity across all models simultaneously.

**Why orthogonal rotation only?** The Procrustes solution is constrained to orthogonal matrices (rotation + reflection, no scaling or shearing). This preserves vector norms and cosine similarities. The alignment changes coordinate frame without distorting the geometry of the learned space.

**Why float16 storage?** Aligned vectors are stored as float16 in DynamoDB to reduce storage footprint. The precision loss is negligible for cosine similarity computation. Float16 has ~3 decimal digits of precision, and similarity scores are displayed to 3 decimal places.

**Why uniform weights over coverage-based?** With 5 books and a count range of 2-5, coverage-weighted GPA produces lower r-squared than uniform weights. The relative spread between buckets is too small to carry meaningful signal. As the corpus grows and the count range widens, coverage weighting may become beneficial.

**Why normalize the corpus centroid?** The cross-book alignment operates in normalized space (unit vectors). Storing the corpus centroid in normalized space ensures the diagnostic metrics (r-squared, variance) are consistent with the alignment residuals. Raw norm information can be recovered from individual book centroids if needed.

**Why exclude single-book terms via zero weight rather than removing them?** Single-book terms have no cross-book signal (their centroid is their own vector, residual is always zero), so they must not influence the rotation. Setting their count to 0 keeps them in the corpus centroid vocabulary for downstream lookup while excluding them from the SVD. This avoids a separate filtering step and keeps the term list consistent across the pipeline.
