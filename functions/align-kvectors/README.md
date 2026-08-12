# align-kvectors

*Stage 4 of 6. [Pipeline overview](../../docs/internals.md) · [Project README](../../README.md)*
**Libraries:** NumPy, SciPy

`create_book_centroid.py` aligns all seeded models for one book into a shared
vector space and builds a per-book centroid with per-term stability metrics.
Shared primitives in `procrustes_utils.py`. Output: `kvectors/{index}/aligned/`

Independently trained Word2Vec models can learn equivalent internal geometry
while representing it in arbitrarily rotated coordinate systems. Two models may
agree that `labour` is close to `wage` without their axes lining up, so vectors
cannot be averaged or compared directly.

Generalized Procrustes Analysis (Gower, 1975) finds orthogonal rotations bringing
all models into a shared orientation without distorting internal cosine
relationships.

```text
seed models ─▶ within-book GPA ─▶ book centroid + aligned seed models
                                            │
                                            ▼
                          API confidence intervals (per query)
```

## Within-book GPA

**Location:** `create_book_centroid.py`

Seed models train on identical data with identical hyperparameters, so their norm
distributions are comparable. Raw vectors are kept rather than unit-normalized —
that preserves Word2Vec's learned norm structure, where content-bearing terms
carry more influence than low-signal words.

1. Compute centroid (element-wise mean across seed models)
2. Align each seed to the centroid via Orthogonal Procrustes
3. Compute normalized `book_disparity` (mean across seeds)
4. Repeat until `prev_book_disparity - current_book_disparity <= MIN_GRADIENT`
   (0.0001) or `MAX_ITERATIONS` (40)

Convergence typically occurs within 5–10 iterations.

## Procrustes rotation

For each model, solve `minimize ‖A·R − B‖²` via SVD:

```text
U Σ Vᵀ = Bᵀ A
R = U Vᵀ
```

`R` is constrained orthogonal: it can rotate or reflect, but not scale or shear.
This preserves vector norms and cosine similarities — the coordinate frame
changes without distorting the geometry of the learned space.

## Metrics

**Disparity (global):**

```text
disparity = sum(||target_i - rotated_i||^2) / sum(||target_i - target_mean||^2)
```

Equivalent to `1 - R²`. Lower is better; 0 is perfect alignment. The disparity
GPA reports is the mean of per-model disparities.

**Per-term (stored on centroid), in raw Euclidean space:**

| Metric | Meaning |
|---|---|
| `count` | Token frequency (within-book) |
| `disparity` | Mean per-anchor squared error from centroid (higher = less stable) |
| `variance` | Squared distance of term vector from global mean vector |
| `r_squared` | `1 - (disparity / variance)` (higher = more reliably positioned) |

## Variable glossary

The in-code identifiers `book_disparity` and `term_disparity` exist to keep
`disparity` from meaning two different things — a whole-point-set fit score vs. a
single term's leftover error. The persisted field name stays `disparity` for
schema stability.

### Tuning constants (`procrustes_utils.py`)

| Name | Value | Meaning | Why this value |
|---|---|---|---|
| `VECTOR_SIZE` | 200 | Embedding dimensionality for every `KeyedVectors` object | Fixed by the upstream training config — has to match it |
| `MAX_ITERATIONS` | 40 | Hard cap before raising `"Kvectors not aligned"` | Convergence happens in 5–10; 40 is headroom, not a tuned bound |
| `MIN_GRADIENT` | 0.0001 | GPA stop threshold | Empirical tolerance — below this, further iterations don't meaningfully improve fit |
| `EPS` | 1e-6 | Floor for denominators that could be zero | Sized to the float16 quantization noise floor of a ~200-dim vector: below this two vectors are indistinguishable at storage precision anyway |

### Per-alignment quantities

One value per Procrustes solve — per seed or per book, not per term.

| Name | Meaning |
|---|---|
| `R` | Orthogonal rotation matrix from `scipy.linalg.orthogonal_procrustes` |
| `book_disparity` | Normalized fit score for one solve: `sse / target_var`. 0 = perfect. Named to never be confused with the per-term `disparity` field |
| `mean_book_disparity` | Mean across every seed in one book's GPA stack; logged as `mean_disparity` in the pipeline table |
| `prev_/current_book_disparity` | GPA loop convergence tracking |

### Per-term quantities

Stored as `KeyedVectors` vecattrs on the centroid, surfaced via
`BookTermTable.alignment_stats`.

| Name | Meaning |
|---|---|
| `residuals` | Leftover squared Euclidean error for one term in one seed, after best-fit rotation |
| `combined_residuals` | Seed-averaged `residuals` via `np.nanmean` — how *reproducible* a term's position was across independent runs |
| `"disparity"` (persisted) | Same value as `combined_residuals[i]`. `publish_utils.py`'s `alignment_quality_attr` list reads this key by name, so the name is load-bearing |
| `term_variances` / `safe_variances` | Squared distance of a term's vector from the mean of *every* term vector in that centroid — how far it sits from the vocabulary's centre of mass. `safe_variances = max(term_variances, EPS)` |
| `r_squared` | `1 − (residual / safe_variance)` per term. 1.0 = perfect seed agreement, ≤ 0 = residual exceeds the term's own variance |

`term_variances` is the `r_squared` denominator because it is exactly the
quantity queries read downstream — a term's distinctiveness is the payload, so
`r_squared` asks "is this term's distinctiveness bigger than its own measurement
noise?"

**Float16 storage:** aligned vectors are stored as float16 in DynamoDB. The
precision loss (~3 decimal digits) is negligible for cosine similarity displayed
to 3 decimal places.
