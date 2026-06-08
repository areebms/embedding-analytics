import logging
import tempfile
from collections import defaultdict
from typing import NamedTuple, Optional

import numpy as np
from scipy.linalg import orthogonal_procrustes
from statistics import mean
from gensim.models import KeyedVectors

from shared.s3 import S3Loader, upload_file
from shared.session import get_session

CORPUS_CENTROID_KEY = "kvectors/corpus_centroid.model"
MAX_ITERATIONS = 40
MIN_GRADIENT = 0.0001
VECTOR_SIZE = 200

logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


def load_corpus_centroid(session=None) -> Optional[KeyedVectors]:
    """
    Returns the corpus centroid if it exists, else None.

    None is the bootstrap case (first book ever, or a wiped corpus state).
    Callers should treat None as "skip corpus rotation; rely on rebuild_corpus
    to fold this book in later."
    """
    session = session or get_session()
    loader = S3Loader(session)
    try:
        with loader.load_file(CORPUS_CENTROID_KEY) as (_, local_path):
            return KeyedVectors.load(local_path)
    except Exception as exc:
        logger.info("No corpus centroid at %s: %s", CORPUS_CENTROID_KEY, exc)
        return None


def rotate(kvector: KeyedVectors, rotation: np.ndarray) -> None:
    """
    Apply a rotation to a KeyedVectors object in place. Clears cached norms
    so subsequent similarity calls recompute against the rotated vectors.
    """
    kvector.vectors = kvector.vectors @ rotation
    if hasattr(kvector, "vectors_norm"):
        kvector.vectors_norm = None
    if hasattr(kvector, "norms"):
        kvector.norms = None
    kvector.fill_norms(force=True)


class ProcrustesResult(NamedTuple):
    """
    Return value of weighted_orthogonal_procrustes.

    R: orthogonal rotation matrix, shape (dim, dim). Apply to other
       matrices via `M @ R` to bring them into the target frame.
       Valid for both normalized and un-normalized data — orthogonal
       rotations commute with per-row scaling.
    disparity: scalar. Weighted SSE between rotated source and target,
       normalized by weighted variance of target. Smaller is better;
       0 is perfect alignment. NOT comparable across the two modes —
       within-book disparity is in raw Euclidean units, cross-book
       disparity is in unit-sphere distance units bounded by [0, 4].
    residuals: per-anchor squared distances after rotation. Shape
       (n_anchors,). Useful as a diagnostic — sorting descending shows
       which anchors the rotation struggled with. In cross-book mode
       (with counts) residuals are in normalized space [0, 4]; in
       within-book mode they are in raw Euclidean units (unbounded).
    """

    R: np.ndarray
    disparity: float
    residuals: np.ndarray


def weighted_orthogonal_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    counts: Optional[np.ndarray] = None,
) -> ProcrustesResult:
    """
    Find the orthogonal rotation R that aligns source to target, with
    per-anchor weighting when counts are provided.

    Two modes, dispatched on whether counts is None:

    Within-book mode (counts=None):
        Plain orthogonal Procrustes on the raw input matrices, uniform
        weights, no normalization. Equivalent to scipy's
        orthogonal_procrustes plus a normalized disparity computation.
        Use for within-book seed alignment where r_squared downstream
        depends on raw Euclidean distances.

    Cross-book mode (counts provided):
        Anchors are unit-normalized internally; rotation minimizes
        weighted Frobenius distance with weight sqrt(counts) per row.
        The sqrt is what makes the weighting linear in coverage —
        loss contribution of an anchor scales as count_i, not count_i².

    Mathematically, in cross-book mode:

        R* = argmin_R  Σᵢ countsᵢ · ‖x̂ᵢ R − ŷᵢ‖²    s.t.  RᵀR = I

    where x̂ᵢ, ŷᵢ are unit-normalized rows. Pre-multiplying both
    matrices by diag(√counts) is equivalent: scipy's closed-form solver
    sees only the matrix product YᵀWᵀWX = YᵀΛX in the SVD step.

    Parameters
    ----------
    source : (n_anchors, dim) array
        Source anchor matrix. The rotation maps source toward target.
        Not mutated.
    target : (n_anchors, dim) array
        Target anchor matrix. Row i pairs with row i of source.
    counts : (n_anchors,) array of non-negative ints, optional
        Per-anchor coverage for cross-book mode. None for within-book.

    Returns
    -------
    ProcrustesResult(R, disparity, residuals) — see class docstring.
    """
    if source.shape != target.shape:
        raise ValueError(f"source {source.shape} and target {target.shape} must match")
    if source.ndim != 2:
        raise ValueError(f"expected 2D arrays, got {source.ndim}D")
    n_anchors, _ = source.shape
    if n_anchors == 0:
        raise ValueError("need at least one anchor")

    eps = 1e-12

    if counts is None:
        # Within-book: raw vectors, uniform weights. Preserves the
        # implicit norm-based weighting that biases the rotation toward
        # higher-norm (typically more content-bearing) terms, and keeps
        # downstream r_squared math in raw Euclidean units.
        source_proc = source
        target_proc = target
        weights = np.ones(n_anchors)
    else:
        counts = np.asarray(counts)
        if counts.shape != (n_anchors,):
            raise ValueError(f"counts shape {counts.shape} != ({n_anchors},)")
        if (counts < 0).any():
            raise ValueError("counts must be non-negative")

        # Cross-book: unit-normalize to remove cross-book scale
        # heterogeneity before weighting. eps guards against zero-norm
        # rows propagating NaN through SVD.
        source_norms = np.linalg.norm(source, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target, axis=1, keepdims=True)
        source_proc = source / np.maximum(source_norms, eps)
        target_proc = target / np.maximum(target_norms, eps)
        weights = counts.astype(np.float64)

    # Apply sqrt-weights as row scaling. The squared-distance loss then
    # has each row contributing in proportion to its weight.
    sqrt_w = np.sqrt(weights)[:, np.newaxis]
    R, _ = orthogonal_procrustes(source_proc * sqrt_w, target_proc * sqrt_w)

    # Diagnostics computed in unweighted space — the weighting was meant
    # to bias the rotation, not the per-anchor errors you inspect after.
    rotated = source_proc @ R
    diff = target_proc - rotated
    residuals = np.sum(diff * diff, axis=1)

    weighted_sse = float(np.sum(weights * residuals))
    weight_sum = float(np.sum(weights))
    target_mean = np.sum(target_proc * weights[:, np.newaxis], axis=0) / weight_sum
    target_centered = target_proc - target_mean
    weighted_target_var = float(np.sum(weights * np.sum(target_centered**2, axis=1)))
    disparity = (
        weighted_sse / weighted_target_var
        if weighted_target_var > eps
        else float("inf")
    )

    return ProcrustesResult(R=R, disparity=disparity, residuals=residuals)


def normalized_disparity_alignment(terms, kvector, centroid_vectors, counts=None):
    term_indices, present_terms = zip(
        *[(i, term) for i, term in enumerate(terms) if term in kvector]
    )
    term_indices = list(term_indices)

    centroid_matrix = centroid_vectors[term_indices].astype(np.float32)
    kv_matrix = np.stack([kvector[term] for term in present_terms]).astype(np.float32)

    if counts is not None:
        term_counts = counts[term_indices].astype(np.float32)
    else:
        term_counts = None

    result = weighted_orthogonal_procrustes(kv_matrix, centroid_matrix, term_counts)

    rotate(kvector, result.R)

    return result, term_indices


def compute_centroid_vectors(terms, kvector_stack):
    result = []
    for term in terms:
        vectors = []
        for kv in kvector_stack:
            if term in kv.key_to_index:
                vectors.append(kv[term])
        if not vectors:
            raise KeyError(f"Term not found: {term}")
        result.append(np.mean(np.stack(vectors), axis=0))

    return np.stack(result).astype(np.float32)


def gradient_descent_alignment(
    terms,
    kvector_stack,
    counts=None,
    max_iterations=MAX_ITERATIONS,
    min_gradient=MIN_GRADIENT,
):
    # Generalized Procrustes Analysis (Gower, 1975)
    prev_disparity = float("inf")
    for iteration in range(max_iterations):
        normalized_disparities = []
        stack_residuals = []
        centroid_vectors = compute_centroid_vectors(terms, kvector_stack)

        n_terms = len(terms)
        for kvector in kvector_stack:
            result, term_indices = normalized_disparity_alignment(
                terms, kvector, centroid_vectors, counts
            )
            normalized_disparities.append(result.disparity)
            full_residuals = np.full(n_terms, np.nan)
            full_residuals[term_indices] = result.residuals
            stack_residuals.append(full_residuals)
        current_disparity = mean(normalized_disparities)

        if (prev_disparity - current_disparity) <= min_gradient:
            combined_residuals = np.nanmean(stack_residuals, axis=0)
            return (
                centroid_vectors,
                combined_residuals,
                current_disparity,
                iteration + 1,
            )
        prev_disparity = current_disparity

    raise Exception("Kvectors not aligned")


def load_book_centroid(session, index) -> KeyedVectors:
    loader = S3Loader(session)
    with loader.load_file(f"kvectors/{index}/aligned/centroid.model") as (_, local_path):
        return KeyedVectors.load(local_path)


def build_centroid_kvector(terms, counts, residuals, centroid_vectors):
    centroid = KeyedVectors(vector_size=VECTOR_SIZE)
    centroid.add_vectors(terms, centroid_vectors)
    centroid.fill_norms(force=True)

    term_variances = np.sum(
        (centroid.vectors - centroid.vectors.mean(axis=0)) ** 2, axis=1
    )

    r_squares = []
    r_squared_by_count = defaultdict(list)
    for i in range(len(counts)):
        r_squares.append(float(1 - (residuals[i] / term_variances[i])))
        r_squared_by_count[counts[i]].append(r_squares[-1])

    logger.info(
        "mean r_squared: %s - breakdown: %s",
        round(mean(r_squares), 5),
        [(int(k), len(v), round(mean(v), 3)) for k, v in sorted(r_squared_by_count.items())],
    )

    for i, term in enumerate(terms):
        centroid.set_vecattr(term, "count", int(counts[i]))
        centroid.set_vecattr(term, "disparity", float(residuals[i]))
        centroid.set_vecattr(term, "variance", float(term_variances[i]))
        centroid.set_vecattr(
            term, "r_squared", float(1 - (residuals[i] / term_variances[i]))
        )

    return centroid


def upload_kvector(session, kvector, s3_key):
    with tempfile.NamedTemporaryFile() as file:
        kvector.save(file.name, separately=[])
        upload_file(session, s3_key, file.name)


class S3Kvectors:

    def __init__(self, session, index):
        self.session = session
        self.book_index = index

    def load(self, subprefix):
        file_names = []
        kvector_stack = []

        for key, tmp_path in S3Loader(self.session).yield_s3_files(
            f"kvectors/{self.book_index}/{subprefix}/", ".model"
        ):
            kvector_stack.append(KeyedVectors.load(tmp_path))
            file_names.append(key.split("/")[-1])

        return file_names, kvector_stack

    def upload(self, subprefix, centroid, kvectors, file_names):
        for i in range(len(kvectors)):
            upload_kvector(
                self.session,
                kvectors[i],
                f"kvectors/{self.book_index}/{subprefix}/{file_names[i]}",
            )

        upload_kvector(
            self.session, centroid, f"kvectors/{self.book_index}/{subprefix}/centroid.model"
        )
