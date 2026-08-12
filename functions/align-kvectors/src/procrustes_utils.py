import logging
import tempfile
from collections import defaultdict
from typing import NamedTuple

import numpy as np
from scipy.linalg import orthogonal_procrustes
from statistics import mean
from gensim.models import KeyedVectors

from shared.s3 import S3Loader, upload_file

MAX_ITERATIONS = 40
MIN_GRADIENT = 0.0001
VECTOR_SIZE = 200
EPS = 1e-6  # Epsilon: floor for denominators. indistinguishable at float16 precision

logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


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
    Return value of orthogonal_procrustes_alignment.

    R: orthogonal rotation matrix, shape (dim, dim). Apply to other
       matrices via `M @ R` to bring them into the target frame.
    book_disparity: scalar. SSE between rotated source and target,
       normalized by the total variance of target. Smaller is better;
       0 is perfect alignment. In raw Euclidean units.
    residuals: per-anchor squared distances after rotation. Shape
       (n_anchors,). Useful as a diagnostic — sorting descending shows
       which anchors the rotation struggled with. In raw Euclidean
       units (unbounded).
    """

    R: np.ndarray
    book_disparity: float
    residuals: np.ndarray


def orthogonal_procrustes_alignment(
    source: np.ndarray,
    target: np.ndarray,
) -> ProcrustesResult:
    """
    Find the orthogonal rotation R that aligns source to target.

    Plain orthogonal Procrustes on the raw input matrices, no weighting,
    no normalization. Equivalent to scipy's orthogonal_procrustes plus a
    normalized book_disparity computation. Used for within-book seed
    alignment, where downstream r_squared depends on raw Euclidean
    distances. The raw (un-normalized) inputs preserve the implicit
    norm-based weighting that biases the rotation toward higher-norm
    (typically more content-bearing) terms.

    Parameters
    ----------
    source : (n_anchors, dim) array
        Source anchor matrix. The rotation maps source toward target.
        Not mutated.
    target : (n_anchors, dim) array
        Target anchor matrix. Row i pairs with row i of source.

    Returns
    -------
    ProcrustesResult(R, book_disparity, residuals) — see class docstring.
    """
    if source.shape != target.shape:
        raise ValueError(f"source {source.shape} and target {target.shape} must match")
    if source.ndim != 2:
        raise ValueError(f"expected 2D arrays, got {source.ndim}D")
    n_anchors, _ = source.shape
    if n_anchors == 0:
        raise ValueError("need at least one anchor")

    R, _ = orthogonal_procrustes(source, target)

    rotated = source @ R
    diff = target - rotated
    residuals = np.sum(diff * diff, axis=1)

    sse = float(np.sum(residuals))
    target_centered = target - target.mean(axis=0)
    target_var = float(np.sum(target_centered**2))
    book_disparity = sse / target_var if target_var > EPS else float("inf")

    return ProcrustesResult(R=R, book_disparity=book_disparity, residuals=residuals)


def normalized_disparity_alignment(terms, kvector, centroid_vectors):
    term_indices, present_terms = zip(
        *[(i, term) for i, term in enumerate(terms) if term in kvector]
    )
    term_indices = list(term_indices)

    centroid_matrix = centroid_vectors[term_indices].astype(np.float32)
    kv_matrix = np.stack([kvector[term] for term in present_terms]).astype(np.float32)

    result = orthogonal_procrustes_alignment(kv_matrix, centroid_matrix)

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
    max_iterations=MAX_ITERATIONS,
    min_gradient=MIN_GRADIENT,
):
    # Generalized Procrustes Analysis (Gower, 1975)
    prev_book_disparity = float("inf")
    for iteration in range(max_iterations):
        normalized_book_disparities = []
        stack_residuals = []
        centroid_vectors = compute_centroid_vectors(terms, kvector_stack)

        n_terms = len(terms)
        for kvector in kvector_stack:
            result, term_indices = normalized_disparity_alignment(
                terms, kvector, centroid_vectors
            )
            normalized_book_disparities.append(result.book_disparity)
            full_residuals = np.full(n_terms, np.nan)
            full_residuals[term_indices] = result.residuals
            stack_residuals.append(full_residuals)
        current_book_disparity = mean(normalized_book_disparities)

        if (prev_book_disparity - current_book_disparity) <= min_gradient:
            combined_residuals = np.nanmean(stack_residuals, axis=0)
            return (
                centroid_vectors,
                combined_residuals,
                current_book_disparity,
                iteration + 1,
            )
        prev_book_disparity = current_book_disparity

    raise Exception("Kvectors not aligned")


def build_centroid_kvector(terms, counts, residuals, centroid_vectors):
    centroid = KeyedVectors(vector_size=VECTOR_SIZE)
    centroid.add_vectors(terms, centroid_vectors)
    centroid.fill_norms(force=True)

    term_variances = np.sum(
        (centroid.vectors - centroid.vectors.mean(axis=0)) ** 2, axis=1
    )

    # prevent division by near zero
    r_squares = 1.0 - (np.asarray(residuals) / np.maximum(term_variances, EPS))

    r_squared_by_count = defaultdict(list)
    for i in range(len(counts)):
        r_squared_by_count[counts[i]].append(float(r_squares[i]))

    logger.info(
        "mean r_squared: %s - breakdown: %s",
        round(float(np.mean(r_squares)), 5),
        [
            (int(k), len(v), round(mean(v), 3))
            for k, v in sorted(r_squared_by_count.items())
        ],
    )

    for i, term in enumerate(terms):
        centroid.set_vecattr(term, "count", int(counts[i]))
        centroid.set_vecattr(term, "disparity", float(residuals[i]))
        centroid.set_vecattr(term, "variance", float(term_variances[i]))
        centroid.set_vecattr(term, "r_squared", float(r_squares[i]))

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
            self.session,
            centroid,
            f"kvectors/{self.book_index}/{subprefix}/centroid.model",
        )
