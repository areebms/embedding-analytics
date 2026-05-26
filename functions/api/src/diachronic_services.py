import hashlib
import logging
from typing import Optional

import numpy as np
from fastapi_cache import FastAPICache

from describe_services import serialize_expression
from schemas import OpNode, TermNode
from services import evaluate_tree, get_confidence_intervals
from shared.aws import TermTable


logger = logging.getLogger(__name__)

CENTROID_CACHE_TTL = 300  # seconds


def _centroid_cache_key(tree: TermNode | OpNode, ref_book_ids: list[int]) -> str:
    """Stable key for caching centroids by (expression, sorted ref book ids)."""
    canonical = serialize_expression(tree, strip_outer=True)
    payload = f"{canonical}|{sorted(ref_book_ids)}"
    return f"centroid:{hashlib.sha1(payload.encode()).hexdigest()}"


def _get_redis():
    """Return the Redis client from the fastapi-cache backend, or None.

    Falls back to None when Redis isn't configured (e.g. local dev), in which
    case the centroid is computed inline on every request.
    """
    try:
        backend = FastAPICache.get_backend()
    except (AssertionError, AttributeError):
        return None
    return getattr(backend, "redis", None)


async def get_reference_centroid(
    tree: TermNode | OpNode,
    ref_book_ids: list[int],
    table: TermTable,
) -> np.ndarray:
    """Mean of per-book mean concept vectors for the tree, normalized to unit.

    Returns a (dim,) array. Books that don't contain all required terms are
    silently skipped. Raises ValueError if no ref book has the expression or
    if the resulting centroid has zero norm.

    Cached in Redis for CENTROID_CACHE_TTL seconds when Redis is configured.
    """
    cache_key = _centroid_cache_key(tree, ref_book_ids)
    redis = _get_redis()

    if redis is not None:
        cached = await redis.get(cache_key)
        if cached is not None:
            return np.frombuffer(cached, dtype=np.float32).copy()

    book_means = []
    for ref_id in ref_book_ids:
        platform_data = f"gutenberg-{ref_id}"
        try:
            vectors = evaluate_tree(tree, table, platform_data)
        except ValueError:
            continue
        # vectors is already (n_seeds, dim) and normalized per row.
        book_means.append(np.mean(vectors, axis=0))

    if not book_means:
        raise ValueError(
            "No reference books contain all terms in the expression"
        )

    centroid = np.mean(np.stack(book_means), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-12:
        raise ValueError("Reference centroid has zero norm")
    centroid = (centroid / norm).astype(np.float32)

    if redis is not None:
        await redis.set(cache_key, centroid.tobytes(), ex=CENTROID_CACHE_TTL)

    return centroid


def compute_diachronic_similarity(
    tree: TermNode | OpNode,
    book_id: int,
    reference: np.ndarray,
    table: TermTable,
) -> Optional[dict]:
    """Per-seed cosine similarity between one book and a fixed reference vector.

    Returns {id, similarity, similarity_ci, n} or None if the book doesn't
    contain all terms in the expression.

    The reference is treated as a single point (shape (dim,)) and tiled to
    match the queried book's seed count. This gives n samples whose variance
    reflects training noise in the queried book only, producing tighter and
    more interpretable confidence intervals than paired-seed comparison.
    """
    platform_data = f"gutenberg-{book_id}"
    try:
        queried_vectors = evaluate_tree(tree, table, platform_data)
    except ValueError:
        return None

    n = len(queried_vectors)
    ref_tiled = np.tile(reference.astype(np.float64), (n, 1))

    similarity, ci_half = get_confidence_intervals(queried_vectors, ref_tiled)

    return {
        "id": book_id,
        "similarity": similarity,
        "similarity_ci": (similarity - ci_half, similarity + ci_half),
        "n": n,
    }
