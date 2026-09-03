from collections import Counter
from collections.abc import Sequence
from itertools import chain

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from constants import (
    ALPHA,
    CDS,
    GAMMA,
    MIN_COUNT,
    SOLVER,
    V0_SEED,
    VECTOR_SIZE,
    WINDOW,
)

Passages = Sequence[Sequence[str]]


def build_vocab(
    passages: Passages,
) -> tuple[list[str], dict[str, int], Counter[str]]:
    counts = Counter(token for passage in passages for token in passage)
    kept = [term for term, count in counts.items() if count >= MIN_COUNT]
    kept.sort(key=lambda term: (-counts[term], term))
    return kept, {term: i for i, term in enumerate(kept)}, counts


def term_cooccurrence_in_window(
    passages: Passages, term_indexes: dict[str, int]
) -> sp.csr_matrix:
    """term co-occurrence sparse matrix,
    entry (i, j) is how many times term j appeared within WINDOW."""

    passage_term_indexes = [
        [term_indexes[token] for token in passage if token in term_indexes]
        for passage in passages
    ]  # turns each passage of token strings into a list of vocabulary ids
    ids = np.fromiter(chain.from_iterable(passage_term_indexes), dtype=np.int32)
    term_passages = np.repeat(
        np.arange(len(passage_term_indexes)),
        [len(passage) for passage in passage_term_indexes],
    )

    rows, cols = [], []
    for offset in range(1, min(WINDOW + 1, len(ids))):
        same_passage = term_passages[:-offset] == term_passages[offset:]
        rows.append(ids[:-offset][same_passage])
        cols.append(ids[offset:][same_passage])

    row = np.concatenate(rows or [np.empty(0, np.int32)])
    col = np.concatenate(cols or [np.empty(0, np.int32)])

    size = len(term_indexes)
    counts = sp.coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, col)), shape=(size, size)
    ).tocsr()
    return counts + counts.T


def positive_pointwise_mutual_info(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Positive PMI with context distribution smoothing."""
    total = counts.sum()
    if total == 0:
        return counts.copy()

    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    col_sums = np.asarray(counts.sum(axis=0)).ravel()

    # smoothed context marginal
    col_smooth = col_sums**CDS
    col_total = col_smooth.sum()

    coo = counts.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        term_surprisal = np.log(total) - np.log(row_sums)
        context_surprisal = np.log(col_total) - np.log(col_smooth)
        pair_surprisal = np.log(total) - np.log(coo.data)
        vals = (
            term_surprisal[coo.row]
            + context_surprisal[coo.col]
            - pair_surprisal
            - ALPHA
        )
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    return sp.coo_matrix(
        (vals[vals > 0], (coo.row[vals > 0], coo.col[vals > 0])), shape=counts.shape
    ).tocsr()


def truncated_svd(matrix: sp.csr_matrix) -> np.ndarray:
    """Deterministic truncated SVD."""
    v0 = np.random.default_rng(V0_SEED).standard_normal(matrix.shape[0])
    vectors, latent_dimension_strength, _ = svds(
        matrix, k=VECTOR_SIZE, v0=v0, solver=SOLVER
    )

    order = np.argsort(-latent_dimension_strength)
    vectors, latent_dimension_strength = (
        vectors[:, order],
        latent_dimension_strength[order],
    )

    return vectors * (latent_dimension_strength**GAMMA)
