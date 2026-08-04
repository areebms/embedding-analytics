from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from app.search.constants import (
    BARE_TERM_PATTERN,
    MIN_MATCHING_BOOKS,
    NUM_SIMILAR_TERMS,
    NEAREST_TERM_COUNT,
    T_CRIT_95,
)
from app.search.errors import MissingTermsError, NoLocalNearestTermsError
from app.search.schemas.semantic_drift import BookLocalMeanSimilarity, OpNode, TermNode
from app.search.services.utils import extract_terms, serialize_expression
from shared.commons import BookIndex


class LocalCosineSimilarity(NamedTuple):

    similarity: np.ndarray
    n_local_terms: int


class SearchExpr(NamedTuple):

    tree: TermNode | OpNode
    terms: list[str]
    serialized: str

    @classmethod
    def from_query(cls, query: TermNode | OpNode | str) -> SearchExpr:
        if isinstance(query, str):

            if not BARE_TERM_PATTERN.fullmatch(query):
                raise ValueError(f"expected a single bare term, got {query!r}")
            query = TermNode(term=query)
        return cls(
            query,
            extract_terms(query),
            serialize_expression(query, strip_outer=True),
        )


def get_trend_scores(
    group_iloc: np.ndarray,
    years: np.ndarray,
    similarities: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """`slope * r_squared` of similarity-against-year, one score per group."""

    def group_sum(weights: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(group_iloc, weights=weights, minlength=n_groups)

    x, y = years, similarities  # the usual least-squares naming.
    n, sum_x, sum_y = group_sum(), group_sum(x), group_sum(y)

    ss_xy = n * group_sum(x * y) - sum_x * sum_y
    ss_xx = n * group_sum(x * x) - sum_x * sum_x
    ss_yy = n * group_sum(y * y) - sum_y * sum_y

    is_fit = (ss_xx != 0) & (ss_yy != 0)
    slope = np.zeros(n_groups)
    r_squared = np.zeros(n_groups)
    slope[is_fit] = ss_xy[is_fit] / ss_xx[is_fit]
    r_squared[is_fit] = ss_xy[is_fit] ** 2 / (ss_xx[is_fit] * ss_yy[is_fit])

    return slope * r_squared


def get_nearest_terms(
    books_similarity_cache,
    book_ids: list[BookIndex],
    query: SearchExpr,
    book_years: dict[BookIndex, int],
    sort: Literal["mean_similarity", "slope"] = "mean_similarity",
    *,
    selected_book_id: BookIndex | None = None,
):

    books_vectors = [
        books_similarity_cache.load_book(book_id, query) for book_id in book_ids
    ]

    all_terms = np.concatenate([book_vectors.terms for book_vectors in books_vectors])
    all_similarities = np.concatenate(
        [book_vectors.similarity_vectors.mean(axis=0) for book_vectors in books_vectors]
    )

    terms, term_iloc = np.unique(all_terms, return_inverse=True)
    total_similarity = np.bincount(term_iloc, weights=all_similarities)
    book_count = np.bincount(term_iloc)  # Terms are unique within a book

    is_valid = (book_count >= MIN_MATCHING_BOOKS) & ~np.isin(terms, query.terms)
    if selected_book_id is not None:
        selected_terms = books_similarity_cache.books_term_cache[selected_book_id].terms
        if len(selected_terms):
            position = np.minimum(
                np.searchsorted(selected_terms, terms), len(selected_terms) - 1
            )
            is_valid &= selected_terms[position] == terms
        else:
            is_valid[:] = False
    valid_iloc = np.flatnonzero(is_valid)

    mean_similarity = total_similarity[valid_iloc] / book_count[valid_iloc]
    most_similar_iloc = valid_iloc[np.argsort(-mean_similarity, kind="stable")]

    if sort == "mean_similarity":
        ranked_iloc = most_similar_iloc[:NEAREST_TERM_COUNT]
    elif sort == "slope":
        pool_iloc = np.sort(most_similar_iloc[:NUM_SIMILAR_TERMS])

        pool_rank = np.full(len(terms), -1)
        pool_rank[pool_iloc] = np.arange(len(pool_iloc))
        row_group = pool_rank[term_iloc]
        in_pool = row_group >= 0

        years = [book_years.get(book_vectors.book_id) for book_vectors in books_vectors]
        term_counts = [len(book_vectors.terms) for book_vectors in books_vectors]
        all_years = np.repeat([float(year or 0) for year in years], term_counts)
        is_dated = np.repeat([year is not None for year in years], term_counts)

        is_fit_row = in_pool & is_dated
        trend_score = get_trend_scores(
            row_group[is_fit_row],
            all_years[is_fit_row],
            all_similarities[is_fit_row],
            len(pool_iloc),
        )
        ranked_iloc = pool_iloc[
            np.argsort(-np.abs(trend_score), kind="stable")[:NEAREST_TERM_COUNT]
        ]
    else:
        raise ValueError(f"unknown sort {sort!r}")

    return [str(term) for term in terms[ranked_iloc]]


def get_local_mean_similarity_per_book(
    book_id: BookIndex,
    local_similarities_per_peer: list[np.ndarray],
    shared_term_counts_per_peer: list[int],
):

    n_seeds = min(len(similarity) for similarity in local_similarities_per_peer)
    mean_local_similarities_per_seed = np.mean(
        [similarity[:n_seeds] for similarity in local_similarities_per_peer], axis=0
    )
    mean_local_similarity = float(np.mean(mean_local_similarities_per_seed))

    if n_seeds > 1:
        t_crit = T_CRIT_95[n_seeds - 1] if n_seeds - 1 < len(T_CRIT_95) else 1.96
        ci_half = float(
            t_crit * np.std(mean_local_similarities_per_seed, ddof=1) / np.sqrt(n_seeds)
        )
    else:
        ci_half = 0.0

    return BookLocalMeanSimilarity(
        book_id=book_id.source_id,
        similarity=mean_local_similarity,
        similarity_ci=(
            mean_local_similarity - ci_half,
            mean_local_similarity + ci_half,
        ),
        n_seeds=n_seeds,
        min_local_terms=int(np.min(shared_term_counts_per_peer)),
        n_books=len(local_similarities_per_peer),
    )


def get_local_mean_similarities(
    books_similarity_cache,
    expr: SearchExpr,
    book_ids: list[BookIndex],
    selected_book_id: BookIndex | None = None,
) -> list[BookLocalMeanSimilarity]:

    books_similarity_vectors = []
    for book_id in book_ids:
        try:
            books_similarity_vectors.append(
                books_similarity_cache.load_book(book_id, expr)
            )
        except MissingTermsError:
            continue

    if selected_book_id is None:
        peers = books_similarity_vectors
    else:
        peers = [books_similarity_cache.load_book(selected_book_id, expr)]

    books_data: list[BookLocalMeanSimilarity] = []
    for book_similarity_vectors in books_similarity_vectors:
        local_similarities_per_peer = []
        shared_term_counts_per_peer = []
        for peer in peers:
            if peer.book_id == book_similarity_vectors.book_id:
                continue
            try:
                local_similarity, n_terms = (
                    book_similarity_vectors.get_local_cosine_similarity(peer)
                )
            except NoLocalNearestTermsError:
                continue
            local_similarities_per_peer.append(local_similarity)
            shared_term_counts_per_peer.append(n_terms)

        if not local_similarities_per_peer:
            continue

        books_data.append(
            get_local_mean_similarity_per_book(
                book_similarity_vectors.book_id,
                local_similarities_per_peer,
                shared_term_counts_per_peer,
            )
        )

    return books_data
