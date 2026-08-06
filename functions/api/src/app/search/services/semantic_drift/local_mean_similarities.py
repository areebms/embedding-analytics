from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from app.search.constants import (
    MIN_MATCHING_BOOKS,
    NEAREST_TERM_COUNT,
    NUM_SIMILAR_TERMS,
    T_CRIT_95,
)
from app.search.errors import NoLocalNearestTermsError
from app.search.schemas.semantic_drift import BookLocalMeanSimilarity, TermStats
from app.search.services.semantic_drift.book_similarity_vectors import (
    BooksSimilarityCache,
)
from app.search.services.semantic_drift.utils import SearchExpr
from shared.commons import BookIndex


class TrendFit(NamedTuple):

    slope: np.ndarray
    r_squared: np.ndarray

    @property
    def score(self) -> np.ndarray:
        return self.slope * self.r_squared


def get_trend_fits(
    group_iloc: np.ndarray,
    years: np.ndarray,
    similarities: np.ndarray,
    n_groups: int,
) -> TrendFit:
    """Slope and r-squared of similarity-against-year, one pair per group."""

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

    return TrendFit(slope, r_squared)


def get_nearest_terms(
    books_similarity_cache: BooksSimilarityCache,
    book_ids: list[BookIndex],
    query: SearchExpr,
    book_years: dict[BookIndex, int],
    sort: Literal["mean_similarity", "slope", "variance"] = "mean_similarity",
    *,
    selected_book_id: BookIndex | None = None,
) -> list[TermStats]:

    books_vectors = [
        books_similarity_cache.load_book(book_id, query) for book_id in book_ids
    ]

    all_terms = np.concatenate([book_vectors.terms for book_vectors in books_vectors])
    all_similarities = np.concatenate(
        [book_vectors.similarity_vectors.mean(axis=0) for book_vectors in books_vectors]
    )

    # Every book's terms are already sorted, so the concatenation is one run per
    # book, which a merge sort walks at about twice np.unique's quicksort pace.
    order = np.argsort(all_terms, kind="stable")
    sorted_terms = all_terms[order]
    is_first = np.ones(len(all_terms), dtype=bool)
    np.not_equal(sorted_terms[1:], sorted_terms[:-1], out=is_first[1:])

    terms = sorted_terms[is_first]
    term_iloc = np.empty(len(all_terms), dtype=np.intp)
    term_iloc[order] = np.cumsum(is_first) - 1

    book_count = np.bincount(term_iloc)  # Terms are unique within a book
    mean_similarity = np.bincount(term_iloc, weights=all_similarities) / book_count

    # How much the books disagree about where the term sits, not about how far
    # away it is: a term holding one cosine throughout still spreads if the
    # terms nearest it move. Two-pass, because summing the squares
    # themselves leaves float32 rounding behind instead. A term only one book
    # carries spreads not at all.
    deviation = all_similarities - mean_similarity[term_iloc]
    variance = np.bincount(term_iloc, weights=deviation**2) / np.maximum(
        book_count - 1, 1
    )

    is_valid = (book_count >= MIN_MATCHING_BOOKS) & ~np.isin(terms, query.terms)
    if selected_book_id is not None:
        # Both arrays are sorted, so this beats np.isin. The "" sentinel catches
        # terms sorting past the book's last one, and no real term equals it.
        selected_terms = books_similarity_cache.books_term_cache[selected_book_id].terms
        position = np.searchsorted(selected_terms, terms)
        is_valid &= np.append(selected_terms, "")[position] == terms
    valid_iloc = np.flatnonzero(is_valid)

    # Every sort ranks the same relevance-floored pool and differs only in the
    # key. Sorted so that ties break alphabetically rather than by relevance rank.
    most_similar_iloc = valid_iloc[
        np.argsort(-mean_similarity[valid_iloc], kind="stable")
    ]
    pool_iloc = np.sort(most_similar_iloc[:NUM_SIMILAR_TERMS])

    # Fit the pool against publication year, in pool order. Undated books
    # contribute no rows, and an unfittable term scores zero.
    group = np.full(len(terms), -1)
    group[pool_iloc] = np.arange(len(pool_iloc))
    row_group = group[term_iloc]

    # Undated books carry NaN, which drops their rows out of the fit below.
    years = [
        float(book_years.get(book_vectors.book_id, np.nan))
        for book_vectors in books_vectors
    ]
    n_terms_per_book = [len(book_vectors.terms) for book_vectors in books_vectors]
    all_years = np.repeat(years, n_terms_per_book)

    is_fit_row = (row_group >= 0) & ~np.isnan(all_years)
    trend = get_trend_fits(
        row_group[is_fit_row],
        all_years[is_fit_row],
        all_similarities[is_fit_row],
        len(pool_iloc),
    )

    if sort == "mean_similarity":
        rank_by = mean_similarity[pool_iloc]
    elif sort == "slope":
        rank_by = np.abs(trend.score)
    elif sort == "variance":
        rank_by = variance[pool_iloc]
    else:
        raise ValueError(f"unknown sort {sort!r}")

    ranked = np.argsort(-rank_by, kind="stable")[:NEAREST_TERM_COUNT]
    return [
        TermStats(
            term=str(terms[iloc]),
            mean_similarity=float(mean_similarity[iloc]),
            n_books_with_term=int(book_count[iloc]),
            slope=float(trend.slope[rank]),
            r_squared=float(trend.r_squared[rank]),
        )
        for rank, iloc in zip(ranked, pool_iloc[ranked])
    ]


def get_local_mean_similarity_per_book(
    book_id: BookIndex,
    local_similarities_per_peer: list[np.ndarray],
    count: int,
):

    n_seeds = min(len(similarity) for similarity in local_similarities_per_peer)
    mean_local_similarities_per_seed = np.mean(
        [similarity[:n_seeds] for similarity in local_similarities_per_peer], axis=0
    )
    mean_local_similarity = float(np.mean(mean_local_similarities_per_seed))

    if n_seeds > 1:
        t_crit = T_CRIT_95[n_seeds - 1] if n_seeds - 1 < len(T_CRIT_95) else 1.96
        sd = float(np.std(mean_local_similarities_per_seed, ddof=1))
        ci_half = t_crit * sd / np.sqrt(n_seeds)
    else:
        sd = ci_half = 0.0

    return BookLocalMeanSimilarity(
        book_id=book_id.source_id,
        similarity=mean_local_similarity,
        similarity_ci=(
            mean_local_similarity - ci_half,
            mean_local_similarity + ci_half,
        ),
        similarity_sd=sd,
        count=count,
        n_seeds=n_seeds,
        n_books=len(local_similarities_per_peer),
    )


def get_local_mean_similarities(
    books_similarity_cache: BooksSimilarityCache,
    expr: SearchExpr,
    book_ids: list[BookIndex],
    selected_book_id: BookIndex | None = None,
) -> list[BookLocalMeanSimilarity]:

    books_similarity_vectors = books_similarity_cache.load_books(book_ids, expr)

    if selected_book_id is None:
        peers = books_similarity_vectors
    else:
        peers = [books_similarity_cache.load_book(selected_book_id, expr)]

    books_data: list[BookLocalMeanSimilarity] = []
    for book_similarity_vectors in books_similarity_vectors:
        local_similarities_per_peer = []
        for peer in peers:
            if peer.book_id == book_similarity_vectors.book_id:
                continue
            try:
                local_similarity = book_similarity_vectors.get_local_cosine_similarity(
                    peer
                )
            except NoLocalNearestTermsError:
                continue
            local_similarities_per_peer.append(local_similarity)

        if not local_similarities_per_peer:
            continue

        # For a compound expression, the vocabulary volume behind the line.
        book_terms = books_similarity_cache.books_term_cache[
            book_similarity_vectors.book_id
        ]
        count = sum(book_terms.get_term_count(term) for term in expr.terms)

        books_data.append(
            get_local_mean_similarity_per_book(
                book_similarity_vectors.book_id, local_similarities_per_peer, count
            )
        )

    return books_data
