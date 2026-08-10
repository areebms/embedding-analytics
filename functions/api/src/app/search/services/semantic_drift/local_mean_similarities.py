from __future__ import annotations

import numpy as np

from app.search.constants import (
    MAX_RANK_FOR_STABLE_TERM,
    MAX_RANK_FOR_UNSTABLE_TERM,
    MIN_BOOKS_WITH_TERM,
    MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS,
    MIN_BOOKS_WITH_UNSTABLE_TERM_AS_TOP_50,
    NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING,
    NUM_RELEVANT_TERMS_FOR_INSTABILITY,
    NUM_TERMS_KEPT,
    T_CRIT_95,
)
from app.search.errors import NoLocalNearestTermsError
from app.search.schemas.semantic_drift import BookLocalMeanSimilarity, TermStats
from app.search.services.semantic_drift.book_similarity_vectors import (
    BooksSimilarityCache,
)
from app.search.services.semantic_drift.utils import SearchExpr
from shared.commons import BookIndex


def get_n_highest_similarities(similarities: np.ndarray, n: int) -> np.ndarray:

    if not len(similarities):
        return similarities

    n = min(n, len(similarities))

    return np.partition(similarities, -n)[-n:]


def center_locally(similarities: np.ndarray, n: int) -> np.ndarray:

    highest_similarities = get_n_highest_similarities(similarities, n)

    if not len(highest_similarities):
        return similarities

    return similarities - highest_similarities.mean()


def get_is_local(similarities: np.ndarray, n: int) -> np.ndarray:

    highest_similarities = get_n_highest_similarities(similarities, n)

    if not len(highest_similarities):
        return np.zeros(0, dtype=bool)

    return similarities >= highest_similarities.min()


def get_unique_terms(all_terms):
    # faster equivalent of np.unique(all_terms, return_inverse=True)
    order = np.argsort(all_terms, kind="stable")
    sorted_terms = all_terms[order]
    is_first = np.ones(len(all_terms), dtype=bool)
    np.not_equal(sorted_terms[1:], sorted_terms[:-1], out=is_first[1:])

    terms = sorted_terms[is_first]
    term_iloc = np.empty(len(all_terms), dtype=np.intp)
    term_iloc[order] = np.cumsum(is_first) - 1
    return terms, term_iloc


def get_comparative_terms(
    books_similarity_cache: BooksSimilarityCache,
    book_ids: list[BookIndex],
    query: SearchExpr,
    *,
    selected_book_id: BookIndex | None = None,
) -> list[TermStats]:

    book_similarities_list = []
    book_terms_list = []
    book_similarities_centered_list = []
    book_terms_is_in_top_50_list = []
    book_terms_is_in_top_100_list = []

    for book_id in book_ids:
        book_similarity_vectors = books_similarity_cache.load_book(book_id, query)
        book_similarities = book_similarity_vectors.mean_similarities
        book_similarities_list.append(book_similarities)
        book_terms_list.append(book_similarity_vectors.terms)
        book_similarities_centered_list.append(
            center_locally(
                book_similarities, NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING
            )
        )
        book_terms_is_in_top_50_list.append(
            get_is_local(book_similarities, MAX_RANK_FOR_STABLE_TERM)
        )
        book_terms_is_in_top_100_list.append(
            get_is_local(book_similarities, MAX_RANK_FOR_UNSTABLE_TERM)
        )

    terms, term_iloc = get_unique_terms(np.concat(book_terms_list))
    n_books_in = np.bincount(term_iloc)  # Terms are unique within a book

    is_relevant_to_corpus = (n_books_in >= MIN_BOOKS_WITH_TERM) & ~np.isin(
        terms, query.terms
    )
    if selected_book_id is not None:
        selected_terms = books_similarity_cache.books_term_cache[selected_book_id].terms
        position = np.searchsorted(selected_terms, terms)
        is_relevant_to_corpus &= np.append(selected_terms, "")[position] == terms

    is_in_top_50 = np.concat(book_terms_is_in_top_50_list)
    is_in_top_100 = np.concat(book_terms_is_in_top_100_list)

    n_books_50 = np.bincount(term_iloc, weights=is_in_top_50).astype(np.intp)
    n_books_100 = np.bincount(term_iloc, weights=is_in_top_100).astype(np.intp)

    all_similarities = np.concat(book_similarities_centered_list)
    mean_similarities = np.bincount(term_iloc, weights=all_similarities) / n_books_in

    variance = np.bincount(
        term_iloc, weights=(all_similarities - mean_similarities[term_iloc]) ** 2
    ) / np.maximum(n_books_in - 1, 1)

    def make_term_stats(iloc: np.intp) -> TermStats:
        return TermStats(
            term=str(terms[iloc]),
            stability=float(mean_similarities[iloc]),
            instability=float(variance[iloc]),
            n_books_in=int(n_books_in[iloc]),
            n_books_as_top50=int(n_books_50[iloc]),
            n_books_as_top100=int(n_books_100[iloc]),
        )

    mean_iloc = np.flatnonzero(
        is_relevant_to_corpus & (n_books_50 >= MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS)
    )
    mean_ranked = mean_iloc[np.argsort(-mean_similarities[mean_iloc], kind="stable")][
        :NUM_TERMS_KEPT
    ]

    variance_iloc = np.flatnonzero(
        is_relevant_to_corpus
        & (n_books_100 >= MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS)
        & (n_books_50 >= MIN_BOOKS_WITH_UNSTABLE_TERM_AS_TOP_50)
    )
    most_similar_iloc = variance_iloc[
        np.argsort(-mean_similarities[variance_iloc], kind="stable")
    ]
    pool_iloc = np.sort(most_similar_iloc[:NUM_RELEVANT_TERMS_FOR_INSTABILITY])
    variance_ranked = pool_iloc[
        np.argsort(-variance[pool_iloc], kind="stable")[:NUM_TERMS_KEPT]
    ]

    return [make_term_stats(iloc) for iloc in np.union1d(mean_ranked, variance_ranked)]


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
        mean_similarity=mean_local_similarity,
        similarity_ci=(
            mean_local_similarity - ci_half,
            mean_local_similarity + ci_half,
        ),
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
