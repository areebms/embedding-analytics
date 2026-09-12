from __future__ import annotations

import math

import numpy as np

from app.search.constants import (
    BOOKS_WITH_TERM,
    BOOKS_WITH_TERM_IN_NEAREST_TERMS,
    MAX_RANK_FOR_TERM_SELECTION,
    MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS,
    NUM_LOCAL_NEAREST_TERMS,
    NUM_COMPARATIVE_TERMS,
)
from app.search.errors import NoLocalNearestTermsError
from app.search.schemas.semantic_drift import (
    SecondOrderSimilarity,
    MeanSecondOrderSimilarity,
    RelativeTermSimilarity,
)
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
) -> list[RelativeTermSimilarity]:

    book_similarities_list = []
    book_terms_list = []
    book_similarities_centered_list = []
    book_terms_is_local_list = []

    for book_id in book_ids:
        book_similarity_vectors = books_similarity_cache.load_book(book_id, query)
        book_similarities = book_similarity_vectors.similarity_vectors
        book_similarities_list.append(book_similarities)
        book_terms_list.append(book_similarity_vectors.terms)
        book_similarities_centered_list.append(
            center_locally(book_similarities, NUM_LOCAL_NEAREST_TERMS)
        )
        book_terms_is_local_list.append(
            get_is_local(book_similarities, MAX_RANK_FOR_TERM_SELECTION)
        )

    terms, term_iloc = get_unique_terms(np.concat(book_terms_list))
    n_books_in = np.bincount(term_iloc)  # Terms are unique within a book

    is_relevant_to_corpus = (
        n_books_in >= math.ceil(BOOKS_WITH_TERM * len(book_ids))
    ) & ~np.isin(terms, query.terms)
    if selected_book_id is not None:
        selected_terms = books_similarity_cache.books_term_cache[selected_book_id].terms
        position = np.searchsorted(selected_terms, terms)
        is_relevant_to_corpus &= np.append(selected_terms, "")[position] == terms

    is_local = np.concat(book_terms_is_local_list)
    n_books_local = np.bincount(term_iloc, weights=is_local).astype(np.intp)

    all_similarities = np.concat(book_similarities_centered_list)
    similarity_mean = np.bincount(term_iloc, weights=all_similarities) / n_books_in

    similarity_variance = np.bincount(
        term_iloc, weights=(all_similarities - similarity_mean[term_iloc]) ** 2
    ) / np.maximum(n_books_in - 1, 1)

    min_books_in_nearest_terms = np.maximum(
        MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS,
        np.ceil(BOOKS_WITH_TERM_IN_NEAREST_TERMS * n_books_in),
    )
    local_iloc = np.flatnonzero(
        is_relevant_to_corpus & (n_books_local >= min_books_in_nearest_terms)
    )

    mean_ranked = local_iloc[
        np.argsort(-similarity_mean[local_iloc], kind="stable")
    ][:NUM_COMPARATIVE_TERMS]
    variance_ranked = local_iloc[
        np.argsort(-similarity_variance[local_iloc], kind="stable")
    ][:NUM_COMPARATIVE_TERMS]

    return [
        RelativeTermSimilarity(
            term=str(terms[iloc]),
            similarity_mean=float(similarity_mean[iloc]),
            similarity_variance=float(similarity_variance[iloc]),
            n_books_in=int(n_books_in[iloc]),
            n_books_local_in=int(n_books_local[iloc]),
        )
        for iloc in np.union1d(mean_ranked, variance_ranked)
    ]


def get_mean_local_similarity_per_book(
    book_id: BookIndex,
    local_similarities_per_peer: list[np.ndarray],
    occurrences: int,
    *,
    against_corpus: bool,
):

    mean_local_similarity = float(np.mean(local_similarities_per_peer))

    if against_corpus:
        return MeanSecondOrderSimilarity(
            book_id=book_id.source_id,
            mean_similarity=mean_local_similarity,
            occurrences=occurrences,
            n_books=len(local_similarities_per_peer),
        )
    return SecondOrderSimilarity(
        book_id=book_id.source_id,
        similarity=mean_local_similarity,
        occurrences=occurrences,
    )


def get_mean_local_similarities(
    books_similarity_cache: BooksSimilarityCache,
    expr: SearchExpr,
    book_ids: list[BookIndex],
    selected_book_id: BookIndex | None = None,
) -> list[SecondOrderSimilarity] | list[MeanSecondOrderSimilarity]:

    books_similarity_vectors = books_similarity_cache.load_books(book_ids, expr)

    if selected_book_id is None:
        peers = books_similarity_vectors
    else:
        peers = [books_similarity_cache.load_book(selected_book_id, expr)]

    books_data = []
    for book_similarity_vectors in books_similarity_vectors:
        local_similarities_per_peer = []
        for peer in peers:
            if peer.book_id == book_similarity_vectors.book_id:
                continue
            try:
                local_similarity = book_similarity_vectors.get_local_similarity(peer)
            except NoLocalNearestTermsError:
                continue
            local_similarities_per_peer.append(local_similarity)

        if not local_similarities_per_peer:
            continue

        # For a compound expression, the vocabulary volume behind the line.
        book_terms = books_similarity_cache.books_term_cache[
            book_similarity_vectors.book_id
        ]
        occurrences = sum(book_terms.get_term_count(term) for term in expr.terms)

        books_data.append(
            get_mean_local_similarity_per_book(
                book_similarity_vectors.book_id,
                local_similarities_per_peer,
                occurrences,
                against_corpus=selected_book_id is None,
            )
        )

    return books_data
