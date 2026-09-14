from __future__ import annotations

import math

import numpy as np

from app.search.constants import (
    BOOKS_WITH_TERM,
    BOOKS_WITH_TERM_IN_NEAREST_TERMS,
    MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS,
    NUM_COMPARATIVE_TERMS,
)
from app.search.schemas.semantic_drift import BookSimilarity, RelativeTermSimilarity
from app.search.services.semantic_drift.book_similarity_vectors import (
    BooksSimilarityCache,
)
from app.search.services.semantic_drift.utils import SearchExpr
from shared.commons import BookIndex


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

    book_terms_list = []
    book_similarities_centered_list = []
    book_terms_is_local_list = []

    for book_id in book_ids:
        book_similarity_vectors = books_similarity_cache.load_book(book_id, query)
        book_terms_list.append(book_similarity_vectors.terms)
        book_similarities_centered_list.append(
            book_similarity_vectors.centered_similarity_vectors
        )
        book_terms_is_local_list.append(book_similarity_vectors.is_local)

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

    mean_ranked = local_iloc[np.argsort(-similarity_mean[local_iloc], kind="stable")][
        :NUM_COMPARATIVE_TERMS
    ]
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


def get_book_similarities(
    books_similarity_cache: BooksSimilarityCache,
    expr: SearchExpr,
    terms: list[str],
    book_ids: list[BookIndex],
    selected_book_id: BookIndex | None = None,
) -> tuple[list[BookSimilarity], list[list[BookSimilarity]]]:

    books_similarity_vectors = books_similarity_cache.load_books(book_ids, expr)

    if selected_book_id is None:
        peers = books_similarity_vectors
    else:
        peers = [books_similarity_cache.load_book(selected_book_id, expr)]

    valid_books_similarity_vectors = [
        book
        for book in books_similarity_vectors
        if any(
            peer.book_id != book.book_id and book.is_comparable(peer) for peer in peers
        )
    ]

    local_mean_similarities = [
        book.local_mean_similarity for book in valid_books_similarity_vectors
    ]
    cross_book_mean_similarity = (
        float(np.mean(local_mean_similarities)) if local_mean_similarities else 0.0
    )

    books_term_cache = books_similarity_cache.books_term_cache
    expr_book_similarities = [
        BookSimilarity(
            book_id=book.book_id.source_id,
            similarity=cross_book_mean_similarity,
            # For a compound expression, the vocabulary volume behind the line.
            occurrences=sum(
                books_term_cache[book.book_id].get_term_count(term)
                for term in expr.terms
            ),
        )
        for book in valid_books_similarity_vectors
    ]

    term_book_similarities = []
    for term in terms:
        term_similarities = []
        for book in valid_books_similarity_vectors:
            similarity = book.get_centered_similarity(term)
            if similarity is None:
                continue
            term_similarities.append(
                BookSimilarity(
                    book_id=book.book_id.source_id,
                    similarity=similarity + cross_book_mean_similarity,
                    occurrences=books_term_cache[book.book_id].get_term_count(term),
                )
            )
        term_book_similarities.append(term_similarities)

    return expr_book_similarities, term_book_similarities
