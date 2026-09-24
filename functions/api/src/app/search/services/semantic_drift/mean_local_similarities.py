from __future__ import annotations

import math

import numpy as np

from app.search.constants import BOOKS_WITH_TERM_ABOVE_EXPR, NUM_COMPARATIVE_TERMS
from app.search.schemas.semantic_drift import (
    BookSimilarity,
    ExprSimilarityData,
    TermSimilarityData,
)
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


def get_related_terms(
    books_similarity_cache: BooksSimilarityCache,
    book_ids: list[BookIndex],
    query: SearchExpr,
    *,
    selected_book_id: BookIndex | None = None,
) -> tuple[
    ExprSimilarityData, list[TermSimilarityData], list[TermSimilarityData]
]:

    books_similarity_vectors = books_similarity_cache.load_books(book_ids, query)

    if selected_book_id is None:
        peers = books_similarity_vectors
    else:
        peers = [books_similarity_cache.load_book(selected_book_id, query)]

    valid_books_similarity_vectors = [
        book
        for book in books_similarity_vectors
        if any(
            peer.book_id != book.book_id and book.is_comparable(peer) for peer in peers
        )
    ]

    if not valid_books_similarity_vectors:
        return (
            ExprSimilarityData(
                expr=query.serialized, terms=query.terms, book_similarities=[]
            ),
            [],
            [],
        )

    cross_book_mean_similarity = float(
        np.mean([book.local_mean_similarity for book in valid_books_similarity_vectors])
    )

    all_similarities = (
        np.concat(
            [book.centered_similarity_vectors for book in books_similarity_vectors]
        )
        + cross_book_mean_similarity
    )

    terms, term_iloc = get_unique_terms(
        np.concat([book.terms for book in books_similarity_vectors])
    )

    n_books_above_expr = np.bincount(
        term_iloc, weights=all_similarities > cross_book_mean_similarity
    ).astype(np.intp)
    is_relevant_to_corpus = n_books_above_expr >= math.ceil(
        BOOKS_WITH_TERM_ABOVE_EXPR * len(books_similarity_vectors)
    )
    if selected_book_id is not None:
        selected_terms = books_similarity_cache.books_term_cache[selected_book_id].terms
        position = np.searchsorted(selected_terms, terms)
        is_relevant_to_corpus &= np.append(selected_terms, "")[position] == terms

    n_books_in = np.bincount(term_iloc)  # Terms are unique within a book
    similarity_mean = np.bincount(term_iloc, weights=all_similarities) / n_books_in

    similarity_std = np.sqrt(
        np.bincount(
            term_iloc, weights=(all_similarities - similarity_mean[term_iloc]) ** 2
        )
        / np.maximum(n_books_in - 1, 1)
    )

    local_iloc = np.flatnonzero(is_relevant_to_corpus)
    mean_ranked = local_iloc[np.argsort(-similarity_mean[local_iloc], kind="stable")][
        :NUM_COMPARATIVE_TERMS
    ]
    std_iloc = np.setdiff1d(local_iloc, mean_ranked)
    std_ranked = std_iloc[np.argsort(-similarity_std[std_iloc], kind="stable")][
        :NUM_COMPARATIVE_TERMS
    ]

    books_term_cache = books_similarity_cache.books_term_cache
    expr_data = ExprSimilarityData(
        expr=query.serialized,
        terms=query.terms,
        book_similarities=[
            BookSimilarity(
                book_id=book.book_id.source_id,
                similarity=cross_book_mean_similarity,
                # For a compound expression, the vocabulary volume behind the line.
                occurrences=sum(
                    books_term_cache[book.book_id].get_term_count(term)
                    for term in query.terms
                ),
            )
            for book in valid_books_similarity_vectors
        ],
    )

    def get_term_data(iloc):
        term = str(terms[iloc])
        term_similarities = []
        for book in valid_books_similarity_vectors:
            try:
                similarity = book.get_centered_similarity(term)
            except KeyError:
                continue
            term_similarities.append(
                BookSimilarity(
                    book_id=book.book_id.source_id,
                    similarity=similarity + cross_book_mean_similarity,
                    occurrences=books_term_cache[book.book_id].get_term_count(term),
                )
            )
        return TermSimilarityData(
            term=term,
            similarity_mean=float(similarity_mean[iloc]),
            similarity_std=float(similarity_std[iloc]),
            n_books_in=int(n_books_in[iloc]),
            book_similarities=term_similarities,
        )

    top_mean = [get_term_data(iloc) for iloc in mean_ranked]
    top_std = [get_term_data(iloc) for iloc in std_ranked]

    return expr_data, top_mean, top_std
