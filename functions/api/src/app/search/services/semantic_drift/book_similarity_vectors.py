from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.search.constants import MAX_RANK_FOR_TERM_SELECTION, NUM_LOCAL_NEAREST_TERMS
from app.search.errors import MissingTermsError
from app.search.services.semantic_drift.book_term_vectors import BooksTermCache
from app.search.services.semantic_drift.utils import SearchExpr
from shared.commons import BookIndex


def get_n_highest_similarities(similarities: np.ndarray, n: int) -> np.ndarray:

    if not len(similarities):
        return similarities

    n = min(n, len(similarities))

    return np.partition(similarities, -n)[-n:]


def get_local_mean_similarity(similarities: np.ndarray, n: int) -> float:

    highest_similarities = get_n_highest_similarities(similarities, n)

    if not len(highest_similarities):
        return 0.0

    return float(highest_similarities.mean())


def get_is_local(similarities: np.ndarray, n: int) -> np.ndarray:

    highest_similarities = get_n_highest_similarities(similarities, n)

    if not len(highest_similarities):
        return np.zeros(0, dtype=bool)

    return similarities >= highest_similarities.min()


class BookSimilarityVectors:
    """Second Order Vectors with respect to SearchExpr"""

    def __init__(
        self,
        books_term_cache: BooksTermCache,
        book_id: BookIndex,
        query: SearchExpr,
        similarity_vectors: np.ndarray,
    ):
        self.book_id = book_id
        self.query = query
        self.books_term_cache = books_term_cache

        book_terms = books_term_cache[book_id].terms
        self.is_valid = ~np.isin(book_terms, query.terms)

        self.terms = book_terms[self.is_valid]

        self.similarity_vectors = similarity_vectors[self.is_valid]
        self.local_mean_similarity = get_local_mean_similarity(
            self.similarity_vectors, NUM_LOCAL_NEAREST_TERMS
        )
        self.centered_similarity_vectors = (
            self.similarity_vectors - self.local_mean_similarity
        )
        self.is_local = get_is_local(
            self.similarity_vectors, MAX_RANK_FOR_TERM_SELECTION
        )

    @staticmethod
    def get_similarity_vectors(query_vector: np.ndarray, term_vectors: np.ndarray):
        # return has shape (n_terms,)
        return term_vectors @ query_vector

    def is_comparable(self, peer: BookSimilarityVectors) -> bool:

        indexes, peer_indexes = self.books_term_cache.get_shared_term_indexes(
            self.book_id, peer.book_id
        )
        n_shared = np.count_nonzero(
            self.is_valid[indexes] & peer.is_valid[peer_indexes]
        )
        return n_shared >= NUM_LOCAL_NEAREST_TERMS

    def get_centered_similarity(self, term: str) -> float | None:

        term_iloc = np.searchsorted(self.terms, term)
        if term_iloc == len(self.terms) or self.terms[term_iloc] != term:
            return None
        return float(self.centered_similarity_vectors[term_iloc])


class BooksSimilarityCache:

    def __init__(self, books_term_cache: BooksTermCache):
        self.books_term_cache = books_term_cache
        self.book_similarity_vectors: dict[
            tuple[BookIndex, str], BookSimilarityVectors
        ] = {}

    def load_book(self, book_id: BookIndex, expr: SearchExpr) -> BookSimilarityVectors:

        key = (book_id, expr.serialized)
        if key in self.book_similarity_vectors:
            return self.book_similarity_vectors[key]

        book_term_vectors = self.books_term_cache[book_id]
        missing = book_term_vectors.missing_terms(expr.terms)
        if missing:
            raise MissingTermsError(missing, book_id)

        return self.save(
            book_id,
            expr,
            BookSimilarityVectors.get_similarity_vectors(
                book_term_vectors.get_expr_vector(expr.tree),
                book_term_vectors.term_vectors,
            ),
        )

    def load_books(
        self, book_ids: Iterable[BookIndex], expr: SearchExpr
    ) -> list[BookSimilarityVectors]:
        """Vectors for each named book that carries the expression, in order."""

        return [
            self.load_book(book_id, expr)
            for book_id in self.books_term_cache.get_books_with_expr(
                book_ids, expr
            )
        ]

    def save(
        self, book_id: BookIndex, expr: SearchExpr, similarity_vectors: np.ndarray
    ) -> BookSimilarityVectors:
        key = (book_id, expr.serialized)
        self.book_similarity_vectors[key] = BookSimilarityVectors(
            self.books_term_cache, book_id, expr, similarity_vectors
        )
        return self.book_similarity_vectors[key]
