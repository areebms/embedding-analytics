from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from app.search.constants import NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY
from app.search.errors import MissingTermsError, NoLocalNearestTermsError
from app.search.services.semantic_drift.book_term_vectors import BooksTermCache
from app.search.services.semantic_drift.utils import SearchExpr, center_vectors
from shared.commons import BookIndex


class BookSimilarityVectors:

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
        self.masked_iloc = np.cumsum(self.is_valid) - 1

        self.terms = book_terms[self.is_valid]
        self.similarity_vectors = similarity_vectors[:, self.is_valid]
        self.mean_similarities_to_query = self.similarity_vectors.mean(axis=0)

    @staticmethod
    def get_similarity_vectors(query_vectors: np.ndarray, term_vectors: np.ndarray):
        # return has shape (n_seeds, n_terms)
        n_seeds = min(query_vectors.shape[0], term_vectors.shape[0])
        return np.matmul(term_vectors[:n_seeds], query_vectors[:n_seeds, :, None])[
            ..., 0
        ]

    def get_local_similarity(self, peer: BookSimilarityVectors):
        # returns one value per seed

        indexes, peer_indexes = self.books_term_cache.get_shared_term_indexes(
            self.book_id, peer.book_id
        )
        local_indexes = self.is_valid[indexes] & peer.is_valid[peer_indexes]
        shared_indexes = self.masked_iloc[indexes[local_indexes]]
        shared_peer_indexes = peer.masked_iloc[peer_indexes[local_indexes]]

        if len(shared_indexes) < NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY:
            raise NoLocalNearestTermsError(
                self.book_id, peer.book_id, len(shared_indexes)
            )

        # vectors of terms closest to the expression.
        shared_similarity_vectors = self.mean_similarities_to_query[shared_indexes]
        sorted_iloc = np.argsort(-shared_similarity_vectors)[
            :NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY
        ]

        n_seeds = min(
            self.similarity_vectors.shape[0], peer.similarity_vectors.shape[0]
        )

        # Similarity vectors for terms shared between books.
        shared_book_vectors = self.similarity_vectors[:n_seeds][
            :, shared_indexes[sorted_iloc]
        ]
        shared_peer_vectors = peer.similarity_vectors[:n_seeds][
            :, shared_peer_indexes[sorted_iloc]
        ]
        return self.correlate_vectors(shared_book_vectors, shared_peer_vectors)

    @staticmethod
    def correlate_vectors(a_vectors: np.ndarray, b_vectors: np.ndarray) -> np.ndarray:
        a_centered = center_vectors(a_vectors)
        b_centered = center_vectors(b_vectors)
        a_dot_b = np.einsum("...i,...i->...", a_centered, b_centered)
        a_dot_a = np.einsum("...i,...i->...", a_centered, a_centered)
        b_dot_b = np.einsum("...i,...i->...", b_centered, b_centered)
        return a_dot_b / np.maximum(np.sqrt(a_dot_a * b_dot_b), 1e-12)


class BooksSimilarityCache:

    def __init__(self, books_term_cache: BooksTermCache):
        self.books_term_cache = books_term_cache
        self.book_similarity_vectors: dict[
            tuple[BookIndex, str], BookSimilarityVectors
        ] = {}

    def __getitem__(self, key: tuple[BookIndex, str]) -> BookSimilarityVectors:
        return self.book_similarity_vectors[key]

    def load_book(self, book_id: BookIndex, expr: SearchExpr) -> BookSimilarityVectors:

        key = (book_id, expr.serialized)
        if key in self.book_similarity_vectors:
            return self.book_similarity_vectors[key]

        book_term_vectors = self.books_term_cache[book_id]
        missing = book_term_vectors.missing_terms(expr.terms)
        if missing:
            raise MissingTermsError(missing, book_id)

        return self._store(
            book_id,
            expr,
            BookSimilarityVectors.get_similarity_vectors(
                book_term_vectors.get_expr_vectors(expr.tree),
                book_term_vectors.term_vectors,
            ),
        )

    def load_books(
        self, book_ids: Iterable[BookIndex], expr: SearchExpr
    ) -> list[BookSimilarityVectors]:
        """Vectors for each named book that carries the expression, in order."""

        return [
            self.load_book(book_id, expr)
            for book_id in self.books_term_cache.get_books_with_search_query(
                book_ids, expr
            )
        ]

    def _store(
        self, book_id: BookIndex, expr: SearchExpr, similarity_vectors: np.ndarray
    ) -> BookSimilarityVectors:
        key = (book_id, expr.serialized)
        self.book_similarity_vectors[key] = BookSimilarityVectors(
            self.books_term_cache, book_id, expr, similarity_vectors
        )
        return self.book_similarity_vectors[key]

    def warm_cache(self, book_ids: Iterable[BookIndex], exprs: Sequence[SearchExpr]):
        """One matmul per book across every expression, instead of one each."""

        for book_id in book_ids:
            book_term_vectors = self.books_term_cache[book_id]
            book_exprs = [
                expr
                for expr in exprs
                if (book_id, expr.serialized) not in self.book_similarity_vectors
                and not book_term_vectors.missing_terms(expr.terms)
            ]
            if not book_exprs:
                continue

            book_expr_vectors = np.stack(
                [book_term_vectors.get_expr_vectors(expr.tree) for expr in book_exprs],
                axis=-1,
            )
            n_seeds = min(
                book_expr_vectors.shape[0], book_term_vectors.term_vectors.shape[0]
            )
            # (n_seeds, n_terms, dim) @ (n_seeds, dim, n_exprs)
            book_expr_similarities = np.matmul(
                book_term_vectors.term_vectors[:n_seeds], book_expr_vectors[:n_seeds]
            )

            for index, expr in enumerate(book_exprs):
                self._store(book_id, expr, book_expr_similarities[..., index])
