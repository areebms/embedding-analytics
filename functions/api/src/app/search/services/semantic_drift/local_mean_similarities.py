from __future__ import annotations

from typing import NamedTuple

import numpy as np

from app.search.constants import (
    BARE_TERM_PATTERN,
    MIN_LOCAL_NEAREST_TERMS,
    MIN_MATCHING_BOOKS,
    NEAREST_TERM_COUNT,
    NUM_LOCAL_NEAREST_TERMS,
    T_CRIT_95,
)
from app.search.errors import MissingTermsError, NoLocalNearestTermsError
from app.search.schemas.semantic_drift import BookLocalMeanSimilarity, OpNode, TermNode
from app.search.services.semantic_drift.book_term_vectors import BooksTermCache
from app.search.services.semantic_drift.utils import normalize_vectors
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


class BookSimilarityVectors:

    def __init__(
        self,
        books_cache: BooksTermCache,
        book_id: BookIndex,
        query: SearchExpr,
    ):
        self.book_id = book_id
        self.query = query

        missing = books_cache[book_id].missing_terms(query.terms)
        if missing:
            raise MissingTermsError(missing, book_id)

        query_vectors = books_cache[book_id].get_expr_vectors(query.tree)

        is_valid = ~np.isin(books_cache[book_id].terms, query.terms)

        self.terms = books_cache[book_id].terms[is_valid]
        self.similarity_vectors = self.get_similarity_vectors(
            query_vectors, books_cache[book_id].term_vectors[:, is_valid]
        )

    @staticmethod
    def get_similarity_vectors(query_vectors: np.ndarray, term_vectors: np.ndarray):
        # return has shape (n_seeds, n_terms)
        n_seeds = min(query_vectors.shape[0], term_vectors.shape[0])
        return np.matmul(term_vectors[:n_seeds], query_vectors[:n_seeds, :, None])[
            ..., 0
        ]

    def get_local_cosine_similarity(self, peer: BookSimilarityVectors):
        # returns one dot-product-cosine value per seed.

        shared_indexes = np.flatnonzero(np.isin(self.terms, peer.terms))
        shared_peer_indexes = np.flatnonzero(np.isin(peer.terms, self.terms))

        if len(shared_indexes) < MIN_LOCAL_NEAREST_TERMS:
            raise NoLocalNearestTermsError(
                self.book_id, peer.book_id, len(shared_indexes)
            )

        # vectors of terms closest to the expression.
        shared_similarity_vectors = self.similarity_vectors[:, shared_indexes].mean(
            axis=0
        )
        sorted_iloc = np.argsort(-shared_similarity_vectors)[:NUM_LOCAL_NEAREST_TERMS]

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
        local_similarity = np.sum(
            normalize_vectors(shared_book_vectors)
            * normalize_vectors(shared_peer_vectors),
            axis=1,
        )
        return LocalCosineSimilarity(local_similarity, len(sorted_iloc))


def get_nearest_terms(
    books_cache: BooksTermCache,
    book_ids: list[BookIndex],
    query: SearchExpr,
    *,
    selected_book_id: BookIndex | None = None,
):

    books_vectors = [
        BookSimilarityVectors(books_cache, book_id, query) for book_id in book_ids
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
        is_valid &= np.isin(terms, books_cache[selected_book_id].terms)
    is_valid = np.flatnonzero(is_valid)

    mean_similarity = total_similarity[is_valid] / book_count[is_valid]
    ranked_iloc = is_valid[
        np.argsort(-mean_similarity, kind="stable")[:NEAREST_TERM_COUNT]
    ]
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
    books_cache: BooksTermCache,
    expr: SearchExpr,
    book_ids: list[BookIndex],
    selected_book_id: BookIndex | None = None,
) -> list[BookLocalMeanSimilarity]:

    books_similarity_vectors = []
    for book_id in book_ids:
        try:
            books_similarity_vectors.append(
                BookSimilarityVectors(books_cache, book_id, expr)
            )
        except MissingTermsError:
            continue

    if selected_book_id is None:
        peers = books_similarity_vectors
    else:
        peers = [BookSimilarityVectors(books_cache, selected_book_id, expr)]

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
