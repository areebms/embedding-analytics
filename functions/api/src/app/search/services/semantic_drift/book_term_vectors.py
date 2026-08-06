from __future__ import annotations

import contextvars
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.core.logging import request_log
from app.search.schemas.semantic_drift import OpNode, TermNode
from app.search.services.semantic_drift.utils import normalize_vectors
from shared.commons import BookIndex
from shared.tables.book_terms import ADVERB_TAGS, BookTermTable


class BookTermVectors:

    DTYPE = np.float32  # the stored vectors are float16; float64 buys no precision

    def __init__(
        self,
        terms: Sequence[str] | np.ndarray,
        term_vectors: list[np.ndarray],
        term_counts: Sequence[int],
        term_tags: Sequence[frozenset[str]],
    ):
        terms = np.array(terms, dtype=np.str_)

        if not len(terms):
            self.terms = terms
            self.term_iloc: dict[str, int] = {}
            self.term_vectors = np.zeros((0, 0, 0), dtype=self.DTYPE)
            self.term_counts = np.zeros(0, dtype=np.int64)
            self.term_tags: list[frozenset[str]] = []
            return

        order = np.argsort(terms)
        self.terms = terms[order]
        self.term_iloc = {str(term): i for i, term in enumerate(self.terms)}

        n_seeds = min(term_vector.shape[0] for term_vector in term_vectors)
        self.term_vectors = np.stack([term_vectors[i][:n_seeds] for i in order], axis=1)

        self.term_counts = np.asarray(term_counts, dtype=np.int64)[order]
        self.term_tags = [term_tags[i] for i in order]

    def get_term_count(self, term: str) -> int:
        """Occurrences of the term in this book; 0 if the book does not use it."""
        iloc = self.term_iloc.get(term)
        return 0 if iloc is None else int(self.term_counts[iloc])

    def get_term_tags(self, term: str) -> frozenset[str]:
        """Part-of-speech codes this book tagged the term with."""
        iloc = self.term_iloc.get(term)
        return frozenset() if iloc is None else self.term_tags[iloc]

    def get_expr_vectors(self, node: TermNode | OpNode) -> np.ndarray:

        if isinstance(node, TermNode):
            return self.term_vectors[:, self.term_iloc[node.term]]

        left = self.get_expr_vectors(node.args[0])
        right = self.get_expr_vectors(node.args[1])
        result = left + right if node.op == "+" else left - right
        return normalize_vectors(result, axis=1)

    def missing_terms(self, terms: Iterable[str]) -> set[str]:
        return set(terms) - self.term_iloc.keys()


class BooksTermCache:

    MAX_LOAD_WORKERS = 8

    def __init__(self, table: BookTermTable):
        self.table = table
        self.books_term_vectors: dict[BookIndex, BookTermVectors] = {}
        self.shared_terms_index: dict[
            tuple[BookIndex, BookIndex], tuple[np.ndarray, np.ndarray]
        ] = {}

    def __getitem__(self, book_id: BookIndex) -> BookTermVectors:
        return self.books_term_vectors[book_id]

    def get_shared_term_indexes(
        self, a_book_id: BookIndex, b_book_id: BookIndex
    ) -> tuple[np.ndarray, np.ndarray]:

        key = (a_book_id, b_book_id)
        if key not in self.shared_terms_index:
            a_terms, b_terms = self[a_book_id].terms, self[b_book_id].terms

            if not len(a_terms) or not len(b_terms):
                empty = np.empty(0, dtype=np.intp)
                a_indexes = b_indexes = empty
            else:
                position = np.searchsorted(b_terms, a_terms)
                matched = b_terms[np.minimum(position, len(b_terms) - 1)] == a_terms
                a_indexes, b_indexes = np.flatnonzero(matched), position[matched]

            self.shared_terms_index[key] = (a_indexes, b_indexes)
            self.shared_terms_index[(b_book_id, a_book_id)] = (b_indexes, a_indexes)

        return self.shared_terms_index[key]

    def load_book(self, book_id: BookIndex) -> BookTermVectors:

        if book_id in self.books_term_vectors:
            return self.books_term_vectors[book_id]

        terms, term_vectors, term_counts, term_tags = [], [], [], []
        for entry in self.table.get_entries(
            book_id, fields=["term", "tags", "vectors", "count_"]
        ):
            if entry.get("tags") == ADVERB_TAGS:
                continue

            if "vectors" not in entry:
                continue

            terms.append(entry["term"])
            term_counts.append(int(entry.get("count_", 0)))  # stored as a Decimal
            term_tags.append(frozenset(entry.get("tags") or ()))
            term_vectors.append(
                normalize_vectors(
                    np.stack(
                        [
                            np.frombuffer(bytes(buffer), dtype=np.float16).astype(
                                BookTermVectors.DTYPE
                            )
                            for buffer in entry["vectors"]
                        ]
                    ),
                    axis=1,
                )
            )

        self.books_term_vectors[book_id] = BookTermVectors(
            terms, term_vectors, term_counts, term_tags
        )

        request_log.get({}).setdefault("cache_warmed", {})[str(book_id)] = len(terms)
        return self.books_term_vectors[book_id]

    def warm_cache(self, book_ids: Iterable[BookIndex]):
        """Load several books' term matrices concurrently."""
        with ThreadPoolExecutor(max_workers=self.MAX_LOAD_WORKERS) as pool:
            futures = []
            pending: set[BookIndex] = set()
            for book_id in book_ids:
                if book_id not in self.books_term_vectors and book_id not in pending:
                    pending.add(book_id)
                    futures.append(
                        pool.submit(
                            contextvars.copy_context().run, self.load_book, book_id
                        )
                    )

            for future in futures:
                future.result()

    def get_books_with_search_query(
        self, book_ids: Iterable[BookIndex], search_query
    ) -> list[BookIndex]:
        return [
            book_id
            for book_id in book_ids
            if not self[book_id].missing_terms(search_query.terms)
        ]

    def get_missing_terms_by_book(
        self, book_ids: Iterable[BookIndex], terms: Iterable[str]
    ) -> dict[BookIndex, set[str]]:
        return {book_id: self[book_id].missing_terms(terms) for book_id in book_ids}

    def get_term_tags(self, book_ids: Iterable[BookIndex], term: str) -> list[str]:
        tags: set[str] = set()
        for book_id in book_ids:
            tags |= self[book_id].get_term_tags(term)
        return sorted(tags)

    def get_n_shared_terms(
        self, book_id: BookIndex, peer_ids: Iterable[BookIndex]
    ) -> int:
        return max(
            (
                len(self.get_shared_term_indexes(book_id, peer_id)[0])
                for peer_id in peer_ids
                if peer_id != book_id
            ),
            default=0,
        )
