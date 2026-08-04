import threading
import time
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import VOCAB, book_rows, make_term_entry, set_multi_book_table
from app.core.logging import request_log
from app.search.dependencies import get_books_term_cache
from app.search.errors import MissingTermsError
from app.search.schemas.semantic_drift import OpNode, TermNode
from app.search.services.semantic_drift import (
    BooksSimilarityCache,
    BooksTermCache,
    BookTermVectors,
    SearchExpr,
)
from shared.commons import BookIndex


def _term_cache(entries=None, side_effect=None):
    """A cache over a term-table mock whose get_entries serves one book's rows."""
    entries = list(book_rows(1).values()) if entries is None else list(entries)
    table = MagicMock()
    table.get_entries.side_effect = side_effect or (
        lambda book_id, fields=None: entries
    )
    return BooksTermCache(table)


def _fetched(books):
    """The books get_entries was called with, read off the mock behind the cache."""
    table = cast(MagicMock, books.table)
    return [call.args[0] for call in table.get_entries.call_args_list]


# -- BooksTermCache.warm_cache ------------------------------------------------


def test_warm_cache_loads_books_concurrently():
    book_ids = [BookIndex(i) for i in (1, 2, 3)]
    barrier = threading.Barrier(len(book_ids), timeout=10)
    entries = list(book_rows(1).values())

    def get_entries(book_id, fields=None):
        barrier.wait()
        return entries

    books = _term_cache(side_effect=get_entries)
    books.warm_cache(book_ids)

    assert set(books.books_term_vectors) == set(book_ids)


def test_warm_cache_records_cache_warmed_on_the_request_log():
    book_ids = [BookIndex(1), BookIndex(2)]
    log: dict = {}
    token = request_log.set(log)
    try:
        _term_cache().warm_cache(book_ids)
    finally:
        request_log.reset(token)

    assert set(log["cache_warmed"]) == {str(book_id) for book_id in book_ids}


def test_warm_cache_propagates_a_failed_load():
    def explode(book_id, fields=None):
        raise RuntimeError("dynamodb unavailable")

    with pytest.raises(RuntimeError, match="dynamodb unavailable"):
        _term_cache(side_effect=explode).warm_cache([BookIndex(1), BookIndex(2)])


def test_warm_cache_fetches_each_uncached_book_once():
    cached, repeated, fresh = BookIndex(1), BookIndex(2), BookIndex(3)
    entries = list(book_rows(1).values())

    def get_entries(book_id, fields=None):
        time.sleep(0.05)  # a DynamoDB round trip, not a dict lookup
        return entries

    books = _term_cache(side_effect=get_entries)
    # membership is all the prefetch checks, so any object will do
    books.books_term_vectors[cached] = cast(BookTermVectors, object())

    books.warm_cache([cached, repeated, repeated, fresh])

    assert sorted(_fetched(books)) == [repeated, fresh]


# -- BooksTermCache row filtering and lookups ---------------------------------


def test_load_book_skips_adverbs_and_rows_without_vectors():
    books = _term_cache(
        [
            make_term_entry("labour"),
            make_term_entry("quickly", tags={"R"}),
            make_term_entry("value", tags={"N", "R"}),  # not adverb-ONLY, so it stays
            {"term": "unvectorized", "tags": {"N"}},
        ]
    )
    book_term_vectors = books.load_book(BookIndex(1))

    assert list(book_term_vectors.terms) == ["labour", "value"]
    assert book_term_vectors.missing_terms(["labour", "quickly", "unvectorized"]) == {
        "quickly",
        "unvectorized",
    }


def test_load_book_of_a_book_whose_every_row_is_filtered_is_empty_not_an_error():
    # Same shape as a book_id the table knows nothing about: no vocabulary, no
    # exception -- the missing-terms path reports it from there.
    entries = [make_term_entry("quickly", tags={"R"})]

    book_term_vectors = _term_cache(entries).load_book(BookIndex(1))

    assert not len(book_term_vectors.terms)
    assert book_term_vectors.missing_terms(["labour"]) == {"labour"}


def test_get_missing_terms_by_book_asks_only_the_books_named():
    books = _term_cache()
    for source_id in (1, 3):
        books.load_book(BookIndex(source_id))
    books.books_term_vectors[BookIndex(2)] = BookTermVectors([], [])

    book_ids = [BookIndex(2), BookIndex(1)]
    assert books.get_missing_terms_by_book(book_ids, ["labour", "wage"]) == {
        BookIndex(2): {"labour", "wage"},
        BookIndex(1): set(),
    }
    assert list(books.get_missing_terms_by_book(book_ids, ["labour"])) == book_ids


def test_get_expr_vectors_combines_leaves_by_operator():
    vectors = _term_cache().load_book(BookIndex(1))
    labour = vectors.get_expr_vectors(TermNode(term="labour"))
    value = vectors.get_expr_vectors(TermNode(term="value"))

    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def expr(op, *terms):
        return vectors.get_expr_vectors(
            OpNode(op=op, args=[TermNode(term=term) for term in terms])
        )

    plus, minus = expr("+", "labour", "value"), expr("-", "labour", "value")

    assert plus.shape == labour.shape  # one vector per seed, not one per book
    np.testing.assert_allclose(plus, unit(labour + value), rtol=1e-5)
    np.testing.assert_allclose(minus, unit(labour - value), rtol=1e-5)

    # Nesting recurses on both sides, so an inner result is renormalized before it
    # is combined -- (labour + value) - wage is NOT labour + value - wage.
    nested = vectors.get_expr_vectors(
        OpNode(
            op="-",
            args=[
                OpNode(op="+", args=[TermNode(term="labour"), TermNode(term="value")]),
                TermNode(term="wage"),
            ],
        )
    )
    wage = vectors.get_expr_vectors(TermNode(term="wage"))
    np.testing.assert_allclose(nested, unit(plus - wage), rtol=1e-5)
    assert not np.allclose(nested, unit(labour + value - wage))


def test_books_term_cache_is_reused_across_requests_for_one_table():
    table = MagicMock()

    first = get_books_term_cache(table)

    assert get_books_term_cache(table) is first
    assert get_books_term_cache(MagicMock()) is not first


# -- Shared-term index and batched dot products -------------------------------


def _two_books(a_vocab, b_vocab):
    """A warmed cache holding two books with the given vocabularies."""
    table = MagicMock()
    set_multi_book_table(
        table, {1: book_rows(1, vocab=a_vocab), 2: book_rows(2, vocab=b_vocab)}
    )
    books = BooksTermCache(table)
    books.warm_cache([BookIndex(1), BookIndex(2)])
    return books


def test_shared_term_indexes_pair_the_same_term_in_both_books():
    books = _two_books(["alpha", "delta", "kappa", "omega"],
                       ["beta", "delta", "omega", "zeta"])

    book_indexes, peer_indexes = books.get_shared_term_indexes(
        BookIndex(1), BookIndex(2)
    )

    assert list(books[BookIndex(1)].terms[book_indexes]) == ["delta", "omega"]
    assert list(books[BookIndex(2)].terms[peer_indexes]) == ["delta", "omega"]


def test_shared_term_indexes_are_cached_both_ways_round():
    books = _two_books(VOCAB, VOCAB)

    forward = books.get_shared_term_indexes(BookIndex(1), BookIndex(2))
    reverse = books.get_shared_term_indexes(BookIndex(2), BookIndex(1))

    assert books.get_shared_term_indexes(BookIndex(1), BookIndex(2))[0] is forward[0]
    assert reverse[0] is forward[1] and reverse[1] is forward[0]


def test_shared_term_indexes_of_an_empty_book_are_empty():
    books = _two_books(VOCAB, VOCAB)
    books.books_term_vectors[BookIndex(2)] = BookTermVectors([], [])

    book_indexes, peer_indexes = books.get_shared_term_indexes(
        BookIndex(1), BookIndex(2)
    )

    assert len(book_indexes) == len(peer_indexes) == 0


def test_batched_similarity_vectors_match_the_one_expression_at_a_time_result():
    books = _two_books(VOCAB, VOCAB)
    exprs = [
        SearchExpr.from_query(TermNode(term="labour")),
        SearchExpr.from_query(TermNode(term="value")),
        SearchExpr.from_query(
            OpNode(op="-", args=[TermNode(term="wage"), TermNode(term="rent")])
        ),
    ]

    batched = BooksSimilarityCache(books)
    batched.warm_cache([BookIndex(1), BookIndex(2)], exprs)

    unbatched = BooksSimilarityCache(books)

    assert len(batched.book_similarity_vectors) == 2 * len(exprs)
    for book_id in (BookIndex(1), BookIndex(2)):
        for expr in exprs:
            np.testing.assert_allclose(
                batched[book_id, expr.serialized].similarity_vectors,
                unbatched.load_book(book_id, expr).similarity_vectors,
                atol=1e-6,
            )


def test_batched_similarity_vectors_skip_expressions_a_book_lacks():
    books = _two_books(VOCAB, ["labour", "value"])
    exprs = [
        SearchExpr.from_query(TermNode(term="labour")),
        SearchExpr.from_query(TermNode(term="rent")),
    ]

    batched = BooksSimilarityCache(books)
    batched.warm_cache([BookIndex(1), BookIndex(2)], exprs)

    assert (BookIndex(2), "labour") in batched.book_similarity_vectors
    assert (BookIndex(2), "rent") not in batched.book_similarity_vectors
    assert (BookIndex(1), "rent") in batched.book_similarity_vectors

    with pytest.raises(MissingTermsError):
        batched.load_book(BookIndex(2), exprs[1])


def test_similarity_vectors_are_computed_once_per_book_and_expression():
    books = _two_books(VOCAB, VOCAB)
    expr = SearchExpr.from_query(TermNode(term="labour"))
    cache = BooksSimilarityCache(books)

    first = cache.load_book(BookIndex(1), expr)

    assert cache.load_book(BookIndex(1), expr) is first
    # ...and warming afterwards must not quietly replace it with a batched copy.
    cache.warm_cache([BookIndex(1), BookIndex(2)], [expr])
    assert cache[BookIndex(1), expr.serialized] is first
    assert (BookIndex(2), expr.serialized) in cache.book_similarity_vectors
