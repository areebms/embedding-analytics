"""Tests for the /semantic-drift endpoint.

Uses a multi-book term-table mock: get_entries feeds BooksTermCache.load_book, which
is where every matrix (and so every leaf-term lookup) comes from.
"""

import math
import threading
import time
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import make_term_entry
from app.core.logging import request_log
from app.search.dependencies import get_books_term_cache
from app.search.schemas.semantic_drift import MAX_TREE_DEPTH, OpNode, TermNode
from app.search.services.semantic_drift import (
    NEAREST_TERM_COUNT,
    BooksTermCache,
    BookTermVectors,
)
from shared.commons import BookIndex

# Same anchor vocabulary in every same-genre book, so any pair shares all of it.
VOCAB = ["labour", "value", "wage", "rent", "stock", "price", "profit", "capital"]


def book_rows(book_seed, vocab=VOCAB, n_seeds=5):
    """One book's term entries, seeds offset per book so vectors (and therefore
    cross-book agreement) actually differ. `n_seeds` is how many Word2Vec models
    the book was trained with -- it sets the length of every per-seed sample the
    book takes part in."""
    return {
        term: make_term_entry(term, n_seeds=n_seeds, seed=book_seed * 100 + i)
        for i, term in enumerate(vocab)
    }


def set_multi_book_table(term_table, books):
    """Wire a term-table mock to serve several books, keyed on BookIndex.source_id.

    `books` maps source_book_id -> {term: entry}.
    """

    def get_entries(book_id, fields=None):
        return list(books.get(book_id.source_id, {}).values())

    def batch_get_entries(terms, book_id, fields=None):
        entries = books.get(book_id.source_id, {})
        return [entries[t] for t in terms if t in entries]

    term_table.get_entries.side_effect = get_entries
    term_table.batch_get_entries.side_effect = batch_get_entries


# -- reading the nested payload -----------------------------------------------
#
# The payload is grouped by line already -- the query's own under `expr`, one per
# nearest term under `nearest_terms` -- so these read ACROSS that grouping to ask
# the book-shaped questions: every line a book drew, its point on one line, and
# the book-level rows. Two distinct things a book can lack on a line:
#   - it is ABSENT from the line -- its vocabulary lacks that line's term, which
#     its `missing_terms` row names.
#   - it is present with a NULL `similarity` -- it has the term, but found no
#     peer it could be compared against for it.
# No book is dropped from the lines wholesale, so a book that could be measured
# nowhere is present everywhere it has the term, carrying nulls.


def lines(body):
    """Every line on the chart, the query's first, as (term, books) pairs."""
    return [(body["expr"]["expr"], body["expr"]["books"])] + [
        (term_data["term"], term_data["books"]) for term_data in body["nearest_terms"]
    ]


def nearest_terms(body):
    """Just the nearest-term labels, which the payload no longer lists flat."""
    return [term_data["term"] for term_data in body["nearest_terms"]]


def lines_of(body, book_id):
    """Every line drawn for one book, in payload order."""
    return [
        dict(book_data, term=term)
        for term, books in lines(body)
        for book_data in books
        if book_data["book_id"] == book_id
    ]


def line(body, book_id, term):
    """One book's point on one line."""
    books = next(books for line_term, books in lines(body) if line_term == term)
    return next(book_data for book_data in books if book_data["book_id"] == book_id)


def line_or_none(body, book_id, term):
    """One book's point on one line, or None where the book is absent from it."""
    books = next(books for line_term, books in lines(body) if line_term == term)
    return next(
        (book_data for book_data in books if book_data["book_id"] == book_id), None
    )


def books_by_id(body):
    return {b["id"]: b for b in body["books"]}


# -- /semantic-drift --------------------------------------------------------


def test_chart_happy_path(client, patch_tables):
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: book_rows(3)})

    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2, 3]},
    )

    assert response.status_code == 200
    body = response.json()

    # The query's serialized label, and its nearest terms in the pinned book
    # (excluding the query itself, since a term is its own nearest term).
    # This vocabulary is smaller than NEAREST_TERM_COUNT, so every eligible
    # term is drawn -- the cap itself is exercised in test_chart_caps_nearest_terms.
    assert body["expr"]["expr"] == "labour"
    assert body["expr"]["terms"] == ["labour"]
    assert len(nearest_terms(body)) == len(VOCAB) - 1
    assert "labour" not in nearest_terms(body)
    assert set(nearest_terms(body)).issubset(set(VOCAB))

    # One book row per target, ascending by id -- and ONLY the targets: the pinned
    # book has no row of its own. Both targets drew every line, so both appear on
    # every one: the query's own highlighted line first, then one per nearest term.
    assert [b["id"] for b in body["books"]] == [2, 3]
    # every line x both books
    assert sum(len(books) for _, books in lines(body)) == 2 * len(VOCAB)
    for book_id in (2, 3):
        assert [e["term"] for e in lines_of(body, book_id)] == [
            body["expr"]["expr"],
            *nearest_terms(body),
        ]
        for entry in lines_of(body, book_id):
            assert entry["similarity"] is not None  # measured, so not a gap
            assert 0.0 <= entry["min_local_terms"] <= len(VOCAB)
            assert entry["n_seeds"] == 5
            assert entry["n_books"] == 1  # the pinned book backs every line
            lo, hi = entry["similarity_ci"]
            assert lo <= entry["similarity"] <= hi
    # Gone from the wire, not merely false/renamed: response_model drops unknown
    # dict keys silently, so `is False` (or `== []`) would pass by accident. Read
    # off the raw payload -- lines_of() synthesizes `term` from the grouping, so
    # a book's point on a line carries only its own numbers. Availability is the
    # book's presence on these lines; missing_terms lives on the book row.
    for _, books in lines(body):
        for book_data in books:
            assert set(book_data) == {
                "book_id",
                "similarity",
                "similarity_ci",
                "n_seeds",
                "min_local_terms",
                "n_books",
            }
    assert all(set(b) == {"id", "missing_terms"} for b in body["books"])
    assert all(b["missing_terms"] == [] for b in body["books"])


def test_chart_confidence_interval_is_finite_and_straddles_the_value(
    client, patch_tables
):
    # The interval is the spread across SEEDS. Taken instead across the single
    # scalar those seeds average to, it is 0/0 -> NaN, which pydantic serializes as
    # a null bound -- and `lo <= similarity <= hi` catches that only as a TypeError,
    # never as the NaN it is. So assert finiteness directly, and assert the interval
    # has real width: a degenerate zero-spread result must fail here too.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: book_rows(3)})

    body = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2, 3]},
    ).json()

    for _, books in lines(body):
        for book_data in books:
            lo, hi = book_data["similarity_ci"]
            assert math.isfinite(lo) and math.isfinite(hi)
            # Five seeds that disagree, so the half-width is strictly positive.
            assert book_data["n_seeds"] == 5
            assert lo < book_data["similarity"] < hi


def test_chart_draws_a_compound_expression(client, patch_tables):
    # Every `+`/`-` query resolves through get_expr_vectors' OpNode branch, which no
    # route-level test reached -- the only other test sending an op tree 404s before
    # evaluation. Both leaves are excluded from their own nearest terms.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: book_rows(3)})

    response = client.post(
        "/semantic-drift/1",
        json={
            "tree": {"op": "+", "args": [{"term": "labour"}, {"term": "value"}]},
            "book_ids": [2, 3],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["expr"]["expr"] == "labour + value"  # outermost parens stripped
    assert body["expr"]["terms"] == ["labour", "value"]
    assert not {"labour", "value"} & set(nearest_terms(body))
    assert all(entry["similarity"] is not None for entry in lines_of(body, 2))


def test_chart_caps_nearest_terms(client, patch_tables):
    # How many lines get drawn is fixed server-side, not asked for -- a book with
    # plenty of eligible terms still yields exactly NEAREST_TERM_COUNT of them,
    # and the entries follow at one per line per book plus the query.
    _, term_table = patch_tables
    wide_vocab = ["labour"] + [f"term{n}" for n in range(NEAREST_TERM_COUNT + 4)]
    set_multi_book_table(
        term_table,
        {book_id: book_rows(book_id, vocab=wide_vocab) for book_id in (1, 2, 3)},
    )

    body = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2, 3]},
    ).json()

    assert len(nearest_terms(body)) == NEAREST_TERM_COUNT
    assert "labour" not in nearest_terms(body)
    assert len(lines_of(body, 2)) == NEAREST_TERM_COUNT + 1


def test_chart_resolves_from_loaded_matrix_without_per_term_fetches(client, patch_tables):
    # The chart loads each book's whole matrix once (get_entries) and resolves the
    # query and every nearest-term line from those columns. It must never re-fetch
    # an individual term's vectors -- that was a DynamoDB round trip per nearest
    # term per book (up to NEAREST_TERM_COUNT x targets), all of it already in
    # memory.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: book_rows(3)})

    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2, 3]},
    )

    assert response.status_code == 200
    assert term_table.batch_get_entries.call_count == 0
    assert term_table.get_entries.call_count == 3  # one matrix load per book, cached


def test_chart_thin_neighbourhood_is_drawn_with_its_count(client, patch_tables):
    # Book 2 shares a neighbourhood with the pin, just a small one. That is a
    # measurement -- it is drawn, carrying the count it was taken over -- and NOT
    # the same finding as book 3, which shares nothing and so is on no line at
    # all. Collapsing the two would make "we can't compare these books" and "this
    # number is thin" one indistinguishable null.
    _, term_table = patch_tables
    set_multi_book_table(
        term_table,
        {
            1: book_rows(1),
            2: book_rows(2, vocab=["labour", "value", "wage", "rent"]),
            3: book_rows(3, vocab=["labour", "alpha", "beta"]),
            4: book_rows(4),
        },
    )

    body = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2, 3, 4]},
    ).json()

    # Book 2: three anchors survive leaf exclusion -- above the floor, but far
    # fewer than book 4 gets. Measured, drawn, and the count says how thin.
    thin = line(body, 2, "labour")
    assert thin["similarity"] is not None
    assert thin["min_local_terms"] == 3
    assert lines_of(body, 2)
    # The interval is taken across seeds, so it does NOT widen to warn about the
    # thin neighbourhood -- `min_local_terms` is the only signal there is.
    lo, hi = thin["similarity_ci"]
    assert lo <= thin["similarity"] <= hi

    # Book 4 shares the pin's whole vocabulary, so it is measured over strictly
    # more of it: same line, same shape, a count that dwarfs book 2's.
    assert line(body, 4, "labour")["min_local_terms"] > thin["min_local_terms"]

    # Book 3 shares only the query leaf, which is excluded from the anchors: no
    # neighbourhood at all, so it is absent from every line rather than drawn
    # with a null value -- distinct from book 2's merely-thin measurement.
    assert line_or_none(body, 3, "labour") is None
    assert lines_of(body, 3) == []


def test_chart_unknown_pinned_book_returns_404(client, patch_tables):
    # Pinned, the same empty book means there is no expression to chart against
    # at all -- the existing expression_absent finding, not a 500.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2)})

    response = client.post(
        "/semantic-drift/999",
        json={"tree": {"term": "labour"}, "book_ids": [1, 2]},
    )

    assert response.status_code == 404
    assert response.json()["reason"] == "expression_absent"
    assert response.json()["book_id"] == 999


def test_chart_absent_expression_in_pin_never_reads_the_targets(client, patch_tables):
    # The pin decides the 404 on its own, so the targets must not be fetched --
    # the whole point of asking it before the prefetch. Asserted on which books
    # were read, not just the status: the old order returned this same 404 after
    # loading all three.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: book_rows(3)})

    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "absent"}, "book_ids": [2, 3]},
    )

    assert response.status_code == 404
    assert response.json()["reason"] == "expression_absent"
    fetched = [call.args[0] for call in term_table.get_entries.call_args_list]
    assert fetched == [BookIndex(1)]


def test_chart_query_term_absent_from_pinned_book_returns_404(client, patch_tables):
    # A query leaf term absent from the PINNED book means no lines to draw: a
    # dedicated expression_absent 404, distinct from the per-book anchor gap,
    # carrying the numeric pinned book_id (not the internal BookIndex string).
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2)})

    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"op": "+", "args": [{"term": "labour"}, {"term": "missing"}]},
              "book_ids": [2]},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["reason"] == "expression_absent"
    assert body["book_id"] == 1
    assert body["terms"] == ["missing"]


def test_chart_pinned_single_target_returns_404(client, patch_tables):
    # The pinned book does NOT count toward MIN_MATCHING_BOOKS, so a pin plus one
    # target is the query_in_too_few_books finding rather than a chart. The router
    # named the request's `selected_book_id`, a field that does not exist on it, so
    # this was a 500.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2)})

    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2]},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["reason"] == "query_in_too_few_books"
    assert body["book_id"] == 1  # the pin, named even though it carries the query


def test_chart_incomparable_book_is_absent_from_every_line(client, patch_tables):
    # A target sharing too few terms with the PINNED book cannot be measured, so
    # it is absent from every line rather than drawn with a null value. Since
    # every line is measured against the pinned book specifically, there's
    # nothing to measure it from either -- unlike the old corpus-wide
    # prototypicality, a shared peer among the OTHER targets no longer
    # substitutes.
    _, term_table = patch_tables
    pin_vocab = ["labour", "value", "wage", "rent"]
    # Books 2 and 3 share only "labour" with the pinned book (too few for lines),
    # but share alpha/beta/gamma with each other -- irrelevant now, since the mean
    # no longer looks past the pinned book. Book 4 shares the pin's vocabulary, so
    # the chart still has a line to draw and the flag stays a per-book one (every
    # target being incomparable is a 404, covered separately).
    off_vocab = ["labour", "alpha", "beta", "gamma"]
    set_multi_book_table(
        term_table,
        {
            1: book_rows(1, vocab=pin_vocab),
            2: book_rows(2, vocab=off_vocab),
            3: book_rows(3, vocab=off_vocab),
            4: book_rows(4, vocab=pin_vocab),
        },
    )

    body = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2, 3, 4]},
    ).json()

    # Too few shared with the pin: it has the query term, so it keeps its row in
    # `books`, but it is on no line at all.
    assert lines_of(body, 2) == []
    assert 2 in books_by_id(body)  # and it keeps its row
    assert lines_of(body, 4)


# -- /semantic-drift, no pinned book ---------------------------------------


def test_chart_unpinned_scores_every_book_over_peers(client, patch_tables):
    # With no source_book_id in the path, every book gets its own lines, each the
    # mean over peers. The response shape matches the pinned chart's; the only
    # difference is that the OTHER books back each line, so n_books is 2, not 1.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: book_rows(3)})

    response = client.post(
        "/semantic-drift",
        json={"tree": {"term": "labour"}, "book_ids": [1, 2, 3]},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["expr"]["expr"] == "labour"
    assert len(nearest_terms(body)) == len(VOCAB) - 1
    assert "labour" not in nearest_terms(body)  # a term is its own nearest term

    assert [b["id"] for b in body["books"]] == [1, 2, 3]  # every book, no pin dropped
    for book_id in (1, 2, 3):
        # query line first, then one entry per nearest term -- same as the pinned chart
        assert [e["term"] for e in lines_of(body, book_id)] == [
            body["expr"]["expr"],
            *nearest_terms(body),
        ]
        for entry in lines_of(body, book_id):
            assert entry["similarity"] is not None
            assert entry["n_books"] == 2  # the other two books back every line
            lo, hi = entry["similarity_ci"]
            assert lo <= entry["similarity"] <= hi


def test_chart_unpinned_book_missing_query_leaf_is_absent_from_that_line(
    client, patch_tables
):
    # A book lacking the query's leaf carries no entry on its query line; its
    # nearest-term lines (terms it does have) still draw, and the books that do
    # have the term score against each other (n_books drops to 1).
    _, term_table = patch_tables
    lacking = book_rows(3)
    del lacking["labour"]
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: lacking})

    body = client.post(
        "/semantic-drift",
        json={"tree": {"term": "labour"}, "book_ids": [1, 2, 3]},
    ).json()

    assert books_by_id(body)[3]["missing_terms"] == ["labour"]
    assert lines_of(body, 3)  # its other lines still draw
    assert line_or_none(body, 3, "labour") is None
    # Books 1 and 2 have the term; only each other backs their query line now.
    assert line(body, 1, "labour")["similarity"] is not None
    assert line(body, 1, "labour")["n_books"] == 1


def test_chart_unpinned_query_in_too_few_books_returns_404(client, patch_tables):
    # Only one book has the query term -> no cross-book agreement to draw. There is
    # no chart to return, so it is a finding rather than a 200 whose emptiness the
    # client has to detect -- with a null book_id, since unpinned there is no book
    # for the finding to point at.
    _, term_table = patch_tables
    lacking2, lacking3 = book_rows(2), book_rows(3)
    del lacking2["labour"]
    del lacking3["labour"]
    set_multi_book_table(term_table, {1: book_rows(1), 2: lacking2, 3: lacking3})

    response = client.post(
        "/semantic-drift",
        json={"tree": {"term": "labour"}, "book_ids": [1, 2, 3]},
    )

    assert response.status_code == 404
    # The finding names only itself and the pin -- it does not say which books
    # lacked the term, so the body is exactly these two fields.
    assert response.json() == {"reason": "query_in_too_few_books", "book_id": None}


def test_chart_reports_an_incomparable_book_alike_pinned_and_unpinned(
    client, patch_tables, mock_books_cache
):
    # The one claim behind deciding availability from the scoring outcome rather
    # than from vocabularies up front: both branches answer alike. Book 3 meets the
    # others only at the query leaf, so it has no neighbourhood with any of them --
    # and that must read identically whether or not a book is pinned, since it is
    # the same fact about book 3 either way. What carries the answer is now its
    # absence from every line rather than a flag, but the symmetry is the same.
    _, term_table = patch_tables
    books = {
        1: book_rows(1),
        2: book_rows(2),
        3: book_rows(3, vocab=["labour", "alpha", "beta"]),
    }

    def rows_for(path, book_ids):
        set_multi_book_table(term_table, books)
        mock_books_cache.cache.clear()  # each request re-reads the same three books
        body = client.post(
            path, json={"tree": {"term": "labour"}, "book_ids": book_ids}
        ).json()
        return books_by_id(body), body

    unpinned, unpinned_body = rows_for("/semantic-drift", [1, 2, 3])
    pinned, pinned_body = rows_for("/semantic-drift/1", [2, 3])

    for by_id, body in ((unpinned, unpinned_body), (pinned, pinned_body)):
        assert lines_of(body, 3) == []
        # It HAS the query -- what it lacks is a peer to be read against, so the
        # query term is conspicuously not among what its row reports missing.
        assert "labour" not in by_id[3]["missing_terms"]
        # ...while the book that does share a neighbourhood still draws.
        assert lines_of(body, 2)


def test_chart_unpinned_incomparable_book_does_not_shorten_other_seeds(
    client, patch_tables
):
    # Book 3 carries the query but shares no anchor with anyone, so it is skipped
    # as a peer and backs nobody's mean. Its seed count must be skipped with it:
    # books 1 and 2 average each other alone and are entitled to every seed that
    # pair has. Truncating them to book 3's three widened their intervals on
    # evidence that was never in them -- an unrelated book in book_ids could push
    # a CI from excluding zero to straddling it.
    _, term_table = patch_tables
    set_multi_book_table(
        term_table,
        {
            1: book_rows(1),
            2: book_rows(2),
            3: book_rows(3, vocab=["labour", "alpha", "beta"], n_seeds=3),
        },
    )

    def query_line(book_ids):
        body = client.post(
            "/semantic-drift", json={"tree": {"term": "labour"}, "book_ids": book_ids}
        ).json()
        return body, line(body, 1, "labour")

    body, with_incomparable = query_line([1, 2, 3])
    _, without = query_line([1, 2])

    assert with_incomparable["n_seeds"] == 5  # books 1 and 2 both carry five seeds
    assert with_incomparable["n_books"] == 1  # book 3 backs nothing, and costs nothing
    # Book 3's presence is invisible to the books it can't be compared with.
    assert with_incomparable == without
    # It has no comparable peer of its own, so it is on no line at all, exactly
    # as the pinned chart leaves it.
    assert lines_of(body, 3) == []
    assert 3 in books_by_id(body)  # the row survives too


def test_chart_unpinned_nearest_terms_not_restricted_to_one_books_vocabulary(
    client, patch_tables
):
    # Book 1 lacks "capital"; books 2 and 3 both carry it. Nearest-term candidates
    # must be drawn from the UNION of every carrying book's vocabulary, not just
    # whichever book happens to frame the ranking -- so "capital" is still
    # eligible on the strength of books 2 and 3, even though it can never be
    # measured in book 1's own row.
    _, term_table = patch_tables
    vocab_missing_capital = [term for term in VOCAB if term != "capital"]
    set_multi_book_table(
        term_table,
        {
            1: book_rows(1, vocab=vocab_missing_capital),
            2: book_rows(2),
            3: book_rows(3),
        },
    )

    response = client.post(
        "/semantic-drift",
        json={"tree": {"term": "labour"}, "book_ids": [1, 2, 3]},
    )

    assert response.status_code == 200
    assert "capital" in nearest_terms(response.json())


# -- storage-order independence -----------------------------------------------


def test_chart_is_unaffected_by_the_order_rows_arrive_in(
    client, patch_tables, mock_books_cache
):
    # get_local_cosine_similarity pairs two books' shared terms ELEMENTWISE, which
    # is their intersection only while both vocabularies are in the same order. That
    # holds in production solely because the platform_data-index range key is
    # `term`, so DynamoDB hands them over sorted -- nothing the service controls.
    # Unsorted input does not fail, it silently compares one book's term against
    # another's, so this pins the numbers rather than the ordering: same books,
    # same vectors, rows served in a different order, identical chart.
    _, term_table = patch_tables

    def chart_with(book_3_vocab):
        mock_books_cache.cache.clear()
        set_multi_book_table(
            term_table,
            {
                1: book_rows(1),
                2: book_rows(2),
                # Same term -> same seed offset either way, so the VECTORS are
                # identical between the two runs and only their row order moves.
                3: {term: book_rows(3)[term] for term in book_3_vocab},
            },
        )
        return client.post(
            "/semantic-drift/1", json={"tree": {"term": "labour"}, "book_ids": [2, 3]}
        ).json()

    sorted_rows = chart_with(sorted(VOCAB))
    shuffled_rows = chart_with(list(reversed(VOCAB)))

    assert sorted_rows == shuffled_rows


# -- request validation -------------------------------------------------------
#
# These cover the shared expression-tree validators (check_tree_depth,
# TermNode.term_not_empty) and the request_validation_handler, and they are the
# only coverage of any of them -- they moved here verbatim from the deleted
# test_similarity.py when /similar-terms/* was removed. /semantic-drift is now
# the only route taking an ExprTree.


def test_chart_blank_term_returns_422(client, patch_tables):
    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "   "}, "book_ids": [2]},
    )
    assert response.status_code == 422


def test_chart_pinned_book_among_targets_returns_422(client, patch_tables):
    # The pinned book agrees with itself 1.0 for every term, so it has no row to
    # draw: naming it among its own targets is a client bug. This is the one rule
    # the body model can't check alone -- the pin arrives in the path -- so it
    # also covers the semantic_drift_query dependency recombining the two.
    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [1, 2]},
    )

    assert response.status_code == 422
    assert "1" in response.json()["detail"][0]["msg"]


def test_chart_repeated_book_id_returns_422(client, patch_tables):
    # A repeated target could only draw a duplicate row, so it is a client bug:
    # rejected at the schema, with the offending id named in the message rather
    # than quietly collapsed into one row.
    response = client.post(
        "/semantic-drift/1",
        json={"tree": {"term": "labour"}, "book_ids": [2, 2, 2]},
    )

    assert response.status_code == 422
    assert "2" in response.json()["detail"][0]["msg"]


def test_chart_unpinned_route_ignores_a_source_book_id_query_param(
    client, patch_tables
):
    # `source_book_id` belongs to the path. Taken as an argument of a shared
    # dependency it had no path param to bind to on this route, so FastAPI made it a
    # QUERY parameter and `?source_book_id=1` quietly returned a pinned chart --
    # a second, undocumented way to pin. Book 1 must stay an ordinary target: were
    # it the pin, naming it among book_ids would be a 422 instead.
    _, term_table = patch_tables
    set_multi_book_table(term_table, {1: book_rows(1), 2: book_rows(2), 3: book_rows(3)})

    response = client.post(
        "/semantic-drift?source_book_id=1",
        json={"tree": {"term": "labour"}, "book_ids": [1, 2, 3]},
    )

    assert response.status_code == 200
    body = response.json()
    assert [b["id"] for b in body["books"]] == [1, 2, 3]
    assert line(body, 1, "labour")["n_books"] == 2  # peers back it, not a pin


def test_chart_non_numeric_pinned_book_returns_422(client, patch_tables):
    # The pin is still a declared `int` path param, so a junk one is rejected at the
    # boundary rather than reaching the handler as a string.
    response = client.post(
        "/semantic-drift/abc",
        json={"tree": {"term": "labour"}, "book_ids": [2]},
    )
    assert response.status_code == 422


def test_chart_tree_too_deep_returns_422(client, patch_tables):
    # A schema-valid but too-deep tree is a clean 422, not a RecursionError.
    tree = {"term": "labour"}
    for _ in range(MAX_TREE_DEPTH + 2):  # nest past the cap
        tree = {"op": "+", "args": [tree, {"term": "value"}]}

    response = client.post(
        "/semantic-drift/1", json={"tree": tree, "book_ids": [2]}
    )
    assert response.status_code == 422


# -- documented error responses -----------------------------------------------


def test_error_responses_reach_the_openapi_schema(client, patch_tables):
    # post_route expands a bare response model into OpenAPI's {"model",
    # "description"} form. It once rebound its own `responses` argument to {}
    # before reading it, so every documented error body was silently dropped and
    # nothing noticed -- no test read the schema. This one does.
    schema = client.get("/openapi.json").json()

    chart = schema["paths"]["/semantic-drift/{source_book_id}"]["post"]["responses"]
    body = chart["404"]["content"]["application/json"]["schema"]
    # Both findings share the 404 -- `responses` is keyed by status code -- so it
    # documents as the union the client discriminates on `reason`.
    assert [ref["$ref"].split("/")[-1] for ref in body["anyOf"]] == [
        "ExpressionAbsentResponse",
        "QueryInTooFewBooksResponse",
    ]
    assert chart["404"]["description"].startswith("One or more terms")
    assert "pinned book" in chart["422"]["description"]  # a plain dict, passed through

    # Unpinned there is no pinned book for expression_absent to name, so that
    # route documents only the finding it can actually return.
    unpinned = schema["paths"]["/semantic-drift"]["post"]["responses"]
    assert unpinned["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "QueryInTooFewBooksResponse"
    )

    # The pin is a path parameter of the pinned route and nothing at all on the
    # unpinned one. A shared dependency taking it as an argument documented (and
    # accepted) it as a `?source_book_id=` QUERY param on BOTH.
    assert schema["paths"]["/semantic-drift"]["post"].get("parameters") is None
    assert [
        (p["name"], p["in"], p["schema"]["type"])
        for p in schema["paths"]["/semantic-drift/{source_book_id}"]["post"]["parameters"]
    ] == [("source_book_id", "path", "integer")]

    # The other branch, on the route that mixes both forms.
    describe = schema["paths"]["/parse-describe"]["post"]["responses"]
    assert describe["400"]["description"].startswith("The LLM output")
    assert describe["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "TermResolutionResponse"
    )


# -- BooksTermCache.warm ------------------------------------------------------


def _warm_cache(rows=None, side_effect=None):
    """A cache over a term-table mock whose get_entries serves one book's rows."""
    rows = book_rows(1) if rows is None else rows
    table = MagicMock()
    table.get_entries.side_effect = side_effect or (
        lambda book_id, fields=None: list(rows.values())
    )
    return BooksTermCache(table)


def _fetched(books):
    """The books get_entries was called with, read off the mock behind the cache."""
    table = cast(MagicMock, books.table)
    return [call.args[0] for call in table.get_entries.call_args_list]


def test_warm_loads_books_concurrently():
    # The whole point of the prefetch. Every mocked fetch waits on a barrier, so
    # the call can only return if all three loads are in flight at once -- a
    # sequential loader blocks on the first and fails on timeout rather than
    # quietly passing slowly.
    book_ids = [BookIndex(i) for i in (1, 2, 3)]
    barrier = threading.Barrier(len(book_ids), timeout=10)
    rows = book_rows(1)

    def get_entries(book_id, fields=None):
        barrier.wait()
        return list(rows.values())

    books = _warm_cache(side_effect=get_entries)
    books.warm_cache(book_ids)

    assert set(books.cache) == set(book_ids)


def test_warm_skips_books_already_cached():
    cached, fresh = BookIndex(1), BookIndex(2)
    books = _warm_cache()
    # membership is all the prefetch checks, so any object will do
    books.cache[cached] = cast(BookTermVectors, object())

    books.warm_cache([cached, fresh])

    assert _fetched(books) == [fresh]


def test_warm_records_cache_warmed_on_the_request_log():
    # load writes `cache_warmed` through a ContextVar, which a bare worker thread
    # would not inherit -- the field would vanish silently.
    book_ids = [BookIndex(1), BookIndex(2)]
    log: dict = {}
    token = request_log.set(log)
    try:
        _warm_cache().warm_cache(book_ids)
    finally:
        request_log.reset(token)

    assert set(log["cache_warmed"]) == {str(book_id) for book_id in book_ids}


def test_warm_propagates_a_failed_load():
    # A storage failure must still surface as a 500, exactly as it did when the
    # loads ran inline -- not be swallowed by the pool.
    def explode(book_id, fields=None):
        raise RuntimeError("dynamodb unavailable")

    with pytest.raises(RuntimeError, match="dynamodb unavailable"):
        _warm_cache(side_effect=explode).warm_cache([BookIndex(1), BookIndex(2)])


def test_warm_fetches_a_repeated_book_once():
    # The membership check runs at SUBMIT time, so it only skips a book whose load
    # already finished -- which for a real fetch is none of them. The mock has to
    # be slow to show that: with an instant one, the first load lands inside the
    # submitting thread's own GIL slice and the duplicate is skipped by luck, so
    # the test passes against a loader that has no dedupe at all.
    pinned = BookIndex(1)
    rows = book_rows(1)

    def get_entries(book_id, fields=None):
        time.sleep(0.05)  # a DynamoDB round trip, not a dict lookup
        return list(rows.values())

    books = _warm_cache(side_effect=get_entries)

    books.warm_cache([pinned, pinned, BookIndex(2)])

    assert sorted(_fetched(books)) == [pinned, BookIndex(2)]


# -- BooksTermCache row filtering and lookups ---------------------------------


def test_load_skips_adverbs_and_rows_without_vectors():
    # Adverb-only rows are excluded corpus-wide (the same `tags == {"R"}` rule
    # /terms and the vocabulary use), and a row that was never given vectors has
    # no column to contribute -- it used to raise KeyError here.
    rows = [
        make_term_entry("labour"),
        make_term_entry("quickly", tags={"R"}),
        make_term_entry("value", tags={"N", "R"}),  # not adverb-ONLY, so it stays
        {"term": "unvectorized", "tags": {"N"}},
    ]

    books = _warm_cache(rows={entry["term"]: entry for entry in rows})
    book_term_vectors = books.load_book(BookIndex(1))

    assert list(book_term_vectors.terms) == ["labour", "value"]
    assert book_term_vectors.missing_terms(["labour", "quickly", "unvectorized"]) == {
        "quickly",
        "unvectorized",
    }


def test_load_of_a_book_whose_every_row_is_filtered_is_empty_not_an_error():
    # Same shape as a book_id the table knows nothing about: no vocabulary, no
    # exception -- the missing-terms path reports it from there.
    rows = {"quickly": make_term_entry("quickly", tags={"R"})}

    book_term_vectors = _warm_cache(rows=rows).load_book(BookIndex(1))

    assert not len(book_term_vectors.terms)
    assert book_term_vectors.missing_terms(["labour"]) == {"labour"}


def test_get_missing_terms_asks_only_the_books_named():
    # The cache outlives the request, so it holds books this request never asked
    # about -- book 3 here. Answering from the whole cache would put it on the
    # chart. Order follows book_ids, not insertion, since the router reads
    # `valid_book_ids` and the 404's `books` rows straight off this dict.
    books = _warm_cache(rows=book_rows(1))
    for source_id in (3, 2, 1):
        books.load_book(BookIndex(source_id))
    books.cache[BookIndex(2)] = BookTermVectors([], [])

    book_ids = [BookIndex(2), BookIndex(1)]
    assert books.get_missing_terms(book_ids, ["labour", "wage"]) == {
        BookIndex(2): {"labour", "wage"},
        BookIndex(1): set(),
    }
    assert list(books.get_missing_terms(book_ids, ["labour"])) == book_ids


def test_expr_vectors_combine_leaves_by_operator():
    # The `+`/`-` branch of get_expr_vectors, per seed. Expectations are computed
    # here rather than through the service's own normalize_vectors, so a swapped
    # operator or a dropped normalization fails rather than agreeing with itself.
    vectors = _warm_cache(rows=book_rows(1)).load_book(BookIndex(1))
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
    assert not np.allclose(plus, minus)  # the operators are distinguishable
    np.testing.assert_allclose(np.linalg.norm(plus, axis=1), 1.0, rtol=1e-5)

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
    # The matrices are the whole cost of a chart, so the instance must outlive the
    # request that built it -- and be rebuilt only if the table itself is swapped.
    table = MagicMock()

    first = get_books_term_cache(table)

    assert get_books_term_cache(table) is first
    assert get_books_term_cache(MagicMock()) is not first
