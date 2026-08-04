import math

import numpy as np
import pytest

from conftest import VOCAB, book_rows, set_multi_book_table
from app.search.constants import MIN_LOCAL_NEAREST_TERMS, NUM_LOCAL_NEAREST_TERMS
from app.search.schemas.semantic_drift import MAX_TREE_DEPTH
from app.search.services.semantic_drift import (
    MIN_MATCHING_BOOKS,
    NEAREST_TERM_COUNT,
    BooksSimilarityCache,
)
from shared.commons import BookIndex

CORPUS_SIZE = max(MIN_MATCHING_BOOKS, 3)
SELECTED_ID = 1
TARGET_IDS = list(range(2, CORPUS_SIZE + 2))  # the selected book isn't one of them
BOOK_IDS = list(range(1, CORPUS_SIZE + 1))  # unselected: every book counts itself
SPARE_ID = CORPUS_SIZE + 1  # one past the floor: it can drop out and leave a result


def default_books(book_ids=None, **kwargs):
    """The standard corpus: every book carries the whole vocabulary, selected first."""
    if book_ids is None:
        book_ids = [SELECTED_ID, *TARGET_IDS]
    return {book_id: book_rows(book_id, **kwargs) for book_id in book_ids}


def make_fixed_term_entry(term, vector, n_seeds=5, count=100, tags=None):
    arr16 = np.array([vector], dtype=np.float16).repeat(n_seeds, axis=0)
    return {
        "term": term,
        "count_": count,
        "tags": tags if tags is not None else {"N"},
        "vectors": [bytes(arr16[i].tobytes()) for i in range(n_seeds)],
    }


def set_book_years(pipeline_table, years):
    pipeline_table.get_all_entries.return_value = [
        {
            "platform_data": f"gutenberg-{book_id}",
            "s3_prefix_models": f"models/{book_id}",
            **({} if year is None else {"published_year": year}),
        }
        for book_id, year in years.items()
    ]


@pytest.fixture
def post_semantic_drift(client, term_table):
    def post(books=None, tree=None, book_ids=None, selected=SELECTED_ID, **body):
        if books is None:
            books = default_books() if selected is not None else default_books(BOOK_IDS)
        set_multi_book_table(term_table, books)
        if book_ids is None:
            book_ids = TARGET_IDS if selected is not None else BOOK_IDS
        path = "/semantic-drift" if selected is None else f"/semantic-drift/{selected}"
        return client.post(
            path,
            json={"tree": tree or {"term": "labour"}, "book_ids": book_ids, **body},
        )

    return post



def term_books(body):
    """Every scored term, the query's first, as (term, books) pairs."""
    return [(body["expr"]["expr"], body["expr"]["books"])] + [
        (term_data["term"], term_data["books"]) for term_data in body["nearest_terms"]
    ]


def nearest_terms(body):
    """Just the nearest-term labels, which the payload no longer lists flat."""
    return [term_data["term"] for term_data in body["nearest_terms"]]


def book_scores(body, book_id):
    """Every term one book was scored on, in payload order."""
    return [
        dict(book_data, term=term)
        for term, books in term_books(body)
        for book_data in books
        if book_data["book_id"] == book_id
    ]


def score_or_none(body, book_id, term):
    """One book's score for one term, or None where it was not measured for it."""
    books = next(
        books for scored_term, books in term_books(body) if scored_term == term
    )
    return next(
        (book_data for book_data in books if book_data["book_id"] == book_id), None
    )


def score(body, book_id, term):
    """One book's score for one term, which the caller expects to be present."""
    found = score_or_none(body, book_id, term)
    assert found is not None, f"book {book_id} was not measured for {term!r}"
    return found


def books_by_id(body):
    return {b["id"]: b for b in body["books"]}


def assert_response_shape(body, expected_book_ids, expected_n_books):
    assert [b["id"] for b in body["books"]] == expected_book_ids
    assert len(nearest_terms(body)) == NEAREST_TERM_COUNT
    assert body["expr"]["expr"] not in nearest_terms(body)
    assert sum(len(books) for _, books in term_books(body)) == len(
        expected_book_ids
    ) * (NEAREST_TERM_COUNT + 1)
    for book_id in expected_book_ids:
        assert [e["term"] for e in book_scores(body, book_id)] == [
            body["expr"]["expr"],
            *nearest_terms(body),
        ]
        for entry in book_scores(body, book_id):
            assert entry["similarity"] is not None  # measured, so not a gap
            assert entry["min_local_terms"] == NUM_LOCAL_NEAREST_TERMS
            assert entry["n_seeds"] == 5
            assert entry["n_books"] == expected_n_books
            lo, hi = entry["similarity_ci"]
            assert math.isfinite(lo) and math.isfinite(hi)
            assert lo < entry["similarity"] < hi
    for _, books in term_books(body):
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



def test_comparative_happy_path(post_semantic_drift):
    response = post_semantic_drift()

    assert response.status_code == 200
    body = response.json()

    assert body["expr"]["expr"] == "labour"
    assert body["expr"]["terms"] == ["labour"]
    assert set(nearest_terms(body)).issubset(set(VOCAB))

    assert_response_shape(body, TARGET_IDS, expected_n_books=1)


def test_comparative_scores_a_compound_expression(post_semantic_drift):
    response = post_semantic_drift(
        tree={"op": "+", "args": [{"term": "labour"}, {"term": "value"}]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["expr"]["expr"] == "labour + value"  # outermost parens stripped
    assert body["expr"]["terms"] == ["labour", "value"]
    assert not {"labour", "value"} & set(nearest_terms(body))
    assert all(
        entry["similarity"] is not None for entry in book_scores(body, TARGET_IDS[0])
    )


def fixed_similarity_entry(term, similarity):
    """A term sitting at exactly `similarity` to `labour`'s [1.0, 0.0]."""
    return make_fixed_term_entry(term, [similarity, math.sqrt(1 - similarity**2)])


def filler_rows():
    """Terms that exist only to clear MIN_LOCAL_NEAREST_TERMS, for the fixed-vector
    corpora. Identical in every book, so they carry no trend, and all further from
    the query than any term those tests name, so they cannot displace one."""
    return {
        f"filler{n:03d}": fixed_similarity_entry(f"filler{n:03d}", 0.05 + n / 400)
        for n in range(NUM_LOCAL_NEAREST_TERMS)
    }


def trend_book_rows(trending_similarity, flat_similarity=0.8):
    """One book's rows for the sort="slope" tests: `labour` is the query anchor,
    `flat` sits at a constant, high similarity to it in every book, and
    `trending`'s similarity is whatever the caller passes -- fixed vectors (see
    make_fixed_term_entry), so these similarities are exact, not RNG-approximate.

    Ranking never needs more vocabulary than that, so this stays at three terms.
    A test that also reads the scores needs the pair to clear
    MIN_LOCAL_NEAREST_TERMS, and merges in filler_rows() itself.
    """
    return {
        "labour": make_fixed_term_entry("labour", [1.0, 0.0]),
        "trending": fixed_similarity_entry("trending", trending_similarity),
        "flat": fixed_similarity_entry("flat", flat_similarity),
    }


def trend_years(book_ids=BOOK_IDS):
    """One publication year per book, a generation apart."""
    return {book_id: 1800 + 20 * index for index, book_id in enumerate(book_ids)}


def rising_similarities(book_ids=BOOK_IDS):
    """Similarity climbing 0.1 -> 0.9 across the corpus, whatever its size. The
    mean stays 0.5, below `flat`'s constant 0.8, so the two sorts disagree by
    construction rather than by how many books the corpus happens to hold."""
    last = max(len(book_ids) - 1, 1)
    return {book_id: 0.1 + 0.8 * index / last for index, book_id in enumerate(book_ids)}


def test_semantic_drift_slope_sort_ranks_a_drifting_term_over_a_flatter_closer_one(
    post_semantic_drift, pipeline_table
):
    trending = rising_similarities()
    books = {book_id: trend_book_rows(trending[book_id]) for book_id in BOOK_IDS}
    set_book_years(pipeline_table, trend_years())

    mean_body = post_semantic_drift(
        books=books, selected=None, sort="mean_similarity"
    ).json()
    assert nearest_terms(mean_body) == ["flat", "trending"]

    slope_body = post_semantic_drift(books=books, selected=None, sort="slope").json()
    assert nearest_terms(slope_body) == ["trending", "flat"]


def test_semantic_drift_drops_a_term_below_the_book_coverage_floor(
    post_semantic_drift, pipeline_table
):
    rare_book_ids = BOOK_IDS[: MIN_MATCHING_BOOKS - 1]
    rare_similarities = rising_similarities(rare_book_ids)
    books = {}
    for book_id in BOOK_IDS:
        rows = {
            "labour": make_fixed_term_entry("labour", [1.0, 0.0]),
            "common": make_fixed_term_entry("common", [0.2, math.sqrt(1 - 0.2**2)]),
        }
        if book_id in rare_similarities:
            # Nearer the query than `common` ever gets, and climbing.
            s = 0.9 + 0.09 * rare_similarities[book_id]
            rows["rare"] = make_fixed_term_entry("rare", [s, math.sqrt(1 - s**2)])
        books[book_id] = rows
    set_book_years(pipeline_table, trend_years())

    for sort in ("mean_similarity", "slope"):
        body = post_semantic_drift(books=books, selected=None, sort=sort).json()
        assert nearest_terms(body) == ["common"], sort


def test_semantic_drift_slope_sort_fits_around_a_book_with_no_published_year(
    post_semantic_drift, pipeline_table
):
    trending = rising_similarities()
    set_book_years(pipeline_table, {**trend_years(), BOOK_IDS[-1]: None})

    response = post_semantic_drift(
        # Padded: this test reads the scores, not just the ranking.
        books={
            book_id: {**trend_book_rows(trending[book_id]), **filler_rows()}
            for book_id in BOOK_IDS
        },
        selected=None,
        sort="slope",
    )

    assert response.status_code == 200
    # `trending` is the only term that moves at all, so it leads however many
    # flat-by-construction filler terms come after it.
    assert nearest_terms(response.json())[0] == "trending"
    assert {book["book_id"] for book in response.json()["expr"]["books"]} == set(
        BOOK_IDS
    )


def test_comparative_resolves_from_loaded_matrix_without_per_term_fetches(
    post_semantic_drift, term_table
):
    response = post_semantic_drift()

    assert response.status_code == 200
    assert term_table.batch_get_entries.call_count == 0
    # one matrix load per book -- the targets plus the selected one -- then cached
    assert term_table.get_entries.call_count == len(TARGET_IDS) + 1


def test_comparative_thin_local_terms_are_not_measured_at_all(post_semantic_drift):
    thin_id, bare_id, *full_ids = TARGET_IDS
    body = post_semantic_drift(
        books={
            SELECTED_ID: book_rows(SELECTED_ID),
            thin_id: book_rows(thin_id, vocab=VOCAB[: MIN_LOCAL_NEAREST_TERMS - 1]),
            bare_id: book_rows(bare_id, vocab=["labour", "alpha", "beta"]),
            **default_books(full_ids),
        }
    ).json()

    assert book_scores(body, thin_id) == []
    assert score_or_none(body, thin_id, "labour") is None

    assert books_by_id(body)[thin_id]["missing_terms"] == []

    # A full book clears the floor and is measured over exactly the target count.
    full = score(body, full_ids[0], "labour")
    assert full["similarity"] is not None
    assert full["min_local_terms"] == NUM_LOCAL_NEAREST_TERMS
    lo, hi = full["similarity_ci"]
    assert lo <= full["similarity"] <= hi

    assert score_or_none(body, bare_id, "labour") is None
    assert book_scores(body, bare_id) == []


def test_comparative_unknown_selected_book_returns_404(post_semantic_drift):
    response = post_semantic_drift(
        books=default_books(BOOK_IDS), book_ids=BOOK_IDS, selected=999
    )

    assert response.status_code == 404
    assert response.json() == {
        "reason": "expression_absent",
        "book_id": 999,
        "terms": ["labour"],
    }


def test_comparative_absent_expression_never_reads_the_targets(
    post_semantic_drift, term_table
):

    response = post_semantic_drift(tree={"term": "absent"})

    assert response.status_code == 404
    assert response.json()["reason"] == "expression_absent"
    fetched = [call.args[0] for call in term_table.get_entries.call_args_list]
    assert fetched == [BookIndex(SELECTED_ID)]


def test_comparative_query_term_absent_from_selected_book_returns_404(
    post_semantic_drift,
):

    response = post_semantic_drift(
        tree={"op": "+", "args": [{"term": "labour"}, {"term": "missing"}]}
    )

    assert response.status_code == 404
    assert response.json() == {
        "reason": "expression_absent",
        "book_id": SELECTED_ID,
        "terms": ["missing"],
    }


def test_comparative_too_few_targets_returns_404(post_semantic_drift):

    response = post_semantic_drift(book_ids=TARGET_IDS[: MIN_MATCHING_BOOKS - 1])

    assert response.status_code == 404
    body = response.json()
    assert body["reason"] == "query_in_too_few_books"
    # the selected book, named even though it carries the query
    assert body["book_id"] == SELECTED_ID


# -- /semantic-drift, no selected book -----------------------------------------


def test_semantic_drift_scores_every_book_over_peers(post_semantic_drift):

    response = post_semantic_drift(selected=None)

    assert response.status_code == 200
    body = response.json()

    assert body["expr"]["expr"] == "labour"
    # Every book keeps a row -- none is held back -- and every OTHER book backs
    # each score, where the comparative route has the selected one behind all.
    assert_response_shape(body, BOOK_IDS, expected_n_books=len(BOOK_IDS) - 1)


def test_semantic_drift_book_missing_query_leaf_is_absent_from_that_term(
    post_semantic_drift,
):

    lacking = book_rows(SPARE_ID)
    del lacking["labour"]

    body = post_semantic_drift(
        books={**default_books(BOOK_IDS), SPARE_ID: lacking},
        book_ids=[*BOOK_IDS, SPARE_ID],
        selected=None,
    ).json()

    assert books_by_id(body)[SPARE_ID]["missing_terms"] == ["labour"]
    assert book_scores(body, SPARE_ID)  # its other terms are still scored
    assert score_or_none(body, SPARE_ID, "labour") is None
    # The rest have the term; only each other backs their query score now.
    assert score(body, BOOK_IDS[0], "labour")["similarity"] is not None
    assert score(body, BOOK_IDS[0], "labour")["n_books"] == len(BOOK_IDS) - 1


def test_semantic_drift_query_in_too_few_books_returns_404(post_semantic_drift):

    books = default_books(BOOK_IDS)
    for book_id in BOOK_IDS[1:]:  # every book but the first loses the query term
        del books[book_id]["labour"]

    response = post_semantic_drift(books=books, selected=None)

    assert response.status_code == 404
    assert response.json() == {"reason": "query_in_too_few_books", "book_id": None}


def test_incomparable_book_scored_alike_with_and_without_selection(
    post_semantic_drift, mock_books_cache
):

    *shared_ids, lone_id = TARGET_IDS
    books = {
        **default_books([SELECTED_ID, *shared_ids]),
        lone_id: book_rows(lone_id, vocab=["labour", "alpha", "beta"]),
    }

    def body_for(selected, book_ids):
        mock_books_cache.books_term_vectors.clear()  # each request re-reads the books
        return post_semantic_drift(
            books=books, book_ids=book_ids, selected=selected
        ).json()

    unselected_body = body_for(None, [SELECTED_ID, *TARGET_IDS])
    selected_body = body_for(SELECTED_ID, TARGET_IDS)

    for body in (unselected_body, selected_body):
        assert book_scores(body, lone_id) == []
        # It HAS the query -- what it lacks is a peer to be read against, so the
        # query term is conspicuously not among what its row reports missing.
        assert "labour" not in books_by_id(body)[lone_id]["missing_terms"]
        # ...while a book that does share nearest terms is still scored.
        assert book_scores(body, shared_ids[0])


def test_semantic_drift_incomparable_book_does_not_shorten_other_seeds(
    post_semantic_drift,
):

    books = {
        **default_books(BOOK_IDS),
        SPARE_ID: book_rows(SPARE_ID, vocab=["labour", "alpha", "beta"], n_seeds=3),
    }

    def query_score(book_ids):
        body = post_semantic_drift(books=books, book_ids=book_ids, selected=None).json()
        return body, score(body, BOOK_IDS[0], "labour")

    body, with_incomparable = query_score([*BOOK_IDS, SPARE_ID])
    _, without = query_score(BOOK_IDS)

    assert with_incomparable["n_seeds"] == 5  # every other book carries five seeds
    # the spare backs nothing, and costs nothing
    assert with_incomparable["n_books"] == len(BOOK_IDS) - 1
    # The spare's presence is invisible to the books it can't be compared with.
    assert with_incomparable == without
    # It has no comparable peer of its own, so it is measured nowhere, exactly as
    # the comparative route leaves it.
    assert book_scores(body, SPARE_ID) == []
    assert SPARE_ID in books_by_id(body)  # the row survives too


def test_semantic_drift_nearest_terms_not_restricted_to_one_books_vocabulary(
    post_semantic_drift,
):
    lacking_id, *carrying_ids = [*BOOK_IDS, SPARE_ID]

    def rows(with_capital):
        return {
            "labour": make_fixed_term_entry("labour", [1.0, 0.0]),
            **(
                {"capital": fixed_similarity_entry("capital", 0.9)}
                if with_capital
                else {}
            ),
            **filler_rows(),
        }

    response = post_semantic_drift(
        books={
            lacking_id: rows(with_capital=False),
            **{book_id: rows(with_capital=True) for book_id in carrying_ids},
        },
        book_ids=[*BOOK_IDS, SPARE_ID],
        selected=None,
    )

    assert response.status_code == 200
    assert "capital" in nearest_terms(response.json())


# -- storage-order independence -----------------------------------------------


def test_comparative_is_unaffected_by_the_order_rows_arrive_in(
    post_semantic_drift, mock_books_cache
):

    reordered_id = TARGET_IDS[-1]

    def body_with(row_order):
        mock_books_cache.books_term_vectors.clear()
        return post_semantic_drift(
            books={
                **default_books([SELECTED_ID, *TARGET_IDS[:-1]]),
                # Same term -> same seed offset either way, so the VECTORS are
                # identical between the two runs and only their row order moves.
                reordered_id: {
                    term: book_rows(reordered_id)[term] for term in row_order
                },
            }
        ).json()

    sorted_rows = body_with(sorted(VOCAB))
    shuffled_rows = body_with(list(reversed(VOCAB)))

    # Two identical error bodies would satisfy the comparison below while
    # measuring nothing, so insist something was actually measured.
    assert nearest_terms(sorted_rows)
    assert sorted_rows == shuffled_rows


# -- request validation -------------------------------------------------------


def deep_tree():
    """A schema-valid expression nested past the depth cap."""
    tree = {"term": "labour"}
    for _ in range(MAX_TREE_DEPTH + 2):
        tree = {"op": "+", "args": [tree, {"term": "value"}]}
    return tree


@pytest.mark.parametrize(
    "path, body, expected_in_message",
    [
        ("/semantic-drift/1", {"tree": {"term": "   "}, "book_ids": [2]}, None),
        ("/semantic-drift/1", {"tree": {"term": "labour"}, "book_ids": [1, 2]}, "1"),
        ("/semantic-drift/1", {"tree": {"term": "labour"}, "book_ids": [2, 2, 2]}, "2"),
        ("/semantic-drift/abc", {"tree": {"term": "labour"}, "book_ids": [2]}, None),
        ("/semantic-drift/1", {"tree": deep_tree(), "book_ids": [2]}, None),
    ],
    ids=[
        "blank-term",
        "selected-among-targets",
        "repeated-book-id",
        "non-numeric-selected-id",
        "tree-too-deep",
    ],
)
def test_comparative_malformed_request_returns_422(
    client, patch_tables, path, body, expected_in_message
):
    response = client.post(path, json=body)

    assert response.status_code == 422
    if expected_in_message is not None:
        assert expected_in_message in response.json()["detail"][0]["msg"]


def test_semantic_drift_route_ignores_a_source_book_id_query_param(client, term_table):
    set_multi_book_table(term_table, default_books(BOOK_IDS))

    response = client.post(
        f"/semantic-drift?source_book_id={BOOK_IDS[0]}",
        json={"tree": {"term": "labour"}, "book_ids": BOOK_IDS},
    )

    assert response.status_code == 200
    body = response.json()
    assert [b["id"] for b in body["books"]] == BOOK_IDS
    # peers back it, not a selected book
    assert score(body, BOOK_IDS[0], "labour")["n_books"] == len(BOOK_IDS) - 1


# -- documented error responses -----------------------------------------------


def test_error_responses_reach_the_openapi_schema(client, patch_tables):

    schema = client.get("/openapi.json").json()

    comparative = schema["paths"]["/semantic-drift/{source_book_id}"]["post"][
        "responses"
    ]
    body = comparative["404"]["content"]["application/json"]["schema"]
    # Both findings share the 404 -- `responses` is keyed by status code -- so it
    # documents as the union the client discriminates on `reason`.
    assert [ref["$ref"].split("/")[-1] for ref in body["anyOf"]] == [
        "ExpressionAbsentResponse",
        "QueryInTooFewBooksResponse",
    ]
    assert comparative["404"]["description"].startswith("One or more terms")
    # a plain dict, passed through
    assert "selected book" in comparative["422"]["description"]

    # Unselected there is no selected book for expression_absent to name, so that
    # route documents only the finding it can actually return.
    unselected = schema["paths"]["/semantic-drift"]["post"]["responses"]
    assert unselected["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "QueryInTooFewBooksResponse"
    )

    # source_book_id is a path parameter of the comparative route and nothing at
    # all on the other. A shared dependency taking it as an argument documented
    # (and accepted) it as a `?source_book_id=` QUERY param on BOTH.
    assert schema["paths"]["/semantic-drift"]["post"].get("parameters") is None
    assert [
        (p["name"], p["in"], p["schema"]["type"])
        for p in schema["paths"]["/semantic-drift/{source_book_id}"]["post"][
            "parameters"
        ]
    ] == [("source_book_id", "path", "integer")]

    # The other branch, on the route that mixes both forms.
    describe = schema["paths"]["/parse-describe"]["post"]["responses"]
    assert describe["400"]["description"].startswith("The LLM output")
    assert describe["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "TermResolutionResponse"
    )


# -- batched similarity cache, end to end --------------------------------------


def test_semantic_drift_scores_agree_with_and_without_the_batched_cache(
    post_semantic_drift, monkeypatch
):

    with_cache = post_semantic_drift(selected=None).json()

    monkeypatch.setattr(
        BooksSimilarityCache, "warm_cache", lambda self, *args, **kwargs: None
    )
    without_cache = post_semantic_drift(selected=None).json()

    assert nearest_terms(with_cache) == nearest_terms(without_cache)
    assert with_cache["books"] == without_cache["books"]

    batched_terms, plain_terms = term_books(with_cache), term_books(without_cache)
    assert [term for term, _ in batched_terms] == [term for term, _ in plain_terms]
    for (_, batched_books), (_, plain_books) in zip(batched_terms, plain_terms):
        assert [b["book_id"] for b in batched_books] == [
            b["book_id"] for b in plain_books
        ]
        for batched, plain in zip(batched_books, plain_books):
            assert batched["n_seeds"] == plain["n_seeds"]
            assert batched["n_books"] == plain["n_books"]
            assert batched["min_local_terms"] == plain["min_local_terms"]
            np.testing.assert_allclose(
                batched["similarity"], plain["similarity"], atol=1e-6
            )
            np.testing.assert_allclose(
                batched["similarity_ci"], plain["similarity_ci"], atol=1e-6
            )
