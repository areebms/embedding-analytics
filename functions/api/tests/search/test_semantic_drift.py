import math

import numpy as np
import pytest

from conftest import LOCAL_VOCAB_FLOOR, VOCAB, book_rows, set_multi_book_table
from app.search.constants import (
    MIN_BOOKS_WITH_TERM,
    MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS,
    NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY,
    NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING,
    NUM_COMPARATIVE_TERMS,
)
from app.search.schemas.semantic_drift import MAX_TREE_DEPTH
from app.search.services.semantic_drift import BooksSimilarityCache
from app.search.services.semantic_drift.mean_local_similarities import (
    center_locally,
    get_is_local,
    get_mean_local_similarity_per_book,
    get_n_highest_similarities,
)
from shared.commons import BookIndex

CORPUS_SIZE = max(MIN_BOOKS_WITH_TERM, 3)
SELECTED_ID = 1
TARGET_IDS = list(range(2, CORPUS_SIZE + 2))  # the selected book isn't one of them
BOOK_IDS = list(range(1, CORPUS_SIZE + 1))  # unselected: every book counts itself
SPARE_ID = CORPUS_SIZE + 1  # one past the floor: it can drop out and leave a result
FLAT_SIMILARITY = 0.8


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


def nearest_term_data(body):
    return body["comparative_terms"]


def term_books(body):
    """Every scored term, the query's first, as (term, books) pairs."""
    return [(body["expr"]["expr"], body["expr"]["books"])] + [
        (term_data["term"], term_data["books"]) for term_data in nearest_term_data(body)
    ]


def nearest_terms(body):
    return [term_data["term"] for term_data in nearest_term_data(body)]


def scored_terms(body):
    return [term_data["term"] for term_data in nearest_term_data(body)]


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


def assert_response_shape(
    body, expected_book_ids, expected_n_books, *, against_corpus
):
    assert [b["id"] for b in body["books"]] == expected_book_ids
    assert NUM_COMPARATIVE_TERMS <= len(nearest_terms(body)) <= 2 * NUM_COMPARATIVE_TERMS
    assert body["expr"]["expr"] not in scored_terms(body)
    assert sum(len(books) for _, books in term_books(body)) == len(
        expected_book_ids
    ) * (len(scored_terms(body)) + 1)
    for book_id in expected_book_ids:
        assert [e["term"] for e in book_scores(body, book_id)] == [
            body["expr"]["expr"],
            *scored_terms(body),
        ]
        for entry in book_scores(body, book_id):
            assert entry["mean_local_similarity"] is not None  # measured, not a gap
            assert entry["n_seeds"] == 5
            lo, hi = entry["ci"]
            assert math.isfinite(lo) and math.isfinite(hi)
            assert lo < entry["mean_local_similarity"] < hi
            assert entry["occurrences"] > 0  # measured, so the book uses the terms
            # n_books exists only against the corpus; pinned it could only say 1.
            if against_corpus:
                assert entry["n_books"] == expected_n_books
    expected_fields = {
        "book_id",
        "mean_local_similarity",
        "ci",
        "occurrences",
        "n_seeds",
    }
    if against_corpus:
        expected_fields |= {"n_books"}
    for _, books in term_books(body):
        for book_data in books:
            assert set(book_data) == expected_fields
    assert set(body["expr"]) == {"expr", "terms", "books"}
    for term_data in nearest_term_data(body):
        assert set(term_data) == {
            "term",
            "stability",
            "instability",
            "n_books_in",
            "n_books_as_top50",
            "n_books_as_top100",
            "books",
        }
        assert term_data["n_books_in"] >= MIN_BOOKS_WITH_TERM
        assert term_data["instability"] >= 0.0
        # Membership is bounded by vocabulary: a book cannot place a term it
        # does not have, and the floor is what admitted the term at all.
        assert (
            MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS
            <= term_data["n_books_as_top100"]
            <= term_data["n_books_in"]
        )
        assert term_data["n_books_as_top50"] <= term_data["n_books_as_top100"]
    assert all(
        set(b) == {"id", "n_shared_terms", "missing_terms"} for b in body["books"]
    )
    assert all(b["missing_terms"] == [] for b in body["books"])
    # Every book here drew a line, so every one cleared the anchor floor.
    assert all(
        b["n_shared_terms"] >= NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY
        for b in body["books"]
    )


def test_comparative_happy_path(post_semantic_drift):
    response = post_semantic_drift()

    assert response.status_code == 200
    body = response.json()

    assert body["expr"]["expr"] == "labour"
    assert body["expr"]["terms"] == ["labour"]
    assert set(nearest_terms(body)).issubset(set(VOCAB))

    assert_response_shape(body, TARGET_IDS, expected_n_books=1, against_corpus=False)


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
        entry["mean_local_similarity"] is not None
        for entry in book_scores(body, TARGET_IDS[0])
    )


def fixed_similarity_entry(term, similarity):
    """A term sitting at exactly `similarity` to `labour`'s [1.0, 0.0]."""
    return make_fixed_term_entry(term, [similarity, math.sqrt(1 - similarity**2)])


def books_from_similarities(similarities_by_book):
    return {
        book_id: {
            "labour": make_fixed_term_entry("labour", [1.0, 0.0]),
            **{
                term: fixed_similarity_entry(term, similarity)
                for term, similarity in similarities.items()
            },
        }
        for book_id, similarities in similarities_by_book.items()
    }


def filler_similarities():
    return {f"filler{n:03d}": 0.05 + n / 400 for n in range(LOCAL_VOCAB_FLOOR)}


def rising_similarities(book_ids=BOOK_IDS):
    """Similarity climbing 0.1 -> 0.9 across the corpus, whatever its size. The
    mean stays 0.5, below `flat`'s constant 0.8, so the two sorts disagree by
    construction rather than by how many books the corpus happens to hold."""
    last = max(len(book_ids) - 1, 1)
    return {book_id: 0.1 + 0.8 * index / last for index, book_id in enumerate(book_ids)}


def swinging_similarities(book_ids=BOOK_IDS):
    """Similarity alternating book to book. Its mean sits below
    `rising_similarities`'."""
    return {
        book_id: 0.44 if index % 2 else 0.06 for index, book_id in enumerate(book_ids)
    }


def test_semantic_drift_drops_a_term_below_the_book_coverage_floor(
    post_semantic_drift,
):
    rare_book_ids = BOOK_IDS[: MIN_BOOKS_WITH_TERM - 1]
    rare_similarities = rising_similarities(rare_book_ids)
    books = books_from_similarities(
        {
            book_id: {
                "common": 0.2,
                # Nearer the query than `common` ever gets, and climbing.
                **(
                    {"rare": 0.9 + 0.09 * rare_similarities[book_id]}
                    if book_id in rare_similarities
                    else {}
                ),
            }
            for book_id in BOOK_IDS
        }
    )

    body = post_semantic_drift(books=books, selected=None).json()
    assert nearest_terms(body) == ["common"]


def ranking_similarities_by_book(book_ids=BOOK_IDS):
    """Every scored term's similarity in every book: the numbers the ranking
    corpora are built from, which a test reading the statistics back has to
    know. The query's own term is left out, since it never scores."""
    trending = rising_similarities(book_ids)
    swinging = swinging_similarities(book_ids)
    return {
        book_id: {
            "trending": trending[book_id],
            "flat": FLAT_SIMILARITY,
            "swinging": swinging[book_id],
            **filler_similarities(),
        }
        for book_id in book_ids
    }


def swinging_books(book_ids=BOOK_IDS):
    """A corpus carrying all three of `flat`, `trending` and `swinging`, over a
    full complement of nearest terms. The statistics are read against those, so a
    corpus of only the three would let them set their own scale."""
    return books_from_similarities(ranking_similarities_by_book(book_ids))


def local_positions(similarities_by_book):
    """The fixture's own similarities as the response reports them: each book's
    value read against the centre of that book's own nearest terms, since books
    share no scale on which an absolute cosine could be compared.

    Offsets rather than distances, so they sit near zero for a term the book
    holds at about its neighbourhood's usual remove -- compare against these by
    absolute tolerance, not relative, and leave room for the float16 rounding
    in the stored vectors.
    """
    positions = {}
    for book_similarities in similarities_by_book.values():
        values = np.array(list(book_similarities.values()))
        local = np.sort(values)[-NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING:]
        for term, similarity in book_similarities.items():
            positions.setdefault(term, []).append(similarity - local.mean())
    return positions


def test_semantic_drift_returns_a_term_its_spread_alone_selected(
    post_semantic_drift,
):
    body = post_semantic_drift(books=swinging_books(), selected=None).json()
    ranked = nearest_term_data(body)
    by_term = {term_data["term"]: term_data for term_data in ranked}

    assert "swinging" in by_term

    nearer = [
        term_data
        for term_data in ranked
        if term_data["stability"] > by_term["swinging"]["stability"]
    ]
    assert len(nearer) >= NUM_COMPARATIVE_TERMS


def test_semantic_drift_nearest_terms_carry_their_statistics(
    post_semantic_drift,
):
    body = post_semantic_drift(books=swinging_books(), selected=None).json()

    by_term = {term["term"]: term for term in nearest_term_data(body)}
    # The sample variance of exactly the positions the corpus was built to hold.
    expected = local_positions(ranking_similarities_by_book())
    for term in ("swinging", "trending", "flat"):
        assert by_term[term]["instability"] == pytest.approx(
            np.var(expected[term], ddof=1), abs=1e-4
        )
        assert by_term[term]["stability"] == pytest.approx(
            np.mean(expected[term]), abs=1e-3
        )
        assert by_term[term]["n_books_in"] == len(BOOK_IDS)

    # `flat` sits at a constant 0.8 in every book and still spreads: holding one
    # distance while the terms around it move *is* the books disagreeing
    # about where the term sits. It spreads least of the three all the same.
    assert by_term["flat"]["instability"] > 0.0
    assert by_term["flat"]["instability"] < by_term["trending"]["instability"]


def membership_books(book_ids=BOOK_IDS):
    """Three terms every book carries, differing in how many books put them
    among their own nearest ones: `everywhere` in all of them, `sometimes` in
    the later half, `solo` in just one. Below the fillers is outside a book's
    nearest terms, above them is inside, which is what moves the count.

    `solo` swings hardest of the three by construction, but only one book ever
    places it near the query, so both selections gate it out.
    """
    last = len(book_ids) - 1
    half = len(book_ids) // 2
    return books_from_similarities(
        {
            book_id: {
                "everywhere": 0.9 if index < half else 0.4,
                "sometimes": 0.04 if index < half else 0.3,
                "solo": 0.95 if index == last else 0.04,
                **filler_similarities(),
            }
            for index, book_id in enumerate(book_ids)
        }
    )


def test_semantic_drift_drops_a_term_only_one_book_places_near_the_query(
    post_semantic_drift,
):
    body = post_semantic_drift(books=membership_books(), selected=None).json()
    by_term = {term["term"]: term for term in nearest_term_data(body)}

    # One book puts `solo` nearer the query than anything else it carries and
    # the rest bury it below their own nearest terms, which would make it the
    # widest spread in the corpus. It is dropped all the same. A spread taken
    # over a single book's placement measures that book rather than a
    # disagreement between books, so both selections gate on
    # MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS and `solo` clears neither.
    assert "solo" not in by_term

    # Vocabulary is not the same as membership: every book carries all three
    # terms. What separates them is how many books place them among their own
    # nearest, and only that count is allowed to decide.
    assert {"everywhere", "sometimes"} <= set(by_term)
    assert by_term["everywhere"]["n_books_as_top100"] == CORPUS_SIZE
    assert by_term["sometimes"]["n_books_as_top100"] == CORPUS_SIZE // 2
    for term in ("everywhere", "sometimes"):
        assert by_term[term]["n_books_in"] == CORPUS_SIZE
        assert (
            by_term[term]["n_books_as_top100"] >= MIN_BOOKS_WITH_TERM_IN_NEAREST_TERMS
        )


def outlying_books(book_ids=BOOK_IDS):
    return books_from_similarities(
        {
            book_id: {
                "outlying": 0.15 if index < 2 else 0.001,
                "borderline": (0.5, 0.15)[index] if index < 2 else 0.001,
                **filler_similarities(),
            }
            for index, book_id in enumerate(book_ids)
        }
    )


def test_semantic_drift_drops_an_unstable_term_no_book_holds_nearest(
    post_semantic_drift,
):
    body = post_semantic_drift(books=outlying_books(), selected=None).json()
    by_term = {term["term"]: term for term in nearest_term_data(body)}

    assert "outlying" not in by_term

    assert by_term["borderline"]["n_books_as_top50"] == 1
    assert by_term["borderline"]["n_books_as_top100"] == 2


def test_semantic_drift_returns_the_term_the_books_hold_nearest(
    post_semantic_drift,
):
    body = post_semantic_drift(books=swinging_books(), selected=None).json()
    ranked = nearest_term_data(body)

    assert len(ranked) > NUM_COMPARATIVE_TERMS

    nearest = max(ranked, key=lambda term_data: term_data["stability"])
    assert nearest["term"] == "flat"


def offset_similarities_by_book(book_ids=BOOK_IDS):
    """Books that rank the query's nearest terms identically and disagree only
    about how far off it sits: every book's similarities are one base profile
    shifted bodily, the way two texts can hold a whole vocabulary nearer or
    farther without reordering any of it. A shift and not a rescaling, because a
    shift is the whole of what centring claims to remove -- a book whose nearest
    terms are bunched more tightly than another's really does contribute smaller
    offsets, and that difference is left in on purpose.
    """
    span = max(len(book_ids) - 1, 1)
    base = [
        0.1 + 0.3 * n / (LOCAL_VOCAB_FLOOR + 2) for n in range(LOCAL_VOCAB_FLOOR + 3)
    ]
    return {
        book_id: {
            f"term{n:03d}": similarity + 0.45 * index / span
            for n, similarity in enumerate(base)
        }
        for index, book_id in enumerate(book_ids)
    }


def test_semantic_drift_statistics_read_position_not_distance(
    post_semantic_drift,
):
    """Books never share a coordinate frame, so a book that places the whole
    vocabulary farther off is not disagreeing about any term in it."""

    similarities = offset_similarities_by_book()

    body = post_semantic_drift(
        books=books_from_similarities(similarities), selected=None
    ).json()

    # Read as raw cosines, these books disagree about every term in the corpus --
    # all of it an artefact of how far from the query each book happens to sit.
    raw_spreads = [
        np.var([book[term] for book in similarities.values()], ddof=1)
        for term in next(iter(similarities.values()))
    ]
    assert min(raw_spreads) > 0.01

    # Read as position, they agree, because not one of them reordered anything.
    # Measured against the disagreement centring was asked to remove, rather than
    # against a fixed bound, since these are offsets and carry no scale of their
    # own. Three orders of magnitude down, and what is left is float16 rounding.
    negligible = min(raw_spreads) / 1000
    for term_data in nearest_term_data(body):
        assert term_data["instability"] < negligible, term_data["term"]


def test_center_locally_leaves_a_book_with_nothing_to_centre_alone():
    """The query's own terms are masked out before a profile is centred, so a
    book carrying nothing else reaches this with no nearest terms at all."""
    assert (
        len(center_locally(np.array([]), NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING))
        == 0
    )


def test_an_empty_local_profile_is_returned_rather_than_raised_on():
    assert (
        len(
            get_n_highest_similarities(
                np.array([]), NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING
            )
        )
        == 0
    )
    assert (
        len(get_is_local(np.array([]), NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING))
        == 0
    )


def test_semantic_drift_count_is_each_books_own_word_count(post_semantic_drift):
    counts = {book_id: 100 + book_id for book_id in TARGET_IDS}
    books = default_books()
    for book_id, count in counts.items():
        books[book_id] = {
            term: {**entry, "count_": count} for term, entry in books[book_id].items()
        }

    body = post_semantic_drift(books=books).json()

    for book_id, count in counts.items():
        assert [entry["occurrences"] for entry in book_scores(body, book_id)] == [count] * (
            len(scored_terms(body)) + 1
        )


def test_semantic_drift_count_sums_the_leaves_of_a_compound_expression(
    post_semantic_drift,
):
    body = post_semantic_drift(
        tree={"op": "+", "args": [{"term": "labour"}, {"term": "value"}]}
    ).json()

    # Every fixture term is written at the same count, so two leaves is twice one.
    assert all(entry["occurrences"] == 200 for entry in body["expr"]["books"])
    assert all(
        entry["occurrences"] == 100
        for term in nearest_term_data(body)
        for entry in term["books"]
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
            thin_id: book_rows(
                thin_id,
                vocab=VOCAB[: NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY - 1],
            ),
            bare_id: book_rows(bare_id, vocab=["labour", "alpha", "beta"]),
            **default_books(full_ids),
        }
    ).json()

    assert book_scores(body, thin_id) == []
    assert score_or_none(body, thin_id, "labour") is None

    assert books_by_id(body)[thin_id]["missing_terms"] == []
    # Nothing was missing, so `n_shared_terms` is what says why. It is the book's
    # BEST overlap, so falling below the floor proves every comparison failed --
    # the one thing separating this from a book absent for want of vocabulary.
    assert (
        books_by_id(body)[thin_id]["n_shared_terms"]
        == NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY - 1
    )

    # A full book clears the floor and is measured.
    full = score(body, full_ids[0], "labour")
    assert full["mean_local_similarity"] is not None
    lo, hi = full["ci"]
    assert lo <= full["mean_local_similarity"] <= hi
    assert (
        books_by_id(body)[full_ids[0]]["n_shared_terms"]
        >= NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY
    )

    assert score_or_none(body, bare_id, "labour") is None
    assert book_scores(body, bare_id) == []
    assert books_by_id(body)[bare_id]["n_shared_terms"] == 1  # only `labour`


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

    response = post_semantic_drift(book_ids=TARGET_IDS[: MIN_BOOKS_WITH_TERM - 1])

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
    assert_response_shape(
        body, BOOK_IDS, expected_n_books=len(BOOK_IDS) - 1, against_corpus=True
    )


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
    assert score(body, BOOK_IDS[0], "labour")["mean_local_similarity"] is not None
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

    # `n_shared_terms` is the book's BEST overlap, not its worst: the spare is a
    # peer of every book here and shares only `labour` with any of them, and that
    # must not drag down a book the spare could never have stopped from scoring.
    assert books_by_id(body)[BOOK_IDS[0]]["n_shared_terms"] == len(VOCAB)
    assert books_by_id(body)[SPARE_ID]["n_shared_terms"] == 1


def test_corpus_interval_measures_peer_spread_not_seed_spread():
    """Against the corpus the peers are the unit of replication, not the seeds.

    Two peers, each perfectly stable across its own seeds but disagreeing with
    each other, is the case that separates the two estimators: seed spread is
    zero, peer spread is not.
    """

    book_id = BookIndex(SELECTED_ID)
    peers = [np.full(5, 0.2), np.full(5, 0.8)]

    corpus = get_mean_local_similarity_per_book(
        book_id, peers, occurrences=10, against_corpus=True
    )
    pinned = get_mean_local_similarity_per_book(
        book_id, peers, occurrences=10, against_corpus=False
    )

    # Same value either way -- only the interval differs.
    assert corpus.mean_local_similarity == pytest.approx(0.5)
    assert pinned.mean_local_similarity == pytest.approx(0.5)

    # Every seed sees the same average, so seed spread is exactly zero.
    assert pinned.ci[1] - pinned.ci[0] == 0.0
    # The peers disagree, and against the corpus that has to show.
    assert corpus.ci[1] - corpus.ci[0] > 0.0
    assert corpus.ci[0] < corpus.mean_local_similarity < corpus.ci[1]


def test_corpus_interval_falls_back_to_seed_spread_with_one_peer():
    """One peer leaves no between-book variation to estimate."""

    book_id = BookIndex(SELECTED_ID)
    peers = [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]

    corpus = get_mean_local_similarity_per_book(
        book_id, peers, occurrences=10, against_corpus=True
    )
    pinned = get_mean_local_similarity_per_book(
        book_id, peers, occurrences=10, against_corpus=False
    )

    assert corpus.ci == pytest.approx(pinned.ci)
    assert corpus.n_books == 1


def test_semantic_drift_score_is_independent_of_peer_order(post_semantic_drift):

    books = {**default_books(BOOK_IDS), SPARE_ID: book_rows(SPARE_ID, n_seeds=3)}

    def query_score(book_ids):
        body = post_semantic_drift(books=books, book_ids=book_ids, selected=None).json()
        return score(body, BOOK_IDS[0], "labour")

    spare_last = query_score([*BOOK_IDS, SPARE_ID])
    spare_first = query_score([SPARE_ID, *BOOK_IDS])

    assert spare_last["n_seeds"] == 3
    for field in ("book_id", "occurrences", "n_seeds", "n_books"):
        assert spare_last[field] == spare_first[field]
    for field in ("mean_local_similarity", "ci"):
        assert spare_last[field] == pytest.approx(spare_first[field], abs=1e-6)


def test_semantic_drift_nearest_terms_not_restricted_to_one_books_vocabulary(
    post_semantic_drift,
):
    lacking_id, *carrying_ids = [*BOOK_IDS, SPARE_ID]

    # The book lacking `capital` shares nothing else with the others either. Its
    # own terms are in one vocabulary, so the coverage floor drops them, and the
    # fillers do not collect a book `capital` cannot -- which under a
    # presence-led key would outrank it on a count rather than on the reading.
    lacking_similarities = {
        term: 0.9 - index / 20
        for index, term in enumerate(("value", "wage", "rent", "stock", "price"))
    }

    response = post_semantic_drift(
        books=books_from_similarities(
            {
                lacking_id: lacking_similarities,
                **{
                    book_id: {"capital": 0.9, **filler_similarities()}
                    for book_id in carrying_ids
                },
            }
        ),
        book_ids=[*BOOK_IDS, SPARE_ID],
        selected=None,
    )

    assert response.status_code == 200
    assert "capital" in scored_terms(response.json())


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

    assert scored_terms(with_cache) == scored_terms(without_cache)
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
            np.testing.assert_allclose(
                batched["mean_local_similarity"], plain["mean_local_similarity"], atol=1e-6
            )
            np.testing.assert_allclose(
                batched["ci"], plain["ci"], atol=1e-6
            )
