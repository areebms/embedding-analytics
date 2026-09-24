"""Both stages end to end, through the handler the step function invokes."""

import json

import pytest

import app
from shared.tables.pipeline_entries import EntryStatus

from conftest import (
    BATCH_ID,
    BOOK_PAIRS,
    INDEX,
    INDEX_2,
    SUBJECT,
    s3_body,
    s3_content_type,
    status_of,
    succeeded_response,
)


# ── SEND ──────────────────────────────────────────────────────────────


def test_send_opens_one_batch_over_the_books_it_is_handed(
    scraped_book, send_client, entries
):
    index = scraped_book()

    status = app.handler({"book_ids": [str(index)]}, None)

    assert status == {
        "batch_id": BATCH_ID,
        "book_count": 1,
        "batch_status": "in_progress",
    }
    assert status_of(entries, index) == EntryStatus.STANDARDIZE_SUBMITTED
    assert send_client.messages.batches.create.call_count == 1


def test_send_writes_the_book_manifest_it_submitted(scraped_book, send_client, bucket):
    index = scraped_book()

    app.handler({"book_ids": [str(index)]}, None)

    manifest = json.loads(s3_body(bucket, f"standardize-html/books/{index}.json"))
    assert manifest["index"] == str(index)
    assert [tuple(pair) for pair in manifest["tag_text_pairs"]] == BOOK_PAIRS


def test_send_writes_one_batch_manifest_per_batch(scraped_book, send_client, bucket):
    """Keyed by batch id, so a second batch cannot land on the first one's manifest."""
    index = scraped_book()

    app.handler({"book_ids": [str(index)]}, None)

    key = f"standardize-html/batch-details/{BATCH_ID}.json"
    manifest = json.loads(s3_body(bucket, key))
    assert manifest == {
        "llm_batch_id": BATCH_ID,
        "book_ids": [str(index)],
    }
    assert s3_content_type(bucket, key) == "application/json; charset=utf-8"


def test_send_passes_over_a_book_that_is_not_at_scraped_html(
    scraped_book, seed, send_client, entries
):
    """The caller hands over a whole subject; only the books at SCRAPED_HTML go."""
    index = scraped_book()
    seed(EntryStatus.SCRAPED_METADATA, INDEX_2)

    status = app.handler({"book_ids": [str(index), str(INDEX_2)]}, None)

    assert status["book_count"] == 1
    assert status_of(entries, INDEX_2) == EntryStatus.SCRAPED_METADATA


# ── RETRIEVE ──────────────────────────────────────────────────────────


def test_retrieve_renders_the_artifacts_and_advances_the_status(
    submitted_batch, collect_client, entries, bucket
):
    submitted_batch()
    collect_client(responses=[succeeded_response(str(INDEX))])

    status = app.handler({"batch_id": BATCH_ID}, None)

    assert status == {
        "batch_id": BATCH_ID,
        "batch_status": "ended",
        "standardized": 1,
        "book_ids": [str(INDEX)],
        "failed": [],
    }
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZED
    assert s3_content_type(bucket, f"html-standardized/{INDEX}.html") == "text/html; charset=utf-8"


def test_retrieve_rewrites_headings_to_the_levels_the_llm_assigned(
    submitted_batch, collect_client, bucket
):
    submitted_batch()
    collect_client(responses=[succeeded_response(str(INDEX))])

    app.handler({"batch_id": BATCH_ID}, None)

    html = s3_body(bucket, f"html-standardized/{INDEX}.html")
    assert '<h3 data-block="drop">The Wealth of Nations</h3>' in html
    assert '<h2 data-block="chapter">BOOK I.</h2>' in html
    assert '<h3 data-block="section">OF THE CAUSES OF IMPROVEMENT.</h3>' in html


def test_retrieve_keeps_blocks_blank_line_separated_in_the_text_artifact(
    submitted_batch, collect_client, bucket
):
    """tokenize segments sentences within each block, so the blank lines are load-bearing."""
    submitted_batch()
    collect_client(responses=[succeeded_response(str(INDEX))])

    app.handler({"batch_id": BATCH_ID}, None)

    text = s3_body(bucket, f"text/{INDEX}.txt")
    assert text == "\n\n".join(text for _, text in BOOK_PAIRS[2:]) + "\n"


def test_paratext_is_left_out_of_the_text_artifact(
    submitted_batch, collect_client, bucket
):
    """An index is worse for training than plain noise -- it is the book's own
    vocabulary in alphabetical order, so a skip-gram window over it invents
    co-occurrences between exactly the terms the corpus is queried on."""
    submitted_batch()
    collect_client(
        responses=[
            succeeded_response(str(INDEX), text="0|section\n1|chapter\n2|drop\n")
        ]
    )

    app.handler({"batch_id": BATCH_ID}, None)

    text = s3_body(bucket, f"text/{INDEX}.txt")
    assert "OF THE CAUSES OF IMPROVEMENT." not in text
    assert "The greatest improvement" not in text, "prose under it goes too"
    assert "An inquiry into the nature and causes." in text, "the body stays"


def test_paratext_is_still_in_the_html_artifact(
    submitted_batch, collect_client, bucket
):
    """Only `text/` feeds the trainer; the html is the readable whole book."""
    submitted_batch()
    collect_client(
        responses=[
            succeeded_response(str(INDEX), text="0|section\n1|chapter\n2|drop\n")
        ]
    )

    app.handler({"batch_id": BATCH_ID}, None)

    assert "OF THE CAUSES OF IMPROVEMENT." in s3_body(
        bucket, f"html-standardized/{INDEX}.html"
    )


def test_retrieve_saves_the_raw_batch_response(
    submitted_batch, collect_client, bucket
):
    """Kept for the record: the reply is what any later dispute about a rendering is settled against."""
    submitted_batch()
    collect_client(responses=[succeeded_response(str(INDEX))])

    app.handler({"batch_id": BATCH_ID}, None)

    key = f"standardize-html/batch-results/{BATCH_ID}/{INDEX}.json"
    assert json.loads(s3_body(bucket, key))["custom_id"] == str(INDEX)


def test_retrieve_is_safe_to_re_run_over_a_settled_batch(
    submitted_batch, collect_client, entries, bucket
):
    """A second call renders the same artifacts from the same manifest; the repeated
    status write is the no-op, because the guard only lets a status move forward."""
    submitted_batch()
    collect_client(responses=[succeeded_response(str(INDEX))])
    app.handler({"batch_id": BATCH_ID}, None)
    first = s3_body(bucket, f"text/{INDEX}.txt")

    collect_client(responses=[succeeded_response(str(INDEX))])
    status = app.handler({"batch_id": BATCH_ID}, None)

    assert status["standardized"] == 1
    assert s3_body(bucket, f"text/{INDEX}.txt") == first
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZED


# ── SEND by subject ───────────────────────────────────────────────────
#
# `subject_id` is not a stage of its own: the handler expands the subject to the books
# at SCRAPED_HTML and submits them in the same call, so everything SEND does over a
# handed-over list it also does here. What these pin is the expansion -- which books it
# picks, and where it differs from being handed the same ids outright.


def test_send_by_subject_opens_one_batch_over_the_subject_s_pending_books(
    scraped_book, send_client, entries
):
    index = scraped_book()

    status = app.handler({"subject_id": str(SUBJECT)}, None)

    assert status == {
        "batch_id": BATCH_ID,
        "book_count": 1,
        "batch_status": "in_progress",
    }
    assert status_of(entries, index) == EntryStatus.STANDARDIZE_SUBMITTED
    assert send_client.messages.batches.create.call_count == 1


def test_send_by_subject_passes_over_a_book_that_is_not_at_scraped_html(
    scraped_book, seed, send_client, entries
):
    scraped_book()
    seed(EntryStatus.SCRAPED_METADATA, INDEX_2)

    assert app.handler({"subject_id": str(SUBJECT)}, None)["book_count"] == 1
    assert status_of(entries, INDEX_2) == EntryStatus.SCRAPED_METADATA


def test_send_by_subject_submits_the_rest_around_a_book_already_in_flight(
    scraped_book, seed, send_client, entries
):
    """The one place a subject differs from the same ids handed over outright.

    An explicit book_ids list naming a book at STANDARDIZE_SUBMITTED refuses the whole
    call with BooksInFlightError -- get_entries sees it and raises. A subject never
    hands it to get_entries at all: the SCRAPED_HTML filter drops it during the
    expansion, and the rest of the subject submits. Neither opens a second batch over a
    book in flight, which is the property that matters; they differ in whether the
    caller is told.
    """
    index = scraped_book()
    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX_2)

    status = app.handler({"subject_id": str(SUBJECT)}, None)

    assert status["book_count"] == 1
    assert status_of(entries, index) == EntryStatus.STANDARDIZE_SUBMITTED
    assert status_of(entries, INDEX_2) == EntryStatus.STANDARDIZE_SUBMITTED
    assert send_client.messages.batches.create.call_count == 1


def test_send_by_subject_ignores_another_subject_s_books(
    scraped_book, seed, send_client, entries
):
    from shared.commons import BookIndex

    scraped_book()
    seed(EntryStatus.SCRAPED_HTML, INDEX_2, subject_ids={BookIndex(999)})

    assert app.handler({"subject_id": str(SUBJECT)}, None)["book_count"] == 1
    assert status_of(entries, INDEX_2) == EntryStatus.SCRAPED_HTML


def test_send_by_subject_caps_the_batch_and_leaves_the_overflow_for_the_next_run(
    scraped_book, send_client, entries, monkeypatch
):
    """The one path whose size the caller does not set, so the cap is what makes a
    batch's cost known before it is opened. The overflow keeps SCRAPED_HTML and comes
    back next run, which makes re-invoking the drain. Ids are sorted, so which books
    make the cut is the same answer twice rather than whatever the Scan returned first.
    """
    from shared.commons import BookIndex

    monkeypatch.setattr("llm_request.make_request.MAX_BOOKS_PER_SUBJECT", 3)
    indexes = [BookIndex(source_id) for source_id in range(1, 6)]
    for index in indexes:
        scraped_book(index)

    status = app.handler({"subject_id": str(SUBJECT)}, None)

    assert status["book_count"] == 3
    submitted = [
        index
        for index in indexes
        if status_of(entries, index) == EntryStatus.STANDARDIZE_SUBMITTED
    ]
    assert submitted == sorted(indexes)[:3]


def test_send_by_subject_with_nothing_pending_opens_no_batch(seed, send_client):
    """An ordinary outcome, not a raise. submit reports an empty list the way it
    reports a book_ids handover that had no work -- batch_id null, batch_status ended --
    so the machine needs no gate ahead of standardize-submit for this case.
    """
    seed(EntryStatus.STANDARDIZED)

    status = app.handler({"subject_id": str(SUBJECT)}, None)

    assert status == {
        "batch_id": None,
        "book_count": 0,
        "batch_status": "ended",
    }
    assert send_client.messages.batches.create.call_count == 0


# ── Dispatch ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"book_ids": []},
        {"batch_id": None},
        {"subject_id": None},
        {"batch_id": BATCH_ID, "book_ids": [str(INDEX)]},
        {"book_ids": [str(INDEX)], "subject_id": "12345"},
        {"batch_id": BATCH_ID, "subject_id": "12345"},
    ],
    ids=[
        "none",
        "empty book_ids",
        "null batch_id",
        "null subject_id",
        "batch and books",
        "books and subject",
        "batch and subject",
    ],
)
def test_handler_requires_exactly_one_of_the_three_fields(event):
    """The field that carries the work picks the stage, so none and several are the same
    mistake: an empty book_ids is a caller with nothing to hand over, not a collect."""
    with pytest.raises(ValueError, match="Exactly one"):
        app.handler(event, None)


def test_handler_reads_the_field_out_of_a_json_body(scraped_book, send_client):
    """Invoked through a payload rather than a bare event."""
    index = scraped_book()

    status = app.handler({"body": json.dumps({"book_ids": [str(index)]})}, None)

    assert status["book_count"] == 1
