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


# ── Dispatch ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"book_ids": []},
        {"batch_id": None},
        {"batch_id": BATCH_ID, "book_ids": [str(INDEX)]},
    ],
    ids=["neither", "empty book_ids", "null batch_id", "both"],
)
def test_handler_requires_exactly_one_of_the_two_fields(event):
    """The field that carries the work picks the stage, so neither and both are the same
    mistake: an empty book_ids is a caller with nothing to hand over, not a collect."""
    with pytest.raises(ValueError, match="Exactly one"):
        app.handler(event, None)


def test_handler_reads_the_field_out_of_a_json_body(scraped_book, send_client):
    """Invoked through a payload rather than a bare event."""
    index = scraped_book()

    status = app.handler({"body": json.dumps({"book_ids": [str(index)]})}, None)

    assert status["book_count"] == 1
