"""Reading a batch's replies back onto the books.

standardize_tag_text_pairs is where a mistake would be silent rather than loud: a reply
misaligned by one position relabels real headings with the wrong levels and still writes
a plausible-looking book. Every way a reply can fail to line up raises instead.
"""

import logging

import pytest

from shared.tables.pipeline_entries import EntryStatus
from llm_parse_response.standardize import (
    get_llm_content_text,
    standardize_from_batch,
    standardize_tag_text_pairs,
)

from conftest import (
    BATCH_ID,
    BOOK_PAIRS,
    BOOK_REPLY,
    INDEX,
    INDEX_2,
    status_of,
    succeeded_response,
)


# ── The reply, applied to a book ──────────────────────────────────────


def test_headings_take_the_level_of_the_semantic_block_they_were_given():
    assert standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS) == [
        ("h1", "The Wealth of Nations"),
        ("p", "An inquiry into the nature and causes."),
        ("h2", "BOOK I."),
        ("h3", "OF THE CAUSES OF IMPROVEMENT."),
        ("p", "The greatest improvement in the productive powers of labour."),
    ]


def test_the_source_heading_level_is_not_consulted():
    """Levels in the scraped html were assigned by OCR font size, not document logic."""
    pairs = [("h4", "The Wealth of Nations"), ("p", "Prose.")]

    assert standardize_tag_text_pairs("0|title", pairs) == [
        ("h1", "The Wealth of Nations"),
        ("p", "Prose."),
    ]


def test_a_blank_line_inside_the_reply_is_ignored():
    pairs = [("h1", "Title"), ("h2", "A chapter")]

    assert standardize_tag_text_pairs("0|title\n\n1|chapter\n", pairs) == [
        ("h1", "Title"),
        ("h2", "A chapter"),
    ]


def test_a_line_with_no_separator_raises():
    with pytest.raises(ValueError, match="no separator"):
        standardize_tag_text_pairs("0 title", [("h1", "Title")])


def test_an_unknown_semantic_block_raises():
    with pytest.raises(ValueError, match="unknown semantic block 'preamble'"):
        standardize_tag_text_pairs("0|preamble", [("h1", "Title")])


def test_a_heading_the_reply_skipped_raises():
    """Better to lose the book from this batch than to write it mislabelled."""
    with pytest.raises(ValueError, match="position 1"):
        standardize_tag_text_pairs(
            "0|title", [("h1", "Title"), ("h2", "A chapter")]
        )


# ── The reply, read out of the response ───────────────────────────────


def test_content_text_joins_the_text_blocks():
    content = [{"type": "text", "text": "0|title"}, {"type": "text", "text": "1|chapter"}]

    assert get_llm_content_text(content) == "0|title\n1|chapter"


def test_content_text_skips_blocks_that_carry_no_reply(caplog):
    content = [
        {"type": "thinking", "thinking": "considering"},
        {"type": "text", "text": ""},
        {"type": "text", "text": "0|title"},
    ]

    with caplog.at_level(logging.WARNING):
        assert get_llm_content_text(content) == "0|title"

    assert "skipping thinking block" in caplog.text


# ── Settling a batch ──────────────────────────────────────────────────


def test_a_batch_still_running_is_left_alone(submitted_batch, collect_client, entries):
    submitted_batch()
    client = collect_client(processing_status="in_progress")

    status = standardize_from_batch(BATCH_ID)

    assert status == {
        "batch_id": BATCH_ID,
        "batch_status": "in_progress",
        "standardized": 0,
    }
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZE_SUBMITTED
    client.messages.batches.results.assert_not_called()


def test_results_are_keyed_by_custom_id_not_by_position(
    submitted_batch, collect_client, entries
):
    """The Batches API returns results in any order."""
    submitted_batch(indexes=(INDEX, INDEX_2))
    collect_client(
        responses=[succeeded_response(str(INDEX_2)), succeeded_response(str(INDEX))]
    )

    status = standardize_from_batch(BATCH_ID)

    assert status["standardized"] == 2
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZED
    assert status_of(entries, INDEX_2) == EntryStatus.STANDARDIZED


def test_a_result_for_a_book_this_batch_never_carried_is_passed_over(
    submitted_batch, collect_client, entries, caplog
):
    submitted_batch()
    collect_client(responses=[succeeded_response("gutenberg-9999")])

    with caplog.at_level(logging.WARNING):
        status = standardize_from_batch(BATCH_ID)

    assert status["standardized"] == 0
    assert "unknown llm_index gutenberg-9999" in caplog.text
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZE_SUBMITTED


def test_a_book_the_batch_answered_for_nobody_stays_in_flight(
    submitted_batch, collect_client, entries, caplog
):
    """It keeps STANDARDIZE_SUBMITTED, so re-running the subject will not resubmit it
    behind the operator's back -- the batch is settled by hand from the log line."""
    submitted_batch(indexes=(INDEX, INDEX_2))

    collect_client(responses=[succeeded_response(str(INDEX))])
    with caplog.at_level(logging.WARNING):
        status = standardize_from_batch(BATCH_ID)

    assert status["standardized"] == 1
    assert status_of(entries, INDEX_2) == EntryStatus.STANDARDIZE_SUBMITTED
    assert "1 book(s) had no result" in caplog.text
    assert str(INDEX_2) in caplog.text
