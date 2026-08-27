"""Reading a batch's replies back onto the books.

standardize_tag_text_pairs is where a mistake would be silent rather than loud: a reply
misaligned by one position relabels real headings with the wrong levels and still writes
a plausible-looking book. Every way a reply can fail to line up raises instead.
"""

import logging

import pytest

from shared.tables.pipeline_entries import EntryStatus
from llm_parse_response.save_artifacts import render_html, render_text
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
        ("h1", "The Wealth of Nations", "title"),
        ("p", "An inquiry into the nature and causes.", "title"),
        ("h2", "BOOK I.", "chapter"),
        ("h3", "OF THE CAUSES OF IMPROVEMENT.", "subsection"),
        (
            "p",
            "The greatest improvement in the productive powers of labour.",
            "subsection",
        ),
    ]


def test_prose_belongs_to_the_heading_above_it():
    """What makes a whole index droppable rather than just its heading."""
    pairs = [
        ("h1", "A chapter"),
        ("p", "Its prose."),
        ("h1", "INDEX."),
        ("p", "Accumulation of capital, 16."),
    ]

    assert [entry.block for entry in standardize_tag_text_pairs(
        "0|chapter\n1|index", pairs
    )] == ["chapter", "chapter", "index", "index"]


def test_prose_before_the_first_heading_has_no_block():
    pairs = [("p", "Transcriber's note."), ("h1", "Title")]

    assert standardize_tag_text_pairs("0|title", pairs)[0].block is None


def test_the_source_heading_level_is_not_consulted():
    """Levels in the scraped html were assigned by OCR font size, not document logic."""
    pairs = [("h4", "The Wealth of Nations"), ("p", "Prose.")]

    assert standardize_tag_text_pairs("0|title", pairs) == [
        ("h1", "The Wealth of Nations", "title"),
        ("p", "Prose.", "title"),
    ]


def test_a_blank_line_inside_the_reply_is_ignored():
    pairs = [("h1", "Title"), ("h2", "A chapter")]

    assert standardize_tag_text_pairs("0|title\n\n1|chapter\n", pairs) == [
        ("h1", "Title", "title"),
        ("h2", "A chapter", "chapter"),
    ]


# ── The title page ────────────────────────────────────────────────────


def test_a_title_page_set_line_by_line_becomes_one_heading():
    """Six headings spelling out one title, which is how a title page is typeset."""
    pairs = [(f"h{n % 3 + 1}", word) for n, word in enumerate(
        ["ON", "THE PRINCIPLES", "OF", "POLITICAL ECONOMY,", "AND", "TAXATION."]
    )]
    reply = "\n".join(f"{n}|title" for n in range(6))

    assert standardize_tag_text_pairs(reply, pairs) == [
        ("h1", "ON THE PRINCIPLES OF POLITICAL ECONOMY, AND TAXATION.", "title")
    ]


def test_only_adjacent_title_headings_merge():
    """Prose between two title lines means they are not one heading."""
    pairs = [("h1", "ON"), ("p", "A note."), ("h1", "TAXATION.")]

    assert standardize_tag_text_pairs("0|title\n1|title", pairs) == [
        ("h1", "ON", "title"),
        ("p", "A note.", "title"),
        ("h1", "TAXATION.", "title"),
    ]


def test_headings_of_other_blocks_are_never_merged():
    """Two chapter headings in a row stay two chapters."""
    pairs = [("h1", "CHAPTER I."), ("h1", "CHAPTER II.")]

    assert standardize_tag_text_pairs("0|chapter\n1|chapter", pairs) == [
        ("h2", "CHAPTER I.", "chapter"),
        ("h2", "CHAPTER II.", "chapter"),
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


# ── The rendered page ─────────────────────────────────────────────────


def test_the_page_declares_the_language_the_corpus_is_in():
    """Fixed, not read: scrape sends every non-English book to a terminal status, so
    nothing else reaches this stage."""
    page = render_html(standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS), INDEX)

    assert '<html lang="en">' in page


def test_the_page_is_titled_with_the_book_not_its_index():
    page = render_html(standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS), INDEX)

    assert "<title>The Wealth of Nations</title>" in page


def test_the_library_record_titles_the_page_over_any_heading():
    """A title page is not always transcribed as a heading -- gutenberg-30107 has none,
    and would otherwise be titled by its index."""
    blocks = standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS)

    page = render_html(blocks, INDEX, "Principles of Political Economy")

    assert "<title>Principles of Political Economy</title>" in page


def test_a_book_with_no_title_heading_falls_back_to_its_index():
    blocks = standardize_tag_text_pairs("0|chapter", [("h1", "CHAPTER I.")])

    assert f"<title>{INDEX}</title>" in render_html(blocks, INDEX)


def test_prose_carries_the_block_of_the_heading_above_it():
    """What a passage index filters on -- the tag cannot say it, since four semantic
    blocks all render as h3."""
    page = render_html(standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS), INDEX)

    assert '<p data-block="subsection">The greatest improvement' in page


def test_a_table_of_contents_is_kept_in_the_page_but_not_the_text():
    """Nothing is deleted from the artifact a reader or an index reads; only the
    trainer's input drops it."""
    pairs = [("h1", "CONTENTS."), ("p", "I. On Value 1"), ("h1", "CHAPTER I."), ("p", "Real prose.")]
    blocks = standardize_tag_text_pairs("0|contents\n1|chapter", pairs)

    assert '<h3 data-block="contents">CONTENTS.</h3>' in render_html(blocks, INDEX)
    assert "I. On Value 1" not in render_text(blocks)
    assert "Real prose." in render_text(blocks)


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
        "failed": [],
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


def test_one_unreadable_reply_does_not_strand_the_rest_of_the_batch(
    submitted_batch, collect_client, entries
):
    """The reply for INDEX skips a heading position, so it cannot be applied. Letting
    that escape would end the settle and leave INDEX_2 uncollected -- permanently, since
    a re-run streams the same results and fails at the same item."""
    submitted_batch(indexes=(INDEX, INDEX_2))
    collect_client(
        responses=[
            succeeded_response(str(INDEX), text="0|title"),  # 3 headings, 1 line
            succeeded_response(str(INDEX_2)),
        ]
    )

    status = standardize_from_batch(BATCH_ID)

    assert status["standardized"] == 1
    assert status["failed"] == [str(INDEX)]
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZE_SUBMITTED
    assert status_of(entries, INDEX_2) == EntryStatus.STANDARDIZED


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
