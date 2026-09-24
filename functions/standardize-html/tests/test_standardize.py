"""Reading a batch's replies back onto the books.

standardize_tag_text_pairs is where a mistake would be silent rather than loud: a reply
misaligned by one position relabels real headings with the wrong levels and still writes
a plausible-looking book.

The defence used to be to raise on any defect at all. That cost a book -- gutenberg-38194
came back with all 174 classifications right and ``` fences around them, and two fence
lines threw the other 174 away. Positions are explicit in the reply, so a dropped line
shifts nothing; only the headings it named are affected. Unreadable lines are skipped and
unclassified headings take the default rather than the block around them. A reply with
nothing readable in it at all is still a failure, because that is no classification
rather than a damaged one.
"""

import logging

import pytest

from shared.tables.pipeline_entries import EntryStatus
from llm_response.save_artifacts import render_html, render_text
from llm_response.standardize import (
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
        ("h3", "The Wealth of Nations", "drop"),
        ("p", "An inquiry into the nature and causes.", "drop"),
        ("h2", "BOOK I.", "chapter"),
        ("h3", "OF THE CAUSES OF IMPROVEMENT.", "section"),
        (
            "p",
            "The greatest improvement in the productive powers of labour.",
            "section",
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
        "0|chapter\n1|drop", pairs
    )] == ["chapter", "chapter", "drop", "drop"]


def test_prose_before_the_first_heading_has_no_block():
    pairs = [("p", "Transcriber's note."), ("h1", "Title")]

    assert standardize_tag_text_pairs("0|chapter", pairs)[0].block is None


def test_the_source_heading_level_is_not_consulted():
    """Levels in the scraped html were assigned by OCR font size, not document logic."""
    pairs = [("h4", "The Wealth of Nations"), ("p", "Prose.")]

    assert standardize_tag_text_pairs("0|chapter", pairs) == [
        ("h2", "The Wealth of Nations", "chapter"),
        ("p", "Prose.", "chapter"),
    ]


def test_a_blank_line_inside_the_reply_is_ignored():
    pairs = [("h1", "Title"), ("h2", "A chapter")]

    assert standardize_tag_text_pairs("0|section\n\n1|chapter\n", pairs) == [
        ("h3", "Title", "section"),
        ("h2", "A chapter", "chapter"),
    ]


def test_markdown_fences_do_not_cost_the_book():
    """gutenberg-38194, exactly: every classification correct, inside ``` fences."""
    reply = "```\n0|section\n1|chapter\n```"

    assert [entry.block for entry in standardize_tag_text_pairs(
        reply, [("h1", "Title"), ("h2", "A chapter")]
    )] == ["section", "chapter"]


def test_a_line_with_no_separator_is_skipped():
    blocks = standardize_tag_text_pairs(
        "Here are the classifications:\n0|section", [("h1", "Title")]
    )

    assert [entry.block for entry in blocks] == ["section"]


def test_an_unknown_semantic_block_is_skipped():
    """A block the vocabulary does not contain is dropped, not guessed at, and the
    heading it belonged to falls back like any other unnamed one."""
    blocks = standardize_tag_text_pairs(
        "0|chapter\n1|preamble", [("h1", "A chapter"), ("h2", "Something")]
    )

    assert [entry.block for entry in blocks] == ["chapter", "section"]


def test_an_unclassified_heading_never_inherits_a_drop():
    """Inheriting `drop` would delete an author's text on the strength of a missing
    line. The prompt's own rule is to keep when in doubt, so it keeps."""
    blocks = standardize_tag_text_pairs(
        "0|drop", [("h1", "INDEX."), ("h2", "Of the wages of labour")]
    )

    assert [entry.block for entry in blocks] == ["drop", "section"]


def test_the_reply_decides_every_heading_it_names():
    """Every label comes from the reply, paratext included -- there is no second
    opinion to overrule it."""
    blocks = standardize_tag_text_pairs(
        "0|drop\n1|chapter\n2|section",
        [
            ("h1", "Contents"),
            ("h2", "Of the division of labour"),
            ("h3", "Of the wages of labour"),
        ],
    )

    assert [entry.block for entry in blocks] == ["drop", "chapter", "section"]


def test_a_named_introduction_the_model_calls_a_chapter_stays_a_chapter():
    """gutenberg-66710 sets an INTRODUCTION. at chapter depth under each of its parts.
    Calling one `section` changes the heading level and nothing else, so a book that
    treats it as a chapter is taken at its word."""
    blocks = standardize_tag_text_pairs(
        "0|chapter", [("h4", "INTRODUCTION.")]
    )

    assert [entry.block for entry in blocks] == ["chapter"]


def test_an_unnamed_label_becomes_a_section_not_a_chapter():
    """A heading the reply does not name must not be promoted into the block above it.
    A heading sits *under* what precedes it, so inheriting turns every unnamed
    subsection into a chapter of its own -- 474 of them across the two collected
    batches when the label naming them was withdrawn. It becomes a kept h3 instead."""
    blocks = standardize_tag_text_pairs(
        "0|chapter", [("h2", "Of value"), ("h3", "Of price")]
    )

    assert [entry.block for entry in blocks] == ["chapter", "section"]


def test_prose_still_belongs_to_the_heading_above_it():
    """The counterweight: a paragraph does inherit, because it is inside the chapter
    rather than under it. Only headings take the default."""
    blocks = standardize_tag_text_pairs(
        "0|chapter", [("h2", "Of value"), ("p", "The value of a commodity")]
    )

    assert [entry.block for entry in blocks] == ["chapter", "chapter"]


def test_a_reply_with_nothing_readable_in_it_raises():
    """No classification at all is a failure, not a damaged one to be patched up."""
    with pytest.raises(ValueError, match="no semantic blocks"):
        standardize_tag_text_pairs("I could not do that.", [("h1", "Title")])


# ── The rendered page ─────────────────────────────────────────────────


def test_the_page_declares_the_language_the_corpus_is_in():
    """Fixed, not read: scrape sends every non-English book to a terminal status, so
    nothing else reaches this stage."""
    page = render_html(standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS), INDEX)

    assert '<html lang="en">' in page


def test_the_page_is_titled_from_the_library_record():
    """Every book has one: `scrape_book_metadata` writes the record before it advances
    the status, so no book reaches this stage without it."""
    blocks = standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS)

    page = render_html(blocks, INDEX, "Principles of Political Economy")

    assert "<title>Principles of Political Economy</title>" in page


def test_a_page_rendered_without_a_record_falls_back_to_the_index():
    blocks = standardize_tag_text_pairs("0|chapter", [("h1", "CHAPTER I.")])

    assert f"<title>{INDEX}</title>" in render_html(blocks, INDEX)


def test_prose_carries_the_block_of_the_heading_above_it():
    """What a passage index filters on -- the tag cannot say it, since `section` and
    `drop` both render as h3."""
    page = render_html(standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS), INDEX)

    assert '<p data-block="section">The greatest improvement' in page


def test_a_table_of_contents_is_kept_in_the_page_but_not_the_text():
    """Nothing is deleted from the artifact a reader or an index reads; only the
    trainer's input drops it."""
    pairs = [("h1", "CONTENTS."), ("p", "I. On Value 1"), ("h1", "CHAPTER I."), ("p", "Real prose.")]
    blocks = standardize_tag_text_pairs("0|drop\n1|chapter", pairs)

    assert '<h3 data-block="drop">CONTENTS.</h3>' in render_html(blocks, INDEX)
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
        "book_ids": [],
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
    assert "unknown custom_id gutenberg-9999" in caplog.text
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZE_UNRESOLVED


def test_one_unreadable_reply_does_not_strand_the_rest_of_the_batch(
    submitted_batch, collect_client, entries
):
    """The reply for INDEX carries no classification at all, so it cannot be applied.
    Letting that escape would end the settle and leave INDEX_2 uncollected --
    permanently, since a re-run streams the same results and fails at the same item."""
    submitted_batch(indexes=(INDEX, INDEX_2))
    collect_client(
        responses=[
            succeeded_response(str(INDEX), text="I could not do that."),
            succeeded_response(str(INDEX_2)),
        ]
    )

    status = standardize_from_batch(BATCH_ID)

    assert status["standardized"] == 1
    assert status["failed"] == [str(INDEX)]
    assert status_of(entries, INDEX) == EntryStatus.STANDARDIZE_SUBMITTED
    assert status_of(entries, INDEX_2) == EntryStatus.STANDARDIZED


def test_book_ids_carries_only_the_books_that_reached_standardized(
    submitted_batch, collect_client, entries
):
    """`book_ids` is the handover to tokenize, and it is not the batch's book list.

    A book that could not be rendered stays at STANDARDIZE_SUBMITTED, and naming it here
    would hand tokenize a book with no `text/` object to read. The event carries what
    actually moved."""
    submitted_batch(indexes=(INDEX, INDEX_2))
    collect_client(
        responses=[
            succeeded_response(str(INDEX), text="I could not do that."),
            succeeded_response(str(INDEX_2)),
        ]
    )

    status = standardize_from_batch(BATCH_ID)

    assert status["book_ids"] == [str(INDEX_2)]
    assert status["standardized"] == len(status["book_ids"])


def test_a_book_the_batch_answered_for_nobody_is_taken_out_of_flight(
    submitted_batch, collect_client, entries, caplog
):
    """Leaving it at STANDARDIZE_SUBMITTED stranded it: a later collect streams the same
    stored result and skips it again, SEND refuses a call naming a book in flight, and
    the status guard will not take it back to SCRAPED_HTML. STANDARDIZE_UNRESOLVED is a
    forward move, so the guard allows it, and it is what makes the book reachable again.
    Still nothing resubmits it on its own -- that is the operator's `book_ids` call."""
    submitted_batch(indexes=(INDEX, INDEX_2))

    collect_client(responses=[succeeded_response(str(INDEX))])
    with caplog.at_level(logging.WARNING):
        status = standardize_from_batch(BATCH_ID)

    assert status["standardized"] == 1
    assert status_of(entries, INDEX_2) == EntryStatus.STANDARDIZE_UNRESOLVED
    assert "1 book(s) had no result" in caplog.text
    assert str(INDEX_2) in caplog.text
