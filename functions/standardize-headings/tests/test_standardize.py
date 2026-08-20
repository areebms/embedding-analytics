"""Tests for the collect stage: reading a batch's replies back onto the books.

standardize_tag_text_pairs is where a mistake would be silent rather than loud — a
misaligned position would relabel real headings with the wrong levels and still write
a plausible-looking book, so every way the reply can fail to line up raises here.
"""

import pytest

from conftest import BATCH_ID, BOOK_PAIRS, INDEX, INDEX_2, s3_body, succeeded_response
from shared.tables.pipeline_entries import EntryStatus, standardized_html_key, text_key

from book_records.batch_index import save_batch_index
from book_records.schemas import BookTagTextPairs
from llm_classify_request.constants import SYSTEM_PROMPT
from llm_parse_response.standardize import (
    SEMANTIC_BLOCK_TO_LEVEL,
    get_llm_content_text,
    standardize_from_batch,
    standardize_tag_text_pairs,
)


# The reply for BOOK_PAIRS: title, then the two-part chapter heading.
BOOK_REPLY = "0|title\n1|chapter\n2|subsection"

BOOK_STANDARDIZED = [
    ("h1", "The Wealth of Nations"),
    ("p", "An inquiry into the nature and causes."),
    ("h2", "BOOK I."),
    ("h3", "OF THE CAUSES OF IMPROVEMENT."),
    ("p", "The greatest improvement in the productive powers of labour."),
]


def book(index=INDEX, tag_text_pairs=None):
    return BookTagTextPairs(
        llm_index=str(index),
        index=index,
        tag_text_pairs=BOOK_PAIRS if tag_text_pairs is None else tag_text_pairs,
    )


# ── the prompt and the level map are one contract ─────────────────────


def prompt_semantic_blocks():
    for line in SYSTEM_PROMPT.splitlines():
        if line.startswith("Valid semantic blocks:"):
            _, _, listed = line.partition(":")
            return {block.strip() for block in listed.split(",")}
    raise AssertionError("SYSTEM_PROMPT no longer declares its valid semantic blocks")


def test_every_block_the_prompt_allows_has_a_heading_level():
    """A block added to the prompt but not to the map is rejected at collect time,
    after the whole corpus has already been classified and paid for."""
    assert prompt_semantic_blocks() <= set(SEMANTIC_BLOCK_TO_LEVEL)


def test_every_block_the_map_knows_is_one_the_prompt_asks_for():
    """The other direction: a level with no prompt line is a block the model will
    never return, and reads as supported when it is not."""
    assert set(SEMANTIC_BLOCK_TO_LEVEL) <= prompt_semantic_blocks()


def test_every_level_is_one_of_the_three_the_artifact_uses():
    assert set(SEMANTIC_BLOCK_TO_LEVEL.values()) <= {"h1", "h2", "h3"}


# ── get_llm_content_text ──────────────────────────────────────────────


def test_the_text_blocks_are_joined_in_order():
    content = [{"type": "text", "text": "0|title"}, {"type": "text", "text": "1|chapter"}]

    assert get_llm_content_text(content) == "0|title\n1|chapter"


def test_thinking_and_other_blocks_are_left_out_of_the_reply(caplog):
    """Thinking is disabled in the request, so a thinking block arriving here means
    the request changed — the reply is still readable, but it belongs in the log."""
    content = [
        {"type": "thinking", "thinking": "weighing it up"},
        {"type": "text", "text": "0|title"},
    ]

    with caplog.at_level("WARNING"):
        assert get_llm_content_text(content) == "0|title"

    assert "thinking" in caplog.text


def test_an_empty_text_block_is_skipped():
    assert get_llm_content_text([{"type": "text", "text": ""}]) == ""


def test_no_content_at_all_reads_as_an_empty_reply():
    assert get_llm_content_text([]) == ""


# ── standardize_tag_text_pairs ────────────────────────────────────────


def test_each_heading_takes_the_level_its_semantic_block_maps_to():
    assert standardize_tag_text_pairs(BOOK_REPLY, BOOK_PAIRS) == BOOK_STANDARDIZED


def test_paragraphs_pass_through_untouched():
    pairs = [("p", "prose"), ("h1", "Title"), ("p", "more prose")]

    assert standardize_tag_text_pairs("0|chapter", pairs) == [
        ("p", "prose"),
        ("h2", "Title"),
        ("p", "more prose"),
    ]


@pytest.mark.parametrize(
    "block, level",
    sorted(SEMANTIC_BLOCK_TO_LEVEL.items()),
)
def test_every_semantic_block_the_model_may_return_is_mapped(block, level):
    assert standardize_tag_text_pairs(f"0|{block}", [("h4", "x")]) == [(level, "x")]


def test_the_block_name_is_read_case_and_whitespace_insensitively():
    assert standardize_tag_text_pairs("  0 | Chapter  ", [("h4", "x")]) == [("h2", "x")]


def test_blank_lines_in_the_reply_are_ignored():
    assert standardize_tag_text_pairs("\n0|title\n\n1|chapter\n\n", BOOK_PAIRS[:3]) == [
        ("h1", "The Wealth of Nations"),
        ("p", "An inquiry into the nature and causes."),
        ("h2", "BOOK I."),
    ]


def test_a_reply_naming_more_positions_than_there_are_headings_is_tolerated():
    """An extra line costs nothing; a missing one is what would misalign the book."""
    assert standardize_tag_text_pairs("0|title\n1|chapter\n2|section", [("h1", "x")]) == [
        ("h1", "x")
    ]


def test_a_line_with_no_separator_is_refused():
    with pytest.raises(ValueError, match="no separator in line: 0 title"):
        standardize_tag_text_pairs("0 title", BOOK_PAIRS)


def test_a_semantic_block_outside_the_agreed_set_is_refused():
    """The model inventing a block is exactly the drift the prompt and the level map
    sit side by side to prevent."""
    with pytest.raises(ValueError, match="unknown semantic block 'preamble'"):
        standardize_tag_text_pairs("0|preamble", BOOK_PAIRS)


def test_a_position_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError):
        standardize_tag_text_pairs("first|title", BOOK_PAIRS)


def test_a_heading_the_model_never_classified_is_refused():
    """Falling back to the original tag would put an OCR font-size guess into the
    artifact and call it standardized."""
    with pytest.raises(ValueError, match="no semantic block for heading position 1"):
        standardize_tag_text_pairs("0|title", BOOK_PAIRS)


def test_an_empty_reply_for_a_book_with_headings_is_refused():
    with pytest.raises(ValueError, match="no semantic block for heading position 0"):
        standardize_tag_text_pairs("", BOOK_PAIRS)


def test_a_book_with_no_headings_needs_no_reply():
    pairs = [("p", "prose"), ("p", "more prose")]

    assert standardize_tag_text_pairs("", pairs) == pairs


# ── standardize_from_batch ────────────────────────────────────────────


def collect(mocker, client):
    import llm_parse_response.standardize as standardize

    mocker.patch.object(standardize, "get_client", return_value=client)
    return standardize_from_batch(BATCH_ID)


def test_a_batch_still_running_is_left_alone(mocker, batch_client, bucket, entries):
    """This stage never waits on a batch, which is the whole reason it is a separate
    invocation."""
    client = batch_client("in_progress")

    assert collect(mocker, client) == {
        "batch_id": BATCH_ID,
        "batch_status": "in_progress",
        "standardized": 0,
    }
    client.messages.batches.results.assert_not_called()


def test_a_finished_batch_writes_both_artifacts_and_advances_the_book(
    mocker, batch_client, bucket, book_manifest, seed, statuses
):
    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX)
    book_manifest(INDEX)
    save_batch_index(BATCH_ID, [book(INDEX)])
    client = batch_client("ended", [succeeded_response("gutenberg-3300", BOOK_REPLY)])

    result = collect(mocker, client)

    assert result == {"batch_id": BATCH_ID, "batch_status": "ended", "standardized": 1}
    assert s3_body(bucket, text_key(INDEX)).startswith("The Wealth of Nations\n\n")
    assert "<h3>OF THE CAUSES OF IMPROVEMENT.</h3>" in s3_body(
        bucket, standardized_html_key(INDEX)
    )
    assert statuses(INDEX) == EntryStatus.STANDARDIZED


def test_a_whole_batch_of_books_is_settled_one_at_a_time(
    mocker, batch_client, bucket, book_manifest, seed, statuses
):
    """A corpus of flattened text does not fit in memory at once, so each book is
    loaded, rendered and released before the next."""
    for index in (INDEX, INDEX_2):
        seed(EntryStatus.STANDARDIZE_SUBMITTED, index)
        book_manifest(index)
    save_batch_index(BATCH_ID, [book(INDEX), book(INDEX_2)])
    client = batch_client(
        "ended",
        [
            succeeded_response("gutenberg-3300", BOOK_REPLY),
            succeeded_response("gutenberg-11", BOOK_REPLY),
        ],
    )

    assert collect(mocker, client)["standardized"] == 2
    assert statuses(INDEX) == EntryStatus.STANDARDIZED
    assert statuses(INDEX_2) == EntryStatus.STANDARDIZED


def test_a_reply_for_a_book_not_in_the_manifest_is_passed_over(
    mocker, batch_client, bucket, book_manifest, seed, statuses, caplog
):
    """The manifest is the only record of which llm_index means which book. A reply
    it does not name has no book to be written to."""
    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX)
    book_manifest(INDEX)
    save_batch_index(BATCH_ID, [book(INDEX)])
    client = batch_client(
        "ended",
        [
            succeeded_response("gutenberg-99999", BOOK_REPLY),
            succeeded_response("gutenberg-3300", BOOK_REPLY),
        ],
    )

    with caplog.at_level("WARNING"):
        assert collect(mocker, client)["standardized"] == 1

    assert "gutenberg-99999" in caplog.text
    assert statuses(INDEX) == EntryStatus.STANDARDIZED


def test_collecting_a_batch_the_manifest_does_not_describe_is_refused(
    mocker, batch_client, bucket
):
    save_batch_index("msgbatch_older", [book(INDEX)])
    client = batch_client("ended", [succeeded_response("gutenberg-3300", BOOK_REPLY)])

    with pytest.raises(ValueError, match="manifest is for batch msgbatch_older"):
        collect(mocker, client)


def test_a_book_whose_reply_does_not_line_up_stops_the_run(
    mocker, batch_client, bucket, book_manifest, seed, statuses
):
    """Better a failed collect that can be rerun from the archived results than a
    corpus half-written with mislabelled headings."""
    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX)
    book_manifest(INDEX)
    save_batch_index(BATCH_ID, [book(INDEX)])
    client = batch_client("ended", [succeeded_response("gutenberg-3300", "0|title")])

    with pytest.raises(ValueError, match="no semantic block for heading position 1"):
        collect(mocker, client)

    assert statuses(INDEX) == EntryStatus.STANDARDIZE_SUBMITTED


def test_an_ended_batch_with_no_results_reports_nothing_standardized(
    mocker, batch_client, bucket
):
    save_batch_index(BATCH_ID, [])

    assert collect(mocker, batch_client("ended", []))["standardized"] == 0
