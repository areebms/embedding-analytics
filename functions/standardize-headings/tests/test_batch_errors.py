"""The refusals and the failure branches: what the stage does when it should not proceed.

A batch is paid for the moment it opens, and the books it carries are locked to it by
their status until it settles. Most of what is asserted here is a refusal to open one.
"""

import logging
from types import SimpleNamespace

import pytest

import app
from anthropic.types import RedactedThinkingBlock, TextBlock, ThinkingBlock, ToolUseBlock
from pydantic import ValidationError
from shared.tables.pipeline_entries import EntryStatus

from book_records.batch_index import load_batch_index, save_batch_index
from book_records.schemas import BookTagTextPairs
from llm_classify_request.make_request import BooksInFlightError, get_entries
from llm_classify_request.send_anthropic_request import (
    HeadingSemanticBlockError,
    convert_to_anthropic_request,
    get_client,
    to_anthropic_message,
)
from llm_parse_response.fetch import serialize_content_block, yield_anthropic_content

from conftest import (
    BATCH_ID,
    BOOK_PAIRS,
    INDEX,
    INDEX_2,
    PROSE_ONLY_HTML,
    errored_response,
    status_of,
    succeeded_response,
)


# ── Refusing to open a batch ──────────────────────────────────────────


def test_a_book_already_in_flight_refuses_the_whole_call(scraped_book, seed):
    """Not just the books in flight: a second batch over a book that already belongs to
    one is paid for twice, and both would settle against the same row. The caller
    re-runs the subject once the open batch has been collected."""
    scraped_book()
    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX_2)

    with pytest.raises(BooksInFlightError, match="1 of 2 book"):
        get_entries([INDEX, INDEX_2])


def test_a_book_with_no_headings_is_skipped_rather_than_submitted(
    scraped_book, send_client, entries
):
    """Nothing to classify, so the book leaves the pipeline at a terminal status instead
    of costing a request."""
    index = scraped_book(html=PROSE_ONLY_HTML)

    status = app.handler({"book_ids": [str(index)]}, None)

    assert status == {"batch_id": None, "book_count": 0, "batch_status": "ended"}
    assert status_of(entries, index) == EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS
    send_client.messages.batches.create.assert_not_called()


def test_a_missing_api_key_is_refused_before_a_client_is_built(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(HeadingSemanticBlockError, match="ANTHROPIC_API_KEY"):
        get_client()


def test_a_client_is_built_when_the_key_is_set():
    assert get_client().api_key == "test-key"


# ── The request that gets sent ────────────────────────────────────────


def book(index=INDEX, tag_text_pairs=None, llm_index=None):
    return BookTagTextPairs(
        llm_index=str(index) if llm_index is None else llm_index,
        index=index,
        tag_text_pairs=BOOK_PAIRS if tag_text_pairs is None else tag_text_pairs,
    )


def test_one_line_per_heading_carrying_the_prose_gap_after_it():
    """The gap is what tells a table-of-contents run apart from real chapters."""
    message = to_anthropic_message(INDEX, BOOK_PAIRS)

    assert message.content == (
        f"Book: {INDEX}\n\n"
        "0|h1|The Wealth of Nations|7\n"
        "1|h2|BOOK I.|0\n"
        "2|h2|OF THE CAUSES OF IMPROVEMENT.|9"
    )


def test_heading_text_is_truncated_and_stripped_of_the_separator():
    pairs = [("h1", "A|title\nsplit over lines"), ("h2", "x" * 200)]

    lines = to_anthropic_message(INDEX, pairs).content.splitlines()

    assert lines[2] == "0|h1|A/title split over lines|0"
    assert lines[3] == f"1|h2|{'x' * 100}|0"


def test_a_short_book_still_gets_room_to_answer():
    request = convert_to_anthropic_request(book(tag_text_pairs=[("p", "Just prose.")]))

    assert request.params.max_tokens == 256


def test_a_very_long_book_is_capped():
    pairs = [("h2", f"Chapter {n}") for n in range(1400)]

    request = convert_to_anthropic_request(book(tag_text_pairs=pairs))

    assert request.params.max_tokens == 16000


def test_a_custom_id_anthropic_would_reject_raises_here():
    with pytest.raises(ValidationError):
        convert_to_anthropic_request(book(llm_index="not a valid id"))


# ── Reading results back ──────────────────────────────────────────────


def drain(client):
    return list(yield_anthropic_content(client, BATCH_ID))


def client_returning(*responses):
    from unittest.mock import MagicMock

    client = MagicMock()
    client.messages.batches.results.return_value = iter(responses)
    return client


def test_an_errored_result_raises_with_what_anthropic_said(aws):
    client = client_returning(errored_response(str(INDEX), "request too large"))

    with pytest.raises(Exception, match="invalid_request_error: request too large"):
        drain(client)


def test_a_result_that_neither_succeeded_nor_errored_raises(aws):
    from anthropic.types.messages import (
        MessageBatchCanceledResult,
        MessageBatchIndividualResponse,
    )

    response = MessageBatchIndividualResponse(
        custom_id=str(INDEX),
        result=MessageBatchCanceledResult(type="canceled"),
    )

    with pytest.raises(Exception, match="batch result canceled"):
        drain(client_returning(response))


def test_a_truncated_reply_raises_rather_than_being_applied(aws):
    """It would be missing its last heading lines, and standardize would apply the ones
    it did get to the wrong headings."""
    response = succeeded_response(str(INDEX), stop_reason="max_tokens")

    with pytest.raises(Exception, match="truncated at max_tokens"):
        drain(client_returning(response))


@pytest.mark.parametrize(
    "block, expected",
    [
        (TextBlock(type="text", text="0|title"), {"type": "text", "text": "0|title"}),
        (
            ThinkingBlock(type="thinking", thinking="considering", signature="sig"),
            {"type": "thinking", "thinking": "considering"},
        ),
        (
            RedactedThinkingBlock(type="redacted_thinking", data="opaque"),
            {"type": "redacted_thinking"},
        ),
        (
            ToolUseBlock(type="tool_use", id="tu_1", name="lookup", input={"q": "rent"}),
            {"type": "tool_use", "name": "lookup", "input": {"q": "rent"}},
        ),
    ],
    ids=["text", "thinking", "redacted", "tool_use"],
)
def test_each_content_block_type_is_serialized_by_name(block, expected):
    assert serialize_content_block(block) == expected


def test_a_block_type_the_sdk_adds_later_is_logged_not_guessed_at(caplog):
    """Probing with hasattr would let it serialize to a near-empty dict in silence."""
    with caplog.at_level(logging.WARNING):
        assert serialize_content_block(SimpleNamespace(type="future_block")) == {
            "type": "future_block"
        }

    assert "unhandled content block type future_block" in caplog.text


# ── The batch manifest ────────────────────────────────────────────────


def test_a_manifest_round_trips_under_its_own_batch_id(aws):
    save_batch_index(BATCH_ID, [book(), book(INDEX_2)])

    mapping = load_batch_index(BATCH_ID).llm_index_mapping

    assert mapping == {str(INDEX): INDEX, str(INDEX_2): INDEX_2}


def test_a_manifest_belonging_to_another_batch_raises(bucket):
    """The key and the body have to agree: a manifest read under the wrong batch id
    would settle this batch's results against another batch's books."""
    from book_records.schemas import BatchDetail

    bucket.put_object(
        Key=f"standardize-headings/batch-details/{BATCH_ID}.json",
        Body=BatchDetail(
            llm_batch_id="msgbatch_other",
            llm_index_mapping={str(INDEX): str(INDEX)},
        ).model_dump_json().encode("utf-8"),
    )

    with pytest.raises(ValueError, match="manifest is for batch msgbatch_other"):
        load_batch_index(BATCH_ID)
