"""Tests for reading a finished batch back out of the Batch API.

Every result is archived to S3 before it is inspected, because a batch's results are
streamable once and the raw reply is the only evidence left if a book fails to parse.
"""

import json

import pytest

from conftest import (
    BATCH_ID,
    BOOK_PAIRS,
    INDEX,
    errored_response,
    s3_body,
    s3_content_type,
    succeeded_response,
)

from book_records.constants import JSON_CONTENT_TYPE
from book_records.schemas import BookTagTextPairs
from llm_parse_response.fetch import (
    get_batch_status,
    load_book_tag_text_pairs,
    save_batch_response,
    serialize_content_block,
    yield_anthropic_content,
)


def result_key(custom_id, batch_id=BATCH_ID):
    return f"standardize-headings/batch-results/{batch_id}/{custom_id}.json"


# ── get_batch_status ──────────────────────────────────────────────────


@pytest.mark.parametrize("status", ["in_progress", "canceling", "ended"])
def test_the_batch_status_is_reported_verbatim(batch_client, status):
    """standardize_from_batch compares this against "ended" exactly."""
    assert get_batch_status(batch_client(status), BATCH_ID) == status


def test_the_status_is_read_for_the_batch_that_was_asked_for(batch_client):
    client = batch_client("ended")

    get_batch_status(client, BATCH_ID)

    client.messages.batches.retrieve.assert_called_once_with(BATCH_ID)


# ── load_book_tag_text_pairs ──────────────────────────────────────────


def test_a_books_manifest_comes_back_as_the_model_submit_wrote(book_manifest, bucket):
    book_manifest(INDEX)

    loaded = load_book_tag_text_pairs(INDEX)

    assert isinstance(loaded, BookTagTextPairs)
    assert loaded.index == INDEX
    assert loaded.tag_text_pairs == BOOK_PAIRS


def test_loading_a_book_that_was_never_submitted_raises(bucket):
    with pytest.raises(Exception):
        load_book_tag_text_pairs(INDEX)


# ── save_batch_response ───────────────────────────────────────────────


def test_a_result_is_archived_under_its_batch_and_custom_id(bucket):
    save_batch_response(BATCH_ID, succeeded_response("gutenberg-3300", "0|title"))

    saved = json.loads(s3_body(bucket, result_key("gutenberg-3300")))
    assert saved["custom_id"] == "gutenberg-3300"
    assert saved["result"]["message"]["content"][0]["text"] == "0|title"
    assert s3_content_type(bucket, result_key("gutenberg-3300")) == JSON_CONTENT_TYPE


def test_results_from_different_batches_do_not_collide(bucket):
    save_batch_response("msgbatch_one", succeeded_response("gutenberg-3300", "0|title"))
    save_batch_response("msgbatch_two", succeeded_response("gutenberg-3300", "0|chapter"))

    assert "0|title" in s3_body(bucket, result_key("gutenberg-3300", "msgbatch_one"))
    assert "0|chapter" in s3_body(bucket, result_key("gutenberg-3300", "msgbatch_two"))


# ── serialize_content_block ───────────────────────────────────────────


def test_a_text_block_keeps_its_text():
    from anthropic.types import TextBlock

    assert serialize_content_block(TextBlock(type="text", text="0|title")) == {
        "type": "text",
        "text": "0|title",
    }


def test_a_thinking_block_keeps_its_thinking_but_not_its_signature():
    from anthropic.types import ThinkingBlock

    block = ThinkingBlock(type="thinking", thinking="weighing it up", signature="sig")

    assert serialize_content_block(block) == {
        "type": "thinking",
        "thinking": "weighing it up",
    }


def test_a_redacted_thinking_block_keeps_only_its_type():
    from anthropic.types import RedactedThinkingBlock

    block = RedactedThinkingBlock(type="redacted_thinking", data="encrypted")

    assert serialize_content_block(block) == {"type": "redacted_thinking"}


def test_a_tool_use_block_keeps_its_name_and_input():
    from anthropic.types import ToolUseBlock

    block = ToolUseBlock(type="tool_use", id="tu_1", name="lookup", input={"q": "x"})

    assert serialize_content_block(block) == {
        "type": "tool_use",
        "name": "lookup",
        "input": {"q": "x"},
    }


def test_a_block_type_this_code_does_not_know_is_named_rather_than_guessed(caplog):
    """Probing with hasattr would let a block type added in a future SDK release
    serialise to a near-empty dict in silence."""

    class FutureBlock:
        type = "server_tool_use"

    with caplog.at_level("WARNING"):
        assert serialize_content_block(FutureBlock()) == {"type": "server_tool_use"}

    assert "server_tool_use" in caplog.text


def test_a_block_with_no_type_at_all_falls_back_to_its_class_name(caplog):
    class Mystery:
        pass

    with caplog.at_level("WARNING"):
        assert serialize_content_block(Mystery()) == {"type": "Mystery"}


# ── yield_anthropic_content ───────────────────────────────────────────


def test_each_succeeded_result_yields_its_custom_id_and_blocks(batch_client, bucket):
    client = batch_client(
        "ended",
        [
            succeeded_response("gutenberg-3300", "0|title"),
            succeeded_response("gutenberg-11", "0|chapter"),
        ],
    )

    assert list(yield_anthropic_content(client, BATCH_ID)) == [
        ("gutenberg-3300", [{"type": "text", "text": "0|title"}]),
        ("gutenberg-11", [{"type": "text", "text": "0|chapter"}]),
    ]


def test_every_result_is_archived_as_it_is_streamed(batch_client, bucket):
    client = batch_client("ended", [succeeded_response("gutenberg-3300", "0|title")])

    list(yield_anthropic_content(client, BATCH_ID))

    assert s3_body(bucket, result_key("gutenberg-3300"))


def test_an_errored_result_stops_the_collection_and_names_the_error(
    batch_client, bucket
):
    client = batch_client("ended", [errored_response("gutenberg-3300")])

    with pytest.raises(Exception, match=r"batch result errored .*invalid_request_error: request too large"):
        list(yield_anthropic_content(client, BATCH_ID))


def test_a_failed_result_is_archived_before_it_raises(batch_client, bucket):
    """The reply is streamable once. Raising first would lose the only copy of why
    the batch failed."""
    client = batch_client("ended", [errored_response("gutenberg-3300")])

    with pytest.raises(Exception):
        list(yield_anthropic_content(client, BATCH_ID))

    assert json.loads(s3_body(bucket, result_key("gutenberg-3300")))["result"]["type"] == "errored"


def test_a_truncated_reply_is_refused_rather_than_half_applied(batch_client, bucket):
    """A reply cut off at max_tokens is missing its last headings, and applying it
    would silently leave them unclassified."""
    client = batch_client(
        "ended",
        [succeeded_response("gutenberg-3300", "0|title", stop_reason="max_tokens")],
    )

    with pytest.raises(Exception, match="response truncated at max_tokens"):
        list(yield_anthropic_content(client, BATCH_ID))


def test_the_books_before_a_failure_are_still_yielded(batch_client, bucket):
    """The generator is consumed one book at a time, so work already done stands."""
    client = batch_client(
        "ended",
        [
            succeeded_response("gutenberg-3300", "0|title"),
            errored_response("gutenberg-11"),
        ],
    )

    content = yield_anthropic_content(client, BATCH_ID)

    assert next(content)[0] == "gutenberg-3300"
    with pytest.raises(Exception):
        next(content)


def test_an_empty_batch_yields_nothing(batch_client, bucket):
    assert list(yield_anthropic_content(batch_client("ended", []), BATCH_ID)) == []
