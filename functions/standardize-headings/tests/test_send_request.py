"""Tests for the prompt half of the wire format and the batch submission itself.

The model never sees the book, only these heading detail lines, so what this module
builds is the entire input to the classification. The line format here and the reply
format read back in llm_parse_response/standardize.py are two halves of one contract.
"""

import pytest

from conftest import BOOK_PAIRS, INDEX, INDEX_2

from book_records.schemas import BookTagTextPairs
from llm_classify_request.constants import HEADING_TEXT_TRUNCATE
from llm_classify_request.send_request import (
    MAX_OUTPUT_TOKENS,
    HeadingSemanticBlockError,
    convert_to_anthropic_request,
    get_client,
    send_message_batch,
    to_anthropic_message,
)


def book(index=INDEX, tag_text_pairs=None, llm_index=None):
    return BookTagTextPairs(
        llm_index=llm_index or str(index),
        index=index,
        tag_text_pairs=BOOK_PAIRS if tag_text_pairs is None else tag_text_pairs,
    )


def heading_lines(content):
    return content.split("\n\n", 1)[1].splitlines()


# ── get_client ────────────────────────────────────────────────────────


def test_a_missing_api_key_is_refused_before_anything_is_submitted(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(HeadingSemanticBlockError, match="ANTHROPIC_API_KEY is not set"):
        get_client()


def test_a_client_is_built_when_the_key_is_present(monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert isinstance(get_client(), anthropic.Anthropic)


# ── to_anthropic_message ──────────────────────────────────────────────


def test_the_message_names_the_book_then_lists_its_headings():
    content = to_anthropic_message(INDEX, BOOK_PAIRS).content

    assert content.startswith("Book: gutenberg-3300\n\n")
    assert heading_lines(content) == [
        "0|h1|The Wealth of Nations|7",
        "1|h2|BOOK I.|0",
        "2|h2|OF THE CAUSES OF IMPROVEMENT.|9",
    ]


def test_positions_are_dense_and_count_only_headings():
    """The reply is keyed by these positions, so they have to match the order the
    headings are read back in, not the index of the block in the book."""
    pairs = [("p", "prose"), ("h1", "One"), ("p", "more"), ("h2", "Two")]

    assert [line.split("|")[0] for line in heading_lines(
        to_anthropic_message(INDEX, pairs).content
    )] == ["0", "1"]


def test_the_word_gap_is_the_prose_between_this_heading_and_the_next():
    """A run of zeroes is how the model recognises a table of contents."""
    pairs = [
        ("h1", "Title"),
        ("h2", "Contents"),
        ("h2", "Chapter I"),
        ("p", "one two three"),
    ]

    assert [line.split("|")[3] for line in heading_lines(
        to_anthropic_message(INDEX, pairs).content
    )] == ["0", "0", "3"]


def test_prose_before_the_first_heading_is_not_counted_anywhere():
    """There is no heading for it to belong to; attributing it to heading 0 would
    make a title look like it had content under it."""
    pairs = [("p", "a b c d e"), ("h1", "Title"), ("p", "one two")]

    assert heading_lines(to_anthropic_message(INDEX, pairs).content) == [
        "0|h1|Title|2"
    ]


def test_a_long_heading_is_truncated():
    pairs = [("h1", "A" * 200)]

    excerpt = heading_lines(to_anthropic_message(INDEX, pairs).content)[0].split("|")[2]
    assert excerpt == "A" * HEADING_TEXT_TRUNCATE


def test_pipes_and_newlines_in_a_heading_cannot_break_the_line_format():
    """A heading containing a pipe would otherwise add a phantom field."""
    pairs = [("h1", "before|after\nsecond line")]

    line = heading_lines(to_anthropic_message(INDEX, pairs).content)[0]
    assert line == "0|h1|before/after second line|0"
    assert len(line.split("|")) == 4


def test_the_original_tag_is_passed_through_unchanged():
    """Unreliable, but the prompt tells the model to weigh it, so it has to arrive."""
    pairs = [("h4", "Deep"), ("h1", "Shallow")]

    assert [line.split("|")[1] for line in heading_lines(
        to_anthropic_message(INDEX, pairs).content
    )] == ["h4", "h1"]


def test_a_book_with_no_headings_produces_no_lines():
    """submit filters these out before it gets here; this records what would be sent
    if that guard ever went missing."""
    assert heading_lines(to_anthropic_message(INDEX, [("p", "prose")]).content) == []


# ── convert_to_anthropic_request ──────────────────────────────────────


def test_the_request_is_keyed_by_the_sanitised_llm_index():
    request = convert_to_anthropic_request(book(llm_index="gutenberg-3300"))

    assert request.custom_id == "gutenberg-3300"


@pytest.mark.parametrize(
    "heading_count, expected",
    [(0, 256), (1, 256), (13, 256), (14, 268), (100, 1300), (2000, MAX_OUTPUT_TOKENS)],
)
def test_max_tokens_scales_with_headings_between_a_floor_and_a_ceiling(
    heading_count, expected
):
    """Twelve tokens a heading plus slack. The floor keeps a short book from being
    truncated by rounding; the ceiling keeps a pathological one from being priced
    as if the model would answer forever."""
    pairs = [("h1", "Title")] * heading_count

    assert convert_to_anthropic_request(book(tag_text_pairs=pairs)).params.max_tokens == expected


def test_paragraphs_do_not_count_toward_max_tokens():
    """The reply is one line per heading; prose is input, not output."""
    pairs = [("h1", "Title")] + [("p", "prose")] * 500

    assert convert_to_anthropic_request(book(tag_text_pairs=pairs)).params.max_tokens == 256


# ── send_message_batch ────────────────────────────────────────────────


def test_the_whole_corpus_goes_up_as_one_batch(mocker, anthropic_client):
    import llm_classify_request.send_request as send_request

    mocker.patch.object(send_request, "get_client", return_value=anthropic_client)

    batch_id = send_message_batch([book(INDEX), book(INDEX_2)])

    assert batch_id == "msgbatch_test123"
    anthropic_client.messages.batches.create.assert_called_once()
    requests = anthropic_client.messages.batches.create.call_args.kwargs["requests"]
    assert [request["custom_id"] for request in requests] == [
        "gutenberg-3300",
        "gutenberg-11",
    ]


def test_the_requests_are_plain_dicts_not_pydantic_models(mocker, anthropic_client):
    """The SDK serialises what it is given; a BaseModel would not survive the trip."""
    import llm_classify_request.send_request as send_request

    mocker.patch.object(send_request, "get_client", return_value=anthropic_client)

    send_message_batch([book(INDEX)])

    request = anthropic_client.messages.batches.create.call_args.kwargs["requests"][0]
    assert isinstance(request, dict)
    assert isinstance(request["params"]["messages"][0], dict)
