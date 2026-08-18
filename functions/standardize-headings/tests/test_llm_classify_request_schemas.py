"""Tests for the wire format sent to the Batches API.

Pure pydantic and string building — the prompt Claude sees is assembled here, and
collect maps positions in the reply back onto the same headings, so the numbering
and the field separator are a contract between the two stages.
"""

import pytest
from pydantic import ValidationError


def _content(book_id, tag_text_pairs):
    from llm_classify_request.send_request import to_anthropic_message

    return to_anthropic_message(book_id, tag_text_pairs).content


def _lines(book_id, tag_text_pairs):
    """Just the heading detail lines, without the `Book: ...` header."""
    body = _content(book_id, tag_text_pairs).split("\n\n", 1)
    return body[1].split("\n") if len(body) > 1 and body[1] else []


# ── the message ───────────────────────────────────────────────────────


def test_the_message_is_a_single_user_turn():
    from llm_classify_request.send_request import to_anthropic_message

    message = to_anthropic_message("gutenberg-1", [])

    assert message.role == "user"


def test_the_message_opens_with_the_book_it_is_about():
    assert _content("gutenberg-3300", []).startswith("Book: gutenberg-3300")


def test_a_book_with_no_headings_gives_a_header_only_message():
    assert _content("gutenberg-1", [("p", "prose")]) == "Book: gutenberg-1\n\n"


def test_each_heading_becomes_one_pipe_delimited_line():
    lines = _lines("gutenberg-1", [("h1", "Title"), ("h2", "Chapter I")])

    assert lines == ["0|h1|Title|0", "1|h2|Chapter I|0"]


# ── the word-gap column ───────────────────────────────────────────────


def test_the_gap_counts_the_words_up_to_the_next_heading():
    pairs = [("h1", "Title"), ("p", "one two three"), ("h2", "Chapter I")]

    assert _lines("gutenberg-1", pairs)[0] == "0|h1|Title|3"


def test_prose_split_across_blocks_sums_into_one_gap():
    pairs = [("h1", "Title"), ("p", "one two"), ("p", "three"), ("h2", "Chapter I")]

    assert _lines("gutenberg-1", pairs)[0] == "0|h1|Title|3"


def test_adjacent_headings_leave_a_gap_of_zero():
    """Zero is what tells Claude a run of headings is a table of contents."""
    pairs = [("h1", "Title"), ("h2", "Chapter I")]

    assert _lines("gutenberg-1", pairs)[0] == "0|h1|Title|0"


def test_prose_before_the_first_heading_is_not_counted_anywhere():
    """It has no heading to belong to, so it is dropped rather than folded into
    the first heading's gap."""
    pairs = [("p", "one two three four"), ("h1", "Title")]

    assert _lines("gutenberg-1", pairs) == ["0|h1|Title|0"]


# ── the excerpt column ────────────────────────────────────────────────


def test_a_long_heading_is_truncated_to_the_prompt_budget():
    from llm_classify_request.constants import HEADING_TEXT_TRUNCATE

    excerpt = _lines("gutenberg-1", [("h1", "x" * 500)])[0].split("|")[2]

    assert excerpt == "x" * HEADING_TEXT_TRUNCATE


def test_the_field_separator_and_newlines_are_scrubbed_from_the_excerpt():
    """A raw pipe or newline in a heading would desync every column after it."""
    excerpt = _lines("gutenberg-1", [("h1", "A|B\nC")])[0].split("|")[2]

    assert excerpt == "A/B C"


# ── the request envelope ──────────────────────────────────────────────


def test_the_request_defaults_everything_except_the_per_book_fields():
    from llm_classify_request.constants import MODEL, SYSTEM_PROMPT
    from llm_classify_request.schemas import AnthropicRequestParams

    params = AnthropicRequestParams(max_tokens=256, messages=[])

    assert params.model == MODEL
    assert params.system == SYSTEM_PROMPT
    assert params.thinking == {"type": "disabled"}
    assert params.output_config == {"effort": "low"}


def test_max_tokens_has_no_default():
    from llm_classify_request.schemas import AnthropicRequestParams

    with pytest.raises(ValidationError):
        AnthropicRequestParams(messages=[])


@pytest.mark.parametrize(
    "custom_id",
    ["book/with spaces", "", "a" * 65],
    ids=["illegal-characters", "empty", "over-64-characters"],
)
def test_a_custom_id_anthropic_would_reject_never_reaches_the_api(custom_id):
    """Three different guards: the regex, the empty check, and pydantic's own
    max_length — which fires before the validator ever runs."""
    from llm_classify_request.schemas import AnthropicRequest, AnthropicRequestParams

    with pytest.raises(ValidationError):
        AnthropicRequest(
            custom_id=custom_id,
            params=AnthropicRequestParams(max_tokens=256, messages=[]),
        )


def test_a_sanitized_custom_id_is_accepted():
    from llm_classify_request.schemas import AnthropicRequest, AnthropicRequestParams

    request = AnthropicRequest(
        custom_id="gutenberg-3300",
        params=AnthropicRequestParams(max_tokens=256, messages=[]),
    )

    assert request.custom_id == "gutenberg-3300"
