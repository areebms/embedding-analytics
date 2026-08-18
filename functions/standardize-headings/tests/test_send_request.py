"""Tests for the Anthropic client and the one batch call this function makes."""

import pytest

from conftest import BATCH_ID


def _book(heading_count, index="gutenberg-1"):
    from book_records.schemas import BookRecord

    pairs = [("h2", f"Heading {position}") for position in range(heading_count)]
    return BookRecord(custom_id=index, index=index, tag_text_pairs=pairs)


# ── the client ────────────────────────────────────────────────────────


def test_a_missing_api_key_fails_before_any_batch_is_created(monkeypatch):
    """Never inherited from the ambient environment: the deploy gate runs the suite
    with --env-file .env, so the key may or may not be there."""
    import llm_classify_request.send_request as send_request_module

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(
        send_request_module.HeadingSemanticBlockError, match="ANTHROPIC_API_KEY"
    ):
        send_request_module.get_client()


def test_an_empty_api_key_is_treated_as_missing(monkeypatch):
    import llm_classify_request.send_request as send_request_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(send_request_module.HeadingSemanticBlockError):
        send_request_module.get_client()


def test_a_present_api_key_builds_an_anthropic_client(monkeypatch, mocker):
    import llm_classify_request.send_request as send_request_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    anthropic_class = mocker.patch.object(send_request_module.anthropic, "Anthropic")

    assert send_request_module.get_client() is anthropic_class.return_value


# ── building one request ──────────────────────────────────────────────


def test_the_request_carries_the_books_id_and_its_headings():
    from llm_classify_request.send_request import convert_to_anthropic_request

    request = convert_to_anthropic_request(_book(2, "gutenberg-3300"))

    assert request.custom_id == "gutenberg-3300"
    assert request.params.messages[0].content.startswith("Book: gutenberg-3300")
    assert "0|h2|Heading 0|0" in request.params.messages[0].content


@pytest.mark.parametrize(
    "heading_count, expected",
    [(0, 256), (13, 256), (100, 1300), (1325, 16000), (1326, 16000)],
    ids=["floor", "floor-boundary", "scaling", "ceiling-boundary", "ceiling"],
)
def test_max_tokens_is_clamped_at_both_ends(heading_count, expected):
    """One reply line per heading, so the budget tracks the heading count — but a
    book dense enough to exceed the ceiling gets a truncated reply, and collect
    then sees fewer lines than it has headings."""
    from llm_classify_request.send_request import convert_to_anthropic_request

    request = convert_to_anthropic_request(_book(heading_count))

    assert request.params.max_tokens == expected


# ── sending the batch ─────────────────────────────────────────────────


def test_send_message_batch_sends_one_request_per_book(mocker, anthropic_client):
    import llm_classify_request.send_request as send_request_module

    mocker.patch.object(
        send_request_module, "get_client", return_value=anthropic_client
    )

    batch_id = send_request_module.send_message_batch(
        [_book(1, "gutenberg-1"), _book(2, "gutenberg-2")]
    )

    assert batch_id == BATCH_ID
    requests = anthropic_client.messages.batches.create.call_args.kwargs["requests"]
    assert [request["custom_id"] for request in requests] == [
        "gutenberg-1",
        "gutenberg-2",
    ]


def test_every_request_is_plain_json_the_sdk_can_serialize(mocker, anthropic_client):
    import llm_classify_request.send_request as send_request_module

    mocker.patch.object(
        send_request_module, "get_client", return_value=anthropic_client
    )

    send_request_module.send_message_batch([_book(1)])

    request = anthropic_client.messages.batches.create.call_args.kwargs["requests"][0]
    assert isinstance(request, dict)
    assert set(request["params"]) == {
        "model",
        "max_tokens",
        "thinking",
        "output_config",
        "system",
        "messages",
    }
