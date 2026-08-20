"""Tests for the request models the Batch API validates on our behalf — after we have
already paid to submit.

custom_id is the only thing tying a reply back to a book, and the API rejects the whole
batch if one id is malformed, so the model refuses it locally first.
"""

import pytest
from pydantic import ValidationError

from llm_classify_request.constants import MODEL, SYSTEM_PROMPT
from llm_classify_request.schemas import (
    AnthropicRequest,
    AnthropicRequestMessage,
    AnthropicRequestParams,
)


def params(**overrides):
    return AnthropicRequestParams(
        **{
            "max_tokens": 256,
            "messages": [AnthropicRequestMessage(content="Book: gutenberg-3300")],
            **overrides,
        }
    )


# ── custom_id ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "custom_id", ["gutenberg-3300", "under_score", "MiXed123", "a", "x" * 64]
)
def test_an_id_anthropic_accepts_is_accepted(custom_id):
    assert AnthropicRequest(custom_id=custom_id, params=params()).custom_id == custom_id


@pytest.mark.parametrize(
    "custom_id", ["", "has space", "slash/", "dot.", "Ünïcødé", "pipe|"]
)
def test_an_id_anthropic_would_reject_is_refused_here(custom_id):
    with pytest.raises(ValidationError, match="must match"):
        AnthropicRequest(custom_id=custom_id, params=params())


def test_an_id_over_sixty_four_characters_is_refused():
    with pytest.raises(ValidationError):
        AnthropicRequest(custom_id="x" * 65, params=params())


def test_a_sanitised_book_label_always_passes():
    """sanitize_llm_index and this validator have to agree, or submit builds requests
    it cannot send."""
    from book_records.utils import sanitize_llm_index

    custom_id = sanitize_llm_index("The Wealth of Nations (1776) — vol. I/II")

    assert AnthropicRequest(custom_id=custom_id, params=params()).custom_id == custom_id


# ── params ────────────────────────────────────────────────────────────


def test_the_defaults_are_the_ones_the_batch_is_priced_around():
    """Thinking off and effort low: this is a mechanical mapping task run over the
    whole corpus, not a reasoning one."""
    request_params = params()

    assert request_params.model == MODEL
    assert request_params.thinking == {"type": "disabled"}
    assert request_params.output_config == {"effort": "low"}
    assert request_params.system == SYSTEM_PROMPT


def test_the_shared_default_dicts_are_not_shared_between_requests():
    """They are declared as mutable class-level defaults; pydantic deep-copies them
    per instance, and one book's params must never leak into another's."""
    first, second = params(), params()

    first.thinking["type"] = "enabled"

    assert second.thinking == {"type": "disabled"}


def test_max_tokens_has_no_default_because_it_is_computed_per_book():
    with pytest.raises(ValidationError):
        AnthropicRequestParams(messages=[AnthropicRequestMessage(content="x")])


def test_a_message_is_always_from_the_user():
    assert AnthropicRequestMessage(content="x").role == "user"


def test_the_dumped_request_is_the_shape_the_batch_api_takes():
    dumped = AnthropicRequest(custom_id="gutenberg-3300", params=params()).model_dump()

    assert dumped["custom_id"] == "gutenberg-3300"
    assert dumped["params"]["model"] == MODEL
    assert dumped["params"]["messages"] == [
        {"role": "user", "content": "Book: gutenberg-3300"}
    ]
