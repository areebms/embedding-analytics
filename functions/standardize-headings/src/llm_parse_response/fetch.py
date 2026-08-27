import logging
from collections.abc import Iterator
from typing import Any

from anthropic import Anthropic
from anthropic.types import (
    ContentBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from anthropic.types.messages import MessageBatchIndividualResponse

from shared.commons import BookIndex
from shared.s3 import get_s3_loader

from book_records.constants import JSON_CONTENT_TYPE, S3_STANDARDIZE_PREFIX
from book_records.schemas import BookTagTextPairs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BATCH_ENDED = "ended"


def get_batch_status(client: Anthropic, batch_id: str) -> str:
    batch = client.messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    logger.info(
        "batch %s: %s (processing=%s succeeded=%s errored=%s)",
        batch_id,
        batch.processing_status,
        counts.processing,
        counts.succeeded,
        counts.errored,
    )
    return batch.processing_status


def load_book_tag_text_pairs(index: BookIndex) -> BookTagTextPairs:
    return BookTagTextPairs.model_validate_json(
        get_s3_loader().load_text(f"{S3_STANDARDIZE_PREFIX}/books/{index}.json")
    )


def save_batch_response(
    batch_id: str, response: MessageBatchIndividualResponse
) -> None:
    get_s3_loader().upload_object(
        f"{S3_STANDARDIZE_PREFIX}/batch-results/{batch_id}/{response.custom_id}.json",
        response.to_json(indent=None),
        content_type=JSON_CONTENT_TYPE,
    )


def serialize_content_block(block: ContentBlock) -> dict[str, Any]:
    """One JSON-ready dict per content block, branched by named type.

    Probing with hasattr would let a block type added in a future SDK release
    serialize to a near-empty dict in silence; an unhandled type belongs in the log.
    """
    if isinstance(block, TextBlock):
        return {"type": block.type, "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": block.type, "thinking": block.thinking}
    if isinstance(block, RedactedThinkingBlock):
        return {"type": block.type}
    if isinstance(block, ToolUseBlock):
        return {"type": block.type, "name": block.name, "input": block.input}

    block_type = getattr(block, "type", type(block).__name__)
    logger.warning("unhandled content block type %s", block_type)
    return {"type": block_type}


def yield_anthropic_content(
    client: Anthropic, batch_id: str
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for response in client.messages.batches.results(batch_id):

        save_batch_response(batch_id, response)

        result = response.result

        if result.type != "succeeded":
            detail = ""
            if result.type == "errored":
                error = result.error.error
                detail = f" ({error.type}: {error.message})"
            logger.warning(
                "batch %s: %s result %s%s",
                batch_id,
                response.custom_id,
                result.type,
                detail,
            )
            continue

        message = result.message
        if message.stop_reason == "max_tokens":
            logger.warning(
                "batch %s: %s response truncated at max_tokens",
                batch_id,
                response.custom_id,
            )
            continue

        yield response.custom_id, [
            serialize_content_block(block) for block in message.content
        ]
