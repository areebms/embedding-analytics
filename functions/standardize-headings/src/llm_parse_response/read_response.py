import logging

from anthropic.types import (
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from shared.s3 import get_s3_loader

from book_records.constants import (
    HEADING_ELEMENTS,
    JSON_CONTENT_TYPE,
    S3_STANDARDIZE_PREFIX,
)
from book_records.schemas import BookRecord
from llm_classify_request.send_request import HeadingSemanticBlockError
from llm_parse_response.constants import HTML_SEMANTIC_BLOCKS
from llm_parse_response.schemas import AnthropicResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_batch_status(client, batch_id):
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


def read_book_record(index):
    return BookRecord.model_validate_json(
        get_s3_loader().load_text(f"{S3_STANDARDIZE_PREFIX}/books/{index}.json")
    )


def save_batch_result(batch_id, result):
    get_s3_loader().upload_object(
        f"{S3_STANDARDIZE_PREFIX}/batch-results/{batch_id}/{result.custom_id}.json",
        result.to_json(indent=None),
        content_type=JSON_CONTENT_TYPE,
    )


def process_result(text, book):
    """Read one book's reply into rows, and report what it left out.

    A line the model got wrong is logged and dropped rather than raised on: the prompt
    forbids prose and fences, but a stray line must not cost the whole book -- only an
    unclassified heading does that, and the gap shows up in `missing` either way. The
    reply itself stays readable under `batch-results/`.
    """
    num_headings = sum(1 for tag, _ in book.tag_text_pairs if tag in HEADING_ELEMENTS)

    rows = []
    returned_positions = set()
    for line in text.strip().splitlines():
        if not line.strip():
            continue

        position_text, separator, semantic_block_text = line.partition("|")
        if not separator:
            raise ValueError("no separator")
        position = int(position_text.strip())
        semantic_block = semantic_block_text.strip().lower()
        if not 0 <= position < num_headings:
            raise ValueError(f"position outside 0..{num_headings}")
        if semantic_block not in HTML_SEMANTIC_BLOCKS:
            raise ValueError(f"unknown semantic block {semantic_block!r}")

        rows.append(AnthropicResponse(position=position, semantic_block=semantic_block))
        returned_positions.add(position)

    missing = set(range(num_headings)) - returned_positions
    if missing:
        raise HeadingSemanticBlockError(
            f"{len(missing)} of {num_headings} headings unclassified "
            f"(first missing position: {min(missing)})"
        )

    return rows


def serialize_content_block(block):
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


def yield_anthropic_content(client, batch_id):
    for result in client.messages.batches.results(batch_id):

        save_batch_result(batch_id, result)

        if result.result.type != "succeeded":
            detail = ""
            if result.result.type == "errored":
                error = result.result.error.error
                detail = f" ({error.type}: {error.message})"
            raise Exception(f"batch result {result.result.type}{detail}")

        message = result.result.message
        if message.stop_reason == "max_tokens":
            raise Exception("response truncated at max_tokens")

        yield result.custom_id, [
            serialize_content_block(block) for block in message.content
        ]
