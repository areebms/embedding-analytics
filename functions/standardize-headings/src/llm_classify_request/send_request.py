import logging
import os

import anthropic

from book_records.constants import HEADING_ELEMENTS
from book_records.schemas import BookRecord, TagTextPair
from llm_classify_request.constants import HEADING_TEXT_TRUNCATE
from llm_classify_request.schemas import (
    AnthropicRequest,
    AnthropicRequestMessage,
    AnthropicRequestParams,
)

MAX_OUTPUT_TOKENS = 16000

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HeadingSemanticBlockError(Exception):
    """Classification did not produce a complete, trustworthy semantic block map."""


def get_client() -> anthropic.Anthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HeadingSemanticBlockError(
            "ANTHROPIC_API_KEY is not set; heading semantic blocks cannot be assigned"
        )
    return anthropic.Anthropic()


def to_anthropic_message(
    book_id: str, tag_text_pairs: list[TagTextPair]
) -> AnthropicRequestMessage:
    headings: list[tuple[str, str]] = []
    gaps: list[int] = []

    for tag, text in tag_text_pairs:
        if tag in HEADING_ELEMENTS:
            excerpt = text[:HEADING_TEXT_TRUNCATE].replace("|", "/").replace("\n", " ")
            headings.append((tag, excerpt))
            gaps.append(0)
        elif gaps:
            gaps[-1] += len(text.split())

    heading_lines = "\n".join(
        f"{position}|{tag}|{excerpt}|{gap}"
        for position, ((tag, excerpt), gap) in enumerate(zip(headings, gaps))
    )
    return AnthropicRequestMessage(content=f"Book: {book_id}\n\n{heading_lines}")


def convert_to_anthropic_request(book: BookRecord) -> AnthropicRequest:
    heading_count = sum(1 for tag, _ in book.tag_text_pairs if tag in HEADING_ELEMENTS)
    max_tokens = min(MAX_OUTPUT_TOKENS, max(256, heading_count * 12 + 100))

    return AnthropicRequest(
        custom_id=book.custom_id,
        params=AnthropicRequestParams(
            max_tokens=max_tokens,
            messages=[to_anthropic_message(book.index, book.tag_text_pairs)],
        ),
    )


def send_message_batch(book_records: list[BookRecord]) -> str:
    client = get_client()
    request_data = [
        convert_to_anthropic_request(book).model_dump() for book in book_records
    ]

    batch = client.messages.batches.create(requests=request_data)
    logger.info("batch %s: %d request(s) submitted", batch.id, len(request_data))
    return batch.id
