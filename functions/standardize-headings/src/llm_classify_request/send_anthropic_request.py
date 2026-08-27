import logging
import os

import anthropic

from shared.commons import BookIndex

from book_records.constants import HEADING_ELEMENTS
from book_records.schemas import BookTagTextPairs, TagTextPair
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
    book_id: BookIndex,
    tag_text_pairs: list[TagTextPair],
    title: str | None = None,
    author: str | None = None,
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
    preamble = f"Book: {book_id}"
    if title:
        preamble += f"\nKnown title (from the library record): {title}"
    if author:
        preamble += f"\nKnown author: {author}"
    return AnthropicRequestMessage(content=f"{preamble}\n\n{heading_lines}")


def convert_to_anthropic_request(
    book_tag_text_pairs: BookTagTextPairs,
) -> AnthropicRequest:
    heading_count = sum(
        1 for tag, _ in book_tag_text_pairs.tag_text_pairs if tag in HEADING_ELEMENTS
    )
    max_tokens = min(MAX_OUTPUT_TOKENS, max(256, heading_count * 12 + 100))

    return AnthropicRequest(
        custom_id=book_tag_text_pairs.llm_index,
        params=AnthropicRequestParams(
            max_tokens=max_tokens,
            messages=[
                to_anthropic_message(
                    book_tag_text_pairs.index,
                    book_tag_text_pairs.tag_text_pairs,
                    book_tag_text_pairs.title,
                    book_tag_text_pairs.author,
                )
            ],
        ),
    )


def send_message_batch(book_tag_text_pairs: list[BookTagTextPairs]) -> tuple[str, str]:
    """Open one batch. Returns its id and the processing_status it opened at."""
    client = get_client()
    request_data = [
        convert_to_anthropic_request(book_tag_text_pair).model_dump()
        for book_tag_text_pair in book_tag_text_pairs
    ]

    batch = client.messages.batches.create(requests=request_data)
    logger.info("batch %s: %d request(s) submitted", batch.id, len(request_data))
    return batch.id, batch.processing_status
