import logging
import os

import anthropic

from shared.commons import BookIndex

from book_records.schemas import BookTagTextPairs, TagTextPair
from constants import (
    HEADING_ELEMENTS,
    HEADING_TEXT_TRUNCATE,
    MAX_OUTPUT_TOKENS,
)
from llm_request.schemas import (
    AnthropicRequest,
    AnthropicRequestMessage,
    AnthropicRequestParams,
)

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


def max_output_tokens(heading_count: int) -> int:
    """The reply is one short line per heading, so the budget is set from the count."""
    return min(MAX_OUTPUT_TOKENS, max(256, heading_count * 12 + 100))


def heading_gaps(tag_text_pairs) -> list[tuple[str, str, int]]:
    """The headings of a book, each with the prose that follows it.

    The third field is the word count of the plain text separating a heading from the
    next one. It is the signal a table of contents gives itself away by -- its entries
    have nothing between them at all -- which is why every heading line carries it.
    """
    headings = []
    for tag, text in tag_text_pairs:
        if tag in HEADING_ELEMENTS:
            headings.append((tag, text, 0))
        elif headings:
            last_tag, last_text, gap = headings[-1]
            headings[-1] = (last_tag, last_text, gap + len(text.split()))
    return headings


def to_anthropic_message(
    book_id: BookIndex,
    tag_text_pairs: list[TagTextPair],
    title: str | None = None,
    author: str | None = None,
) -> AnthropicRequestMessage:
    lines = []
    for position, (tag, text, gap) in enumerate(heading_gaps(tag_text_pairs)):
        excerpt = text[:HEADING_TEXT_TRUNCATE].replace("|", "/").replace("\n", " ")
        lines.append(f"{position}|{tag}|{excerpt}|{gap}")
    heading_lines = "\n".join(lines)
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
    max_tokens = max_output_tokens(heading_count)

    return AnthropicRequest(
        custom_id=str(book_tag_text_pairs.index),
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
