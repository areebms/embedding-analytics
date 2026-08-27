import logging

from shared.tables.pipeline_entries import (
    EntryStatus,
    get_pipeline_entries,
)

from book_records.batch_index import load_batch_index
from book_records.constants import HEADING_ELEMENTS
from book_records.schemas import StandardizedBlock, TagTextPair
from llm_classify_request.send_anthropic_request import get_client
from llm_parse_response.fetch import (
    BATCH_ENDED,
    get_batch_status,
    load_book_tag_text_pairs,
    yield_anthropic_content,
)
from llm_parse_response.save_artifacts import save_html, save_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SEMANTIC_BLOCK_TO_LEVEL = {
    "title": "h1",
    "front_matter": "h3",
    "back_matter": "h3",
    "contents": "h3",
    "index": "h3",
    "errata": "h3",
    "advertisement": "h3",
    "part": "h2",
    "chapter": "h2",
    "section": "h3",
    "subsection": "h3",
}


def get_llm_content_text(llm_content):
    texts = []
    for block in llm_content:
        if block.get("type") == "text" and block["text"]:
            texts.append(block["text"])
        else:
            logger.warning("skipping %s block", block.get("type"))
    return "\n".join(texts)


TITLE_BLOCK = "title"
HEADING_LEVELS = frozenset(SEMANTIC_BLOCK_TO_LEVEL.values())


def merge_title_headings(blocks: list[StandardizedBlock]) -> list[StandardizedBlock]:
    """Fold a title page's lines back into the one heading they spell out.
    """
    merged: list[StandardizedBlock] = []
    for entry in blocks:
        joinable = (
            merged
            and entry.block == TITLE_BLOCK
            and merged[-1].block == TITLE_BLOCK
            and entry.tag in HEADING_LEVELS
            and merged[-1].tag in HEADING_LEVELS
        )
        if joinable:
            merged[-1] = merged[-1]._replace(text=f"{merged[-1].text} {entry.text}")
            continue
        merged.append(entry)
    return merged


def standardize_tag_text_pairs(
    llm_response: str, tag_text_pairs: list[TagTextPair]
) -> list[StandardizedBlock]:

    semantic_blocks = {}
    for line in llm_response.strip().splitlines():
        if not line.strip():
            continue

        position_text, separator, semantic_block_text = line.partition("|")
        if not separator:
            raise ValueError(f"no separator in line: {line}")
        semantic_block = semantic_block_text.strip().lower()
        if semantic_block not in SEMANTIC_BLOCK_TO_LEVEL:
            raise ValueError(
                f"unknown semantic block {semantic_block!r} in line: {line}"
            )

        semantic_blocks[int(position_text.strip())] = semantic_block

    standardized_blocks = []
    position = 0
    current_block = None
    for tag, block_text in tag_text_pairs:
        if tag not in HEADING_ELEMENTS:
            standardized_blocks.append(
                StandardizedBlock(tag, block_text, current_block)
            )
            continue
        if position not in semantic_blocks:
            raise ValueError(
                f"llm assigned no semantic block for heading position {position}"
            )
        current_block = semantic_blocks.pop(position)
        position += 1
        standardized_blocks.append(
            StandardizedBlock(
                SEMANTIC_BLOCK_TO_LEVEL[current_block], block_text, current_block
            )
        )

    return merge_title_headings(standardized_blocks)


def standardize_from_batch(batch_id):
    """Settle one submitted batch, if it has finished.
    """
    client = get_client()

    batch_status = get_batch_status(client, batch_id)
    if batch_status != BATCH_ENDED:
        return {
            "batch_id": batch_id,
            "batch_status": batch_status,
            "standardized": 0,
            "failed": [],
        }

    llm_index_mapping = dict(load_batch_index(batch_id).llm_index_mapping)

    standardized = 0
    failed = []

    for llm_index, content in yield_anthropic_content(client, batch_id):
        index = llm_index_mapping.pop(llm_index, None)
        if index is None:
            logger.warning("batch %s: unknown llm_index %s", batch_id, llm_index)
            continue

        try:
            book_tag_text_pairs = load_book_tag_text_pairs(index)
            standardized_tag_text_pairs = standardize_tag_text_pairs(
                get_llm_content_text(content), book_tag_text_pairs.tag_text_pairs
            )

            save_html(index, standardized_tag_text_pairs, book_tag_text_pairs.title)
            save_text(index, standardized_tag_text_pairs)
        except Exception:
            logger.exception("batch %s: %s could not be rendered; left at %s",
                             batch_id, index, EntryStatus.STANDARDIZE_SUBMITTED)
            failed.append(str(index))
            continue

        get_pipeline_entries().set_status(index, EntryStatus.STANDARDIZED)
        standardized += 1

    if llm_index_mapping:
        logger.warning(
            "batch %s: %d book(s) had no result and remain at %s: %s",
            batch_id,
            len(llm_index_mapping),
            EntryStatus.STANDARDIZE_SUBMITTED,
            sorted(str(index) for index in llm_index_mapping.values()),
        )

    logger.info(
        "batch %s: %d standardized, %d failed", batch_id, standardized, len(failed)
    )
    return {
        "batch_id": batch_id,
        "batch_status": batch_status,
        "standardized": standardized,
        "failed": failed,
    }
