import logging

from shared.tables.pipeline_entries import (
    EntryStatus,
    get_pipeline_entries,
)

from book_records.io import load_batch_index
from book_records.schemas import (
    StandardizedBlock,
    TagTextPair,
)
from constants import (
    BATCH_ENDED,
    DEFAULT_BLOCK,
    HEADING_ELEMENTS,
    SEMANTIC_BLOCK_TO_LEVEL,
)
from llm_request.send_anthropic_request import get_client
from llm_response.fetch import (
    get_batch_status,
    load_book_tag_text_pairs,
    yield_anthropic_content,
)
from llm_response.save_artifacts import save_html, save_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_llm_content_text(llm_content):
    texts = []
    for block in llm_content:
        if block.get("type") == "text" and block["text"]:
            texts.append(block["text"])
        else:
            logger.warning("skipping %s block", block.get("type"))
    return "\n".join(texts)


def parse_semantic_blocks(llm_response: str) -> dict[int, str]:
    """Position -> semantic block, skipping every line that is not one.

    The reply is prose from a model, not a wire format, and one stray line used to
    cost a whole book: gutenberg-38194 came back with all 174 classifications correct
    and wrapped in ``` fences, and raising on the first line without a separator threw
    the other 174 away. Skipping keeps the good lines; the caller fills the gaps.
    """
    semantic_blocks = {}
    skipped = []
    for line in llm_response.strip().splitlines():
        if not line.strip():
            continue

        position_text, separator, semantic_block_text = line.partition("|")
        position_text = position_text.strip()
        semantic_block = semantic_block_text.strip().lower()

        if not separator or not position_text.isdigit():
            skipped.append(line)
            continue
        if semantic_block not in SEMANTIC_BLOCK_TO_LEVEL:
            skipped.append(line)
            continue

        semantic_blocks[int(position_text)] = semantic_block

    if skipped:
        logger.warning(
            "skipped %d unparseable line(s), first few: %s", len(skipped), skipped[:5]
        )
    return semantic_blocks


def standardize_tag_text_pairs(
    llm_response: str, tag_text_pairs: list[TagTextPair]
) -> list[StandardizedBlock]:
    """Apply the model's classification to a book.

    Every label comes from the reply. The heading list carries signals a rule could read
    instead -- a table of contents is a run of headings with no prose between them, a
    title page is the leading run that reproduces the library record -- and the prompt
    states them as rules for the model to apply, rather than this stage applying them
    itself. What the code still decides is what to do with a heading the reply did not
    name, which is the one thing the reply cannot say.
    """
    semantic_blocks = parse_semantic_blocks(llm_response)
    if not semantic_blocks:
        raise ValueError("no semantic blocks in the llm response")

    standardized_blocks = []
    defaulted = []
    position = 0
    current_block = None
    for tag, block_text in tag_text_pairs:
        if tag not in HEADING_ELEMENTS:
            standardized_blocks.append(
                StandardizedBlock(tag, block_text, current_block)
            )
            continue

        semantic_block = semantic_blocks.pop(position, None)
        if semantic_block is None:
            # A heading with no classification takes the default rather than the block
            # around it. Inheriting is right for a paragraph, which belongs to the
            # chapter it sits in, and wrong for a heading, which sits *under* what
            # precedes it: a subsection beneath a chapter would be promoted into a
            # chapter of its own. Replaying the two collected batches with a label
            # removed, inheritance invented 474 chapters the books do not have and
            # changed 7,609 blocks; defaulting changed 296 and invented none.
            semantic_block = DEFAULT_BLOCK
            defaulted.append(position)

        current_block = semantic_block
        position += 1
        standardized_blocks.append(
            StandardizedBlock(
                SEMANTIC_BLOCK_TO_LEVEL[semantic_block], block_text, semantic_block
            )
        )

    if defaulted:
        logger.warning(
            "%d of %d heading(s) had no classification and defaulted to %s; "
            "positions: %s",
            len(defaulted),
            position,
            DEFAULT_BLOCK,
            defaulted[:20],
        )

    return standardized_blocks


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

    entries = get_pipeline_entries().get_entries(load_batch_index(batch_id).book_ids)
    pending = {str(entry.book_id): entry for entry in entries}

    standardized = 0
    failed = []

    for custom_id, content in yield_anthropic_content(client, batch_id):
        entry = pending.pop(custom_id, None)
        if entry is None:
            logger.warning("batch %s: unknown custom_id %s", batch_id, custom_id)
            continue

        try:
            book_tag_text_pairs = load_book_tag_text_pairs(entry)
            standardized_tag_text_pairs = standardize_tag_text_pairs(
                get_llm_content_text(content), book_tag_text_pairs.tag_text_pairs
            )

            save_html(entry, standardized_tag_text_pairs, book_tag_text_pairs.title)
            save_text(entry, standardized_tag_text_pairs)
        except Exception:
            logger.exception("batch %s: %s could not be rendered; left at %s",
                             batch_id, entry.book_id, EntryStatus.STANDARDIZE_SUBMITTED)
            failed.append(str(entry.book_id))
            continue

        get_pipeline_entries().set_status(entry.book_id, EntryStatus.STANDARDIZED)
        standardized += 1

    if pending:
        logger.warning(
            "batch %s: %d book(s) had no result and remain at %s: %s",
            batch_id,
            len(pending),
            EntryStatus.STANDARDIZE_SUBMITTED,
            sorted(pending),
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
