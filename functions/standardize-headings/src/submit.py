import logging

from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

from book_records.batch_index import save_batch_index
from book_records.utils import get_pending_book_tag_text_pairs
from llm_classify_request.send_request import send_message_batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def submit():
    book_tag_text_pairs = get_pending_book_tag_text_pairs()

    if not book_tag_text_pairs:
        logger.info("submit: nothing to classify")
        return {"batch_id": None, "book_count": 0}

    batch_id = send_message_batch(book_tag_text_pairs)

    save_batch_index(batch_id, book_tag_text_pairs)

    pipeline_entries = get_pipeline_entries()

    for book_tag_text_pair in book_tag_text_pairs:
        pipeline_entries.update_entries(
            PipelineEntry(
                platform_data=book_tag_text_pair.index,
                pipeline_status=EntryStatus.STANDARDIZE_SUBMITTED,
            )
        )

    return {
        "batch_id": batch_id,
        "book_count": len(book_tag_text_pairs),
    }
