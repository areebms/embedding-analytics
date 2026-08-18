import logging

from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

from book_records.utils import get_pending_book_records, save_batch_index
from llm_classify_request.send_request import send_message_batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def submit():
    book_records = get_pending_book_records()

    if not book_records:
        logger.info("submit: nothing to classify")
        return {"batch_id": None, "book_count": 0}

    batch_id = send_message_batch(book_records)

    save_batch_index(batch_id, book_records)

    pipeline_entries = get_pipeline_entries()

    for book in book_records:
        pipeline_entries.update_entries(
            PipelineEntry(
                platform_data=book.index,
                pipeline_status=EntryStatus.STANDARDIZE_SUBMITTED,
            )
        )

    return {
        "batch_id": batch_id,
        "book_count": len(book_records),
    }
