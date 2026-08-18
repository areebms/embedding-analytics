import logging

from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

from book_records.constants import (
    CUSTOM_ID_ILLEGAL,
    HEADING_ELEMENTS,
    JSON_CONTENT_TYPE,
    MANIFEST_PREFIX,
)
from book_records.html_text_tags import load_tag_text_pairs
from book_records.schemas import BatchDetail, BookRecord

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sanitize_custom_id(book_label: str) -> str:
    return CUSTOM_ID_ILLEGAL.sub("_", book_label)[:64]


def save_book_record(book: BookRecord) -> None:
    get_s3_loader().upload_object(
        f"{MANIFEST_PREFIX}/books/{book.index}.json",
        book.model_dump_json(),
        content_type=JSON_CONTENT_TYPE,
    )


def save_batch_index(batch_id: str, book_records: list[BookRecord]) -> None:
    batch_index = BatchDetail(
        batch_id=batch_id,
        custom_ids={book.custom_id: book.index for book in book_records},
    )
    get_s3_loader().upload_object(
        f"{MANIFEST_PREFIX}/index.json",
        batch_index.model_dump_json(),
        content_type=JSON_CONTENT_TYPE,
    )


def get_pending_book_records() -> list[BookRecord]:
    pipeline_entries = get_pipeline_entries()

    submitted_indexes = pipeline_entries.get_indexes(EntryStatus.STANDARDIZE_SUBMITTED)
    if submitted_indexes:
        logger.info(
            "submit: %d book(s) still in flight; nothing submitted",
            len(submitted_indexes),
        )
        return []

    scraped_indexes = pipeline_entries.get_indexes(EntryStatus.SCRAPED_HTML)
    logger.info(
        "submit: %d book(s) at %s", len(scraped_indexes), EntryStatus.SCRAPED_HTML
    )
    if not scraped_indexes:
        return []

    book_records = []

    for index in scraped_indexes:
        try:
            tag_text_pairs = load_tag_text_pairs(index)
        except Exception as error:
            logger.warning("%s could not load html: %s", index, error)
            continue

        tags = {tag for tag, _ in tag_text_pairs}
        if tags.isdisjoint(HEADING_ELEMENTS):
            pipeline_entries.update_entries(
                PipelineEntry(
                    platform_data=index,
                    pipeline_status=EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS,
                )
            )
            logger.info("%s has no headings; skipping.", index)
            continue

        book_records.append(
            BookRecord(
                custom_id=sanitize_custom_id(index),
                index=index,
                tag_text_pairs=tag_text_pairs,
            )
        )
        save_book_record(book_records[-1])

    return book_records
