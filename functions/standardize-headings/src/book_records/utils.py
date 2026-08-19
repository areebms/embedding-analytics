import logging

from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)

from book_records.constants import (
    HEADING_ELEMENTS,
    JSON_CONTENT_TYPE,
    LLM_INDEX_ILLEGAL,
    S3_STANDARDIZE_PREFIX,
)
from book_records.html_text_tags import load_tag_text_pairs
from book_records.schemas import BookTagTextPairs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sanitize_llm_index(book_label: str) -> str:
    return LLM_INDEX_ILLEGAL.sub("_", book_label)[:64]


def save_book_tag_text_pairs(book_tag_text_pairs: BookTagTextPairs) -> None:
    get_s3_loader().upload_object(
        f"{S3_STANDARDIZE_PREFIX}/books/{book_tag_text_pairs.index}.json",
        book_tag_text_pairs.model_dump_json(),
        content_type=JSON_CONTENT_TYPE,
    )


def get_pending_book_tag_text_pairs() -> list[BookTagTextPairs]:
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

    book_tag_text_pairs = []

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

        book_tag_text_pairs.append(
            BookTagTextPairs(
                llm_index=sanitize_llm_index(index),
                index=index,
                tag_text_pairs=tag_text_pairs,
            )
        )
        save_book_tag_text_pairs(book_tag_text_pairs[-1])

    return book_tag_text_pairs
