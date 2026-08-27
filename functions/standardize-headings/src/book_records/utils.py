import json
import logging

from botocore.exceptions import ClientError

from shared.s3 import get_s3_loader
from shared.tables.pipeline_entries import EntryStatus, get_pipeline_entries
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


def load_book_record(entry) -> tuple[str | None, str | None]:
    try:
        record = json.loads(get_s3_loader().load_text(entry.s3_metadata_key))
    except ClientError:
        logger.warning("%s: no metadata record; classifying without it", entry.book_id)
        return None, None

    def joined(field):
        return "; ".join(record.get(field, [])) or None

    return joined("title"), joined("author")


def save_book_tag_text_pairs(books_tag_text_pairs: list[BookTagTextPairs]) -> None:
    for book_tag_text_pairs in books_tag_text_pairs:
        get_s3_loader().upload_object(
            f"{S3_STANDARDIZE_PREFIX}/books/{book_tag_text_pairs.index}.json",
            book_tag_text_pairs.model_dump_json(),
            content_type=JSON_CONTENT_TYPE,
        )


def get_book_tag_text_pairs(entries) -> list[BookTagTextPairs]:

    book_tag_text_pairs = []

    for entry in entries:
        tag_text_pairs = load_tag_text_pairs(entry)

        tags = {tag for tag, _ in tag_text_pairs}
        if tags.isdisjoint(HEADING_ELEMENTS):
            get_pipeline_entries().set_status(
                entry.book_id, EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS
            )
            logger.info("%s has no headings; skipping.", entry.book_id)
            continue

        title, author = load_book_record(entry)

        book_tag_text_pairs.append(
            BookTagTextPairs(
                llm_index=sanitize_llm_index(entry.book_id),
                index=entry.book_id,
                tag_text_pairs=tag_text_pairs,
                title=title,
                author=author,
            )
        )

    return book_tag_text_pairs
