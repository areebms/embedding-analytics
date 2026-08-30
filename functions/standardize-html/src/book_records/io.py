import logging

from botocore.exceptions import ClientError

from shared.s3 import load_json, load_text, upload_json

from book_records.keys import batch_index_key, book_pairs_key
from book_records.schemas import BatchDetail, BookTagTextPairs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_batch_index(
    batch_id: str, book_tag_text_pairs: list[BookTagTextPairs]
) -> None:
    batch_index = BatchDetail(
        llm_batch_id=batch_id,
        book_ids=[pair.index for pair in book_tag_text_pairs],
    )
    upload_json(batch_index_key(batch_id), batch_index.model_dump_json())


def load_batch_index(batch_id: str) -> BatchDetail:
    batch_index = BatchDetail.model_validate_json(
        load_text(batch_index_key(batch_id))
    )

    if batch_index.llm_batch_id != batch_id:
        raise ValueError(
            f"manifest is for batch {batch_index.llm_batch_id}, not {batch_id}"
        )
    return batch_index


def load_html(entry):
    return load_text(entry.s3_html_key)


def load_metadata(entry) -> tuple[str | None, str | None]:
    try:
        record = load_json(entry.s3_metadata_key)
    except ClientError:
        logger.warning("%s: no metadata record; classifying without it", entry.book_id)
        return None, None

    def joined(field):
        return "; ".join(record.get(field, [])) or None

    return joined("title"), joined("author")


def save_book_tag_text_pairs(books_tag_text_pairs: list[BookTagTextPairs]) -> None:
    for book_tag_text_pairs in books_tag_text_pairs:
        upload_json(
            book_pairs_key(book_tag_text_pairs.index),
            book_tag_text_pairs.model_dump_json(),
        )
