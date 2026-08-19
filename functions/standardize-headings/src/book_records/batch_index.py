from shared.s3 import get_s3_loader

from book_records.constants import JSON_CONTENT_TYPE, S3_STANDARDIZE_PREFIX
from book_records.schemas import BatchDetail, BookTagTextPairs

BATCH_INDEX_KEY = f"{S3_STANDARDIZE_PREFIX}/batch-details/index.json"


def save_batch_index(
    batch_id: str, book_tag_text_pairs: list[BookTagTextPairs]
) -> None:
    batch_index = BatchDetail(
        llm_batch_id=batch_id,
        llm_index_mapping={
            book_tag_text_pair.llm_index: book_tag_text_pair.index
            for book_tag_text_pair in book_tag_text_pairs
        },
    )
    get_s3_loader().upload_object(
        BATCH_INDEX_KEY,
        batch_index.model_dump_json(),
        content_type=JSON_CONTENT_TYPE,
    )


def load_batch_index(batch_id: str) -> BatchDetail:
    batch_index = BatchDetail.model_validate_json(
        get_s3_loader().load_text(BATCH_INDEX_KEY)
    )
    if batch_index.llm_batch_id != batch_id:
        raise ValueError(
            f"manifest is for batch {batch_index.llm_batch_id}, not {batch_id}"
        )
    return batch_index
