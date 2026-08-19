from shared.s3 import get_s3_loader

from book_records.constants import JSON_CONTENT_TYPE, S3_STANDARDIZE_PREFIX
from book_records.schemas import BatchDetail, BookRecord

# The roster of a submitted batch, written by submit and read back by collect. The
# two live together so they cannot drift onto different keys, and this module stays
# free of book_records.utils' BeautifulSoup dependency so the collect image -- which
# installs no bs4 -- can import it.
BATCH_INDEX_KEY = f"{S3_STANDARDIZE_PREFIX}/batch-details/index.json"


def save_batch_index(batch_id: str, book_records: list[BookRecord]) -> None:
    batch_index = BatchDetail(
        batch_id=batch_id,
        id_mapping={book.custom_id: book.index for book in book_records},
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
    if batch_index.batch_id != batch_id:
        raise ValueError(f"manifest is for batch {batch_index.batch_id}, not {batch_id}")
    return batch_index
