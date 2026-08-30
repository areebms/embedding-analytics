from typing import NamedTuple

from pydantic import BaseModel

from shared.tables.pipeline_entries import BookIndexField

TagTextPair = tuple[str, str]

S3_STANDARDIZE_PREFIX = "standardize-html"


class StandardizedBlock(NamedTuple):
    tag: str
    text: str
    block: str | None


class BookTagTextPairs(BaseModel):
    index: BookIndexField
    tag_text_pairs: list[TagTextPair]
    title: str | None = None
    author: str | None = None


class BatchDetail(BaseModel):
    llm_batch_id: str
    book_ids: list[BookIndexField]

    @classmethod
    def s3_key(cls, batch_id: str) -> str:
        return f"{S3_STANDARDIZE_PREFIX}/batch-details/{batch_id}.json"

    @classmethod
    def s3_result_key(cls, batch_id: str, custom_id: str) -> str:
        return f"{S3_STANDARDIZE_PREFIX}/batch-results/{batch_id}/{custom_id}.json"
