from typing import NamedTuple

from pydantic import BaseModel

from shared.tables.pipeline_entries import BookIndexField

TagTextPair = tuple[str, str]


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
