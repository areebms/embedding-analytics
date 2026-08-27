from typing import NamedTuple

from pydantic import BaseModel

from shared.tables.pipeline_entries import BookIndexField

TagTextPair = tuple[str, str]


class StandardizedBlock(NamedTuple):
    tag: str
    text: str
    block: str | None


class BookTagTextPairs(BaseModel):
    llm_index: str
    index: BookIndexField
    tag_text_pairs: list[TagTextPair]
    title: str | None = None
    author: str | None = None


class BatchDetail(BaseModel):
    llm_batch_id: str
    llm_index_mapping: dict[str, BookIndexField]  # TODO: we may not need to store both IDs for contingency.
