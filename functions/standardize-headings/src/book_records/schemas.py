from pydantic import BaseModel

TagTextPair = tuple[str, str]


class BookRecord(BaseModel):
    custom_id: str
    index: str
    tag_text_pairs: list[TagTextPair]


class BatchDetail(BaseModel):
    batch_id: str
    custom_ids: dict[str, str]
