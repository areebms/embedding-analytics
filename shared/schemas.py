from typing import Any, List

from pydantic import BaseModel, ConfigDict, field_validator


class PineconeMetadata(BaseModel):
    book_id: str
    pos: List[str]
    count: int


class PineconeEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    values: List[float]
    metadata: PineconeMetadata

    @field_validator("values", mode="before")
    @classmethod
    def validate_values(cls, val: Any):
        if len(val) != 200:
            raise ValueError("Length must equal 200")
        return val
