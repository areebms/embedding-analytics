from pydantic import BaseModel, model_validator

from shared.commons import BookIndex


class BookResponse(BaseModel):
    id: int
    label: str
    author: str
    title: str
    published_year: int

    @model_validator(mode="before")
    @classmethod
    def from_entry(cls, data: dict) -> dict:
        if "platform_data" in data:
            data["id"] = BookIndex.parse(data.pop("platform_data")).source_id
            data["label"] = f"{data['author'].split(',')[0]} ({data['published_year']})"
        return data


class TermResponse(BaseModel):
    term: str
    books: list[str]
