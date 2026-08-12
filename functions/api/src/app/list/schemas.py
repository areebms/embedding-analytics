from pydantic import BaseModel

from app.core.services import BookMetadata


class BookResponse(BaseModel):
    id: int
    label: str
    author: str
    title: str
    published_year: int

    @classmethod
    def from_book_metadata(cls, data: BookMetadata) -> "BookResponse":
        return cls(
            id=data.book_id.source_id,
            label=f"{data.author.split(',')[0]} ({data.published_year})",
            author=data.author,
            title=data.title,
            published_year=data.published_year,
        )


class TermResponse(BaseModel):
    term: str
    books: list[int]
