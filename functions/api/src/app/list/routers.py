from fastapi import APIRouter

from app.core.dependencies import cache
from app.list.services import generate_book_data
from app.list.schemas import BookResponse, TermResponse
from shared.tables.book_terms import get_book_term_table

router = APIRouter()


@router.get("/books", response_model=list[BookResponse])
@cache(expire=None)
def books():
    "Used in the Legend"
    data = []
    for item in generate_book_data(
        ["platform_data", "author", "published_year", "title"]
    ):
        data.append(BookResponse.model_validate(item))
    return data


@router.get("/terms", response_model=list[TermResponse])
@cache(expire=None)
def terms():
    "Used in the dropdown"
    book_ids = []
    for item in generate_book_data(["platform_data"]):
        book_ids.append(item["platform_data"])

    term_books = {}
    term_table = get_book_term_table()
    for book_id in book_ids:
        for item in term_table.get_entries(book_id, fields=["term", "tags"]):
            if item.get("tags") == {"R"}:
                continue

            if item["term"] not in term_books:
                term_books[item["term"]] = []

            term_books[item["term"]].append(book_id)

    return [
        {"term": term, "books": books}
        for term, books in term_books.items()
        if len(books) > 1
    ]
