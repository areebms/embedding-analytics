from fastapi import APIRouter

from app.core.dependencies import BooksMetadataCacheDep, cache
from app.list.schemas import BookResponse, TermResponse
from shared.tables.book_terms import get_book_term_table

router = APIRouter()


@router.get("/books", response_model=list[BookResponse])
@cache(expire=None)
def books(books_metadata_cache: BooksMetadataCacheDep):
    return [
        BookResponse.from_book_metadata(book_metadata)
        for book_metadata in books_metadata_cache.books_metadata.values()
        if book_metadata.published_year is not None
    ]


@router.get("/terms", response_model=list[TermResponse])
@cache(expire=None)
def terms(books_metadata_cache: BooksMetadataCacheDep):
    "Used in the dropdown"
    term_books = {}
    term_table = get_book_term_table()
    for book_id in books_metadata_cache.book_ids:
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
