from typing import Annotated

from fastapi import Depends

from app.search.services.semantic_drift import BooksTermCache
from shared.tables.book_terms import BookTermTable, get_book_term_table

BookTermTableDep = Annotated[BookTermTable, Depends(get_book_term_table)]

_books_term_cache: BooksTermCache | None = None


def get_books_term_cache(table: BookTermTableDep) -> BooksTermCache:

    global _books_term_cache
    if _books_term_cache is None or _books_term_cache.table is not table:
        _books_term_cache = BooksTermCache(table)
    return _books_term_cache


BooksTermCacheDep = Annotated[BooksTermCache, Depends(get_books_term_cache)]
