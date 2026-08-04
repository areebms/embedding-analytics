from collections.abc import Iterable

from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import add_to_log
from app.search.schemas.errors import (
    ExpressionAbsentResponse,
    QueryInTooFewBooksResponse,
    TermResolutionResponse,
)
from shared.commons import BookIndex


class MissingTermsError(ValueError):

    def __init__(self, terms: Iterable[str], book_id: BookIndex):
        self.terms = sorted(terms)
        self.book_id = book_id
        plural = "terms" if len(self.terms) > 1 else "term"
        super().__init__(f"Unknown {plural}: {', '.join(self.terms)}")


class NoLocalNearestTermsError(ValueError):

    def __init__(self, a_book_id: BookIndex, b_book_id: BookIndex, n: int):
        self.a_book_id = a_book_id
        self.b_book_id = b_book_id
        self.n = n
        super().__init__(
            f"too few shared local nearest terms between {a_book_id} and "
            f"{b_book_id}: {n} shared terms"
        )


class ExpressionAbsentError(ValueError):

    def __init__(self, book_id: BookIndex, terms: Iterable[str]):
        self.book_id = book_id
        self.terms = sorted(terms)
        plural = "terms" if len(self.terms) > 1 else "term"
        super().__init__(
            f"selected book {book_id.source_id} is missing {plural}: "
            f"{', '.join(self.terms)}"
        )


class QueryInTooFewBooksError(ValueError):

    def __init__(self, num_books: int, book_id: BookIndex | None = None):
        self.book_id = book_id  # the selected book, None when none was selected
        where = "the corpus" if book_id is None else f"book {book_id.source_id}"
        super().__init__(f"too few books ({num_books}) to compare against {where}")


class TermResolutionError(Exception):
    """Raised when a term cannot be resolved to any vocabulary entry."""

    def __init__(self, term: str, candidates: list[str]):
        self.term = term
        self.candidates = candidates
        if candidates:
            msg = (
                f"No matching term found for '{term}'. "
                f"Did you mean: {', '.join(candidates)}?"
            )
        else:
            msg = (
                f"No matching term found for '{term}'. No similar terms in vocabulary."
            )
        super().__init__(msg)


async def expression_absent_handler(request: Request, exc: ExpressionAbsentError):

    add_to_log(error=f"{type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=404,
        content=ExpressionAbsentResponse(
            book_id=exc.book_id.source_id, terms=exc.terms
        ).model_dump(),
    )


async def query_in_too_few_books_handler(
    request: Request, exc: QueryInTooFewBooksError
):

    add_to_log(error=f"{type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=404,
        content=QueryInTooFewBooksResponse(
            book_id=None if exc.book_id is None else exc.book_id.source_id,
        ).model_dump(),
    )


async def term_resolution_handler(request: Request, exc: TermResolutionError):

    add_to_log(error=f"{type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=404,
        content=TermResolutionResponse(
            message=str(exc), term=exc.term, candidates=exc.candidates
        ).model_dump(),
    )


async def request_validation_handler(request: Request, exc: RequestValidationError):

    add_to_log(query=exc.body, error="validation")
    return await request_validation_exception_handler(request, exc)


def add_exception_handlers(app):
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(TermResolutionError, term_resolution_handler)
    app.add_exception_handler(ExpressionAbsentError, expression_absent_handler)
    app.add_exception_handler(QueryInTooFewBooksError, query_in_too_few_books_handler)
