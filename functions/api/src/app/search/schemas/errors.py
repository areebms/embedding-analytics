
from typing import ClassVar, Literal

from pydantic import BaseModel

from app.search.constants import MIN_MATCHING_BOOKS


class ExpressionAbsentResponse(BaseModel):
    openapi_description: ClassVar[str] = (
        "One or more terms in the expression don't exist in the selected book, "
        "so there is nothing to compare against."
    )

    reason: Literal["expression_absent"] = "expression_absent"
    book_id: int
    terms: list[str]


class QueryInTooFewBooksResponse(BaseModel):

    openapi_description: ClassVar[str] = (
        f"Fewer than {MIN_MATCHING_BOOKS} of the requested books carry the query."
    )

    reason: Literal["query_in_too_few_books"] = "query_in_too_few_books"
    book_id: int | None = None  # the selected book, null when none was selected


class TermResolutionResponse(BaseModel):
    """candidates is empty when nothing in the vocabulary is close."""

    openapi_description: ClassVar[str] = (
        "A term in the request could not be matched to any word in the vocabulary."
    )

    reason: Literal["term_resolution"] = "term_resolution"
    message: str
    term: str
    candidates: list[str]
