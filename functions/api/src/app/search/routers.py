from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.core.logging import add_to_log
from app.core.routing import post_route
from app.search.dependencies import BooksTermCacheDep
from app.search.errors import ExpressionAbsentError, QueryInTooFewBooksError
from app.search.schemas.describe import (
    ParseDescribeRequest,
    ParseDescribeResponse,
    SubstitutionResponse,
)
from app.search.schemas.errors import (
    ExpressionAbsentResponse,
    QueryInTooFewBooksResponse,
    TermResolutionResponse,
)
from app.search.schemas.semantic_drift import (
    BookSummary,
    ExprData,
    SemanticDriftRequest,
    SemanticDriftRequestBody,
    SemanticDriftResponse,
    TermData,
)
from app.search.services.describe import process_describe_query
from app.search.services.semantic_drift import (
    MIN_MATCHING_BOOKS,
    BooksTermCache,
    SearchExpr,
    get_local_mean_similarities,
    get_nearest_terms,
)
from shared.commons import BookIndex

router = APIRouter()


@post_route(
    router,
    "/parse-describe",
    response_model=ParseDescribeResponse,
    responses={
        400: {"description": "The LLM output could not be parsed into an expression."},
        404: TermResolutionResponse,
    },
)
def parse_describe(request: ParseDescribeRequest):
    add_to_log(query=request.message)
    try:
        expression, terms, substitutions = process_describe_query(request.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ParseDescribeResponse(
        expression=expression,
        terms=terms,
        substitutions=[
            SubstitutionResponse(original=s.original, resolved=s.resolved)
            for s in substitutions
        ],
    )


@post_route(
    router,
    "/semantic-drift",
    response_model=SemanticDriftResponse,
    responses={
        404: QueryInTooFewBooksResponse,
        422: {"description": "Malformed request -- e.g. a repeated book_id."},
    },
)
def semantic_drift(request: SemanticDriftRequestBody, books_cache: BooksTermCacheDep):
    return get_semantic_drift(request, books_cache)


@post_route(
    router,
    "/semantic-drift/{source_book_id}",
    response_model=SemanticDriftResponse,
    responses={
        404: {
            "model": ExpressionAbsentResponse | QueryInTooFewBooksResponse,
            "description": (
                f"{ExpressionAbsentResponse.openapi_description} "
                f"{QueryInTooFewBooksResponse.openapi_description}"
            ),
        },
        422: {
            "description": "Malformed request: a repeated book_id, or the pinned "
            "book named among its own targets."
        },
    },
)
def comparative_semantic_drift(
    source_book_id: int,
    request: SemanticDriftRequestBody,
    books_cache: BooksTermCacheDep,
):
    return get_semantic_drift(request, books_cache, source_book_id)


def get_semantic_drift(
    body: SemanticDriftRequestBody,
    books_cache: BooksTermCache,
    source_book_id: int | None = None,
) -> SemanticDriftResponse:

    try:
        request = SemanticDriftRequest(
            **body.model_dump(), source_book_id=source_book_id
        )
    except ValidationError as exc:
        raise RequestValidationError(
            exc.errors(include_url=False), body=body.model_dump()
        ) from exc

    add_to_log(query=request.tree.model_dump())
    search_expr = SearchExpr.from_query(request.tree)
    book_ids = [BookIndex(source_id) for source_id in request.book_ids]
    selected_book_id = (
        None if request.source_book_id is None else BookIndex(request.source_book_id)
    )

    if selected_book_id is not None:
        missing_terms = books_cache.load_book(selected_book_id).missing_terms(
            search_expr.terms
        )
        if missing_terms:
            raise ExpressionAbsentError(selected_book_id, missing_terms)

    books_cache.warm_cache(book_ids)

    valid_book_ids = books_cache.get_books_with_search_query(book_ids, search_expr)

    if len(valid_book_ids) < MIN_MATCHING_BOOKS:
        raise QueryInTooFewBooksError(len(valid_book_ids), selected_book_id)

    nearest_terms = get_nearest_terms(
        books_cache, valid_book_ids, search_expr, selected_book_id=selected_book_id
    )

    plotted_terms = set(search_expr.terms) | set(nearest_terms)
    missing_terms_by_book = books_cache.get_missing_terms_by_book(book_ids, plotted_terms)

    expr_book_data = [
        get_local_mean_similarities(books_cache, expr, book_ids, selected_book_id)
        for expr in [search_expr, *map(SearchExpr.from_query, nearest_terms)]
    ]

    expr_data = ExprData(
        expr=search_expr.serialized, terms=search_expr.terms, books=expr_book_data[0]
    )

    nearest_term_data = [
        TermData(term=term, books=book_data)
        for term, book_data in zip(nearest_terms, expr_book_data[1:])
    ]

    book_summaries = [
        BookSummary(
            id=book_id.source_id, missing_terms=sorted(missing_terms_by_book[book_id])
        )
        for book_id in book_ids
    ]

    return SemanticDriftResponse(
        expr=expr_data, nearest_terms=nearest_term_data, books=book_summaries
    )
