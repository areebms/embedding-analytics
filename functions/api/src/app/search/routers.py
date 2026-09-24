import time

from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.core.dependencies import BooksMetadataCacheDep
from app.core.logging import add_to_log
from app.core.routing import post_route
from app.search.constants import BOOKS_WITH_EXPR
from app.search.dependencies import BooksSimilarityCacheDep, BooksTermCacheDep
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
    SemanticDriftRequest,
    SemanticDriftRequestBody,
    SemanticDriftResponse,
)
from app.search.services.describe import process_describe_query
from app.search.services.semantic_drift import (
    BooksSimilarityCache,
    BooksTermCache,
    SearchExpr,
    get_related_terms,
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
def parse_describe(
    request: ParseDescribeRequest, books_metadata_cache: BooksMetadataCacheDep
):
    add_to_log(query=request.message)
    try:
        expression, terms, substitutions = process_describe_query(
            request.message, tuple(books_metadata_cache.book_ids)
        )
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
        422: {"description": "Malformed request"},
    },
)
def semantic_drift(
    request: SemanticDriftRequestBody,
    books_term_cache: BooksTermCacheDep,
    books_similarity_cache: BooksSimilarityCacheDep,
):
    return get_semantic_drift(request, books_term_cache, books_similarity_cache)


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
            "description": "Malformed request: a repeated book_id, or the selected "
            "book named among its own targets."
        },
    },
)
def comparative_semantic_drift(
    source_book_id: int,
    request: SemanticDriftRequestBody,
    books_term_cache: BooksTermCacheDep,
    books_similarity_cache: BooksSimilarityCacheDep,
):
    return get_semantic_drift(
        request, books_term_cache, books_similarity_cache, source_book_id
    )


def get_semantic_drift(
    body: SemanticDriftRequestBody,
    books_term_cache: BooksTermCache,
    books_similarity_cache: BooksSimilarityCache,
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
        missing_terms = books_term_cache.load_book(selected_book_id).missing_terms(
            search_expr.terms
        )
        if missing_terms:
            raise ExpressionAbsentError(selected_book_id, missing_terms)

    started = time.perf_counter()
    books_term_cache.warm_cache(book_ids)
    warmed_at = time.perf_counter()

    book_ids_with_expr = books_term_cache.get_books_with_expr(book_ids, search_expr)

    min_books_with_expr = int(BOOKS_WITH_EXPR * len(book_ids))
    if len(book_ids_with_expr) < min_books_with_expr:
        raise QueryInTooFewBooksError(len(book_ids_with_expr), selected_book_id)

    ranked_at = time.perf_counter()

    expr_data, top_mean, top_std = get_related_terms(
        books_similarity_cache,
        book_ids_with_expr,
        search_expr,
        selected_book_id=selected_book_id,
    )

    terms = [term_data.term for term_data in (*top_mean, *top_std)]

    add_to_log(
        warm_ms=round((warmed_at - started) * 1000, 1),
        nearest_terms_ms=round((ranked_at - warmed_at) * 1000, 1),
        similarities_ms=round((time.perf_counter() - ranked_at) * 1000, 1),
        scored_terms=len(terms),
    )

    expr_source_ids = {
        book_data.book_id for book_data in expr_data.book_similarities
    }
    expr_book_ids = [
        book_id for book_id in book_ids if book_id.source_id in expr_source_ids
    ]
    missing_terms_by_book = books_term_cache.get_missing_terms_by_book(
        expr_book_ids, {*search_expr.terms, *terms}
    )

    book_summaries = [
        BookSummary(
            id=book_id.source_id,
            n_shared_terms=books_term_cache.get_n_shared_terms(
                book_id,
                book_ids_with_expr if selected_book_id is None else [selected_book_id],
            ),
            missing_terms=sorted(missing_terms_by_book[book_id]),
        )
        for book_id in expr_book_ids
    ]

    return SemanticDriftResponse(
        expr=expr_data,
        book_stats=book_summaries,
        top_mean=top_mean,
        top_std=top_std,
    )
