from fastapi import APIRouter, HTTPException

from app.core.logging import add_to_log
from app.core.routing import post_route
from app.search.schemas.describe import (
    ParseDescribeRequest,
    ParseDescribeResponse,
    SubstitutionResponse,
)
from app.search.schemas.errors import TermResolutionResponse
from app.search.services.describe import process_describe_query

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
