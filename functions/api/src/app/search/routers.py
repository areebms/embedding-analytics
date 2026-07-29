from fastapi import HTTPException, APIRouter

from app.core.logging import add_to_log
from app.search.schemas.describe import ParseDescribeRequest, ParseDescribeResponse
from app.search.services.describe import process_describe_query

router = APIRouter()


@router.post("/parse-describe", response_model=ParseDescribeResponse)
def parse_describe(req: ParseDescribeRequest):
    add_to_log(query=req.message)
    # TermResolutionError propagates to its global handler (a 404 finding with
    # "did you mean" candidates); only a genuine parse failure is a 400 here.
    try:
        expression, terms, substitutions = process_describe_query(req.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ParseDescribeResponse(
        expression=expression,
        terms=terms,
        substitutions=[
            {"original": s.original, "resolved": s.resolved} for s in substitutions
        ],
    )
