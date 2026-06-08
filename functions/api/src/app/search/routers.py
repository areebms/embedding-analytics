import numpy as np
from fastapi import HTTPException, APIRouter

from app.core.dependencies import cache
from app.search.schemas.search_expr import SimilarityRequest, SimilarityResult
from app.search.schemas.describe import ParseDescribeRequest, ParseDescribeResponse
from app.search.services.search_expr import (
    normalize_vector_bytes,
    get_confidence_intervals,
    evaluate_tree,
)
from app.search.services.describe import process_describe_query, TermResolutionError
from shared.tables.book_terms import get_book_term_table

router = APIRouter()


@router.post("/similarity/{book_id}", response_model=list[SimilarityResult])
@cache(expire=None)
def search_expr(book_id: str, query: SimilarityRequest):
    platform_data = f"gutenberg-{book_id}"
    table = get_book_term_table()

    try:
        query_vectors = evaluate_tree(query.tree, table, platform_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    table_data = []
    for item_data in table.get_entries(
        platform_data, fields=["term", "count_", "tags", "vectors"]
    ):
        if item_data["tags"] == {"R"}:
            continue

        cosine_similarity, ci_half = get_confidence_intervals(
            query_vectors, normalize_vector_bytes(item_data["vectors"])
        )

        table_data.append(
            SimilarityResult(
                term=item_data["term"],
                pos=item_data["tags"],
                count=int(item_data["count_"]),
                similarity=cosine_similarity,
                similarity_ci=(
                    cosine_similarity - ci_half,
                    cosine_similarity + ci_half,
                ),
            )
        )
    return table_data


@router.post("/parse-describe", response_model=ParseDescribeResponse)
def parse_describe(req: ParseDescribeRequest):
    try:
        expression, terms, substitutions = process_describe_query(req.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TermResolutionError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(e),
                "term": e.term,
                "candidates": e.candidates,
            },
        )
    return ParseDescribeResponse(
        expression=expression,
        terms=terms,
        substitutions=[
            {"original": s.original, "resolved": s.resolved} for s in substitutions
        ],
    )
