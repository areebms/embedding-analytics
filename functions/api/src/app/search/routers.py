import numpy as np
from fastapi import HTTPException, APIRouter

from app.core.dependencies import cache
from app.search.schemas.search_expr import (
    SimilarityRequest,
    SimilarityResult,
    SimilarityResponse,
    ConfidenceRequest,
    ConfidenceResult,
)
from app.search.schemas.describe import ParseDescribeRequest, ParseDescribeResponse
from app.search.services.describe import process_describe_query, TermResolutionError
from app.search.services.search_expr import (
    normalize_vector_bytes,
    get_confidence_intervals,
    evaluate_tree,
)
from shared.tables.book_terms import get_book_term_table
from shared.tables.vectors import get_pinecone_table

router = APIRouter()


@router.post("/similar-terms/quick/{book_id}", response_model=SimilarityResponse)
@cache(expire=None)
def search_expr(book_id: str, query: SimilarityRequest):
    platform_data = f"gutenberg-{book_id}"
    table = get_book_term_table()

    try:
        query_vectors = evaluate_tree(query.tree, table, platform_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    query_centroid = query_vectors.mean(axis=0) / np.linalg.norm(query_vectors)
    if query_centroid is None:
        return SimilarityResponse(results=[], query_vectors=query_vectors.tolist())

    matches = get_pinecone_table().query_book(
        platform_data, query_centroid, top_k=query.top_k
    )

    results = []
    for match in matches:
        results.append(
            SimilarityResult(
                term=match["term"], count=int(match["count"]), similarity=match["score"]
            )
        )

    results.sort(key=lambda r: r.similarity, reverse=True)
    return SimilarityResponse(results=results, query_vectors=query_vectors.tolist())


@router.post(
    "/similar-terms/detailed/{book_id}/", response_model=list[ConfidenceResult]
)
@cache(expire=None)
def search_confidence(book_id: str, query: ConfidenceRequest):
    platform_data = f"gutenberg-{book_id}"
    table = get_book_term_table()

    query_vectors = np.asarray(query.query_vectors, dtype=np.float64)

    entries = table.batch_get_entries(
        query.terms, platform_data, fields=["term", "vectors"]
    )
    vectors_by_term = {entry["term"]: entry["vectors"] for entry in entries}

    results = []
    for term in query.terms:
        vectors = vectors_by_term.get(term)
        if vectors is None:
            continue
        cosine_similarity, ci_half = get_confidence_intervals(
            query_vectors, normalize_vector_bytes(vectors)
        )
        results.append(
            ConfidenceResult(
                term=term,
                similarity=cosine_similarity,
                similarity_ci=(
                    cosine_similarity - ci_half,
                    cosine_similarity + ci_half,
                ),
            )
        )
    return results


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
