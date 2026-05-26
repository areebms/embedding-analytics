import numpy as np
from fastapi import HTTPException, APIRouter

from dependencies import cache
from services import get_confidence_intervals, normalize_vector_bytes, evaluate_tree
from describe_services import process_chat_query, TermResolutionError
from diachronic_services import (
    get_reference_centroid,
    compute_diachronic_similarity,
)
from schemas import (
    SimilarityRequest,
    SimilarityResult,
    ParseChatRequest,
    ParseChatResponse,
    DiachronicRequest,
    DiachronicResult,
    BookResponse
)
from shared.aws import get_pipeline_table, TermTable, get_session

router = APIRouter()


def _aligned_book_ids() -> set[int]:
    """Gutenberg IDs of books that have been corpus-aligned."""
    return {
        int(item["platform_data"].split("-")[-1])
        for item in get_pipeline_table().get_all_entries(
            ["platform_data", "s3_prefix_models"]
        )
        if "s3_prefix_models" in item
    }


@router.get("/books", response_model=list[BookResponse])
@cache(expire=None)
def books():
    return [
        BookResponse.model_validate(item)
        for item in get_pipeline_table().get_all_entries(
            [
                "platform_data",
                "author",
                "published_year",
                "title",
                "s3_prefix_models",
            ]
        )
        if "s3_prefix_models" in item
    ]



@router.get("/terms")
@cache(expire=None)
def terms():

    book_ids = [
        item["platform_data"]
        for item in get_pipeline_table().get_all_entries(
            ["platform_data", "s3_prefix_models"]
        )
        if "s3_prefix_models" in item
    ]

    term_books = {}
    term_table = TermTable(get_session())
    for book_id in book_ids:
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


@router.post("/similarity/{book_id}", response_model=list[SimilarityResult])
@cache(expire=None)
def similarity(book_id: str, query: SimilarityRequest):
    platform_data = f"gutenberg-{book_id}"
    table = TermTable(get_session())

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


@router.post(
    "/diachronic-similarity/{book_id}",
    response_model=DiachronicResult,
)
@cache(expire=300)
async def diachronic_similarity(book_id: int, query: DiachronicRequest):
    aligned_ids = _aligned_book_ids()
    invalid = [rid for rid in query.ref_book_ids if rid not in aligned_ids]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Reference books not corpus-aligned: {invalid}",
        )
    if book_id not in aligned_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Book {book_id} is not corpus-aligned",
        )

    table = TermTable(get_session())

    try:
        reference = await get_reference_centroid(
            query.tree, query.ref_book_ids, table
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = compute_diachronic_similarity(query.tree, book_id, reference, table)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Expression cannot be evaluated for book {book_id} "
                f"(missing terms)"
            ),
        )

    return result


@router.post("/parse-describe", response_model=ParseChatResponse)
def parse_chat(req: ParseChatRequest):
    try:
        expression, terms, substitutions = process_chat_query(req.message)
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
    return ParseChatResponse(
        expression=expression,
        terms=terms,
        substitutions=[
            {"original": s.original, "resolved": s.resolved} for s in substitutions
        ],
    )
