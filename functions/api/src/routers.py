import numpy as np
from fastapi import HTTPException, APIRouter

from dependencies import cache
from services import get_confidence_intervals, normalize_vector_bytes, evaluate_tree
from describe_services import process_chat_query, TermResolutionError
from schemas import (
    SimilarityRequest,
    SimilarityResult,
    ParseChatRequest,
    ParseChatResponse,
)
from shared.aws import get_pipeline_table, TermTable, get_session

router = APIRouter()


@router.get("/books")
@cache(expire=None)
def books():
    return [
        {
            "id": int(item["platform_data"].split("-")[-1]),
            "label": f"{item['author'].split(',')[0]} ({item['published_year']})",
            "author": item["author"],
            "title": item["title"],
            "published_year": item["published_year"],
        }
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
