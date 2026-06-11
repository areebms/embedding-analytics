import json
import logging

from shared.lambda_event import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    """Per-book alignment: within-book GPA + opportunistic corpus rotation."""
    from create_book_centroid import align_kvectors

    logger.info("Align kvectors request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Align kvectors request missing index")
        return {"statusCode": 400, "body": json.dumps({"error": "index is required"})}

    result = align_kvectors(index)
    if result is None:
        return {
            "statusCode": 200,
            "body": json.dumps({"index": index, "skipped": True}),
        }

    return {"index": index}


def build_corpus_handler(event, context): # Not being Used.
    """Build corpus centroid from all aligned books."""
    from create_corpus_centroid import build_corpus_centroid
    from shared.session import get_session
    from shared.tables.pipeline import get_pipeline_table

    logger.info("Build corpus centroid request received", extra={"event": event})

    session = get_session()
    table = get_pipeline_table()

    book_ids = event.get("book_ids")
    if not book_ids:
        book_ids = sorted(
            [
                item["platform_data"]
                for item in table.get_all_entries(
                    ["platform_data", "s3_prefix_models"]
                )
                if item.get("s3_prefix_models")
            ]
        )

    if len(book_ids) < 2:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {"error": "Need at least 2 aligned books to build a corpus centroid."}
            ),
        }

    build_corpus_centroid(session, book_ids)

    return {
        "statusCode": 200,
        "body": json.dumps({"book_ids": book_ids}),
    }