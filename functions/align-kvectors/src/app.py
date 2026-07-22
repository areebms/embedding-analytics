import json
import logging

from shared.lambda_event import extract_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    """Per-book alignment: within-book GPA."""
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
