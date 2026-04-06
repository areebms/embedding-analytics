import json
import logging

from main import publish
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Publish request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Publish request missing index")
        return {"statusCode": 400, "body": json.dumps({"error": "index is required"})}

    publish(index)

    return {
        "statusCode": 200,
        "body": json.dumps({"index": index}),
    }
