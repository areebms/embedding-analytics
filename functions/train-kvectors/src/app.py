import json
import logging

from main import initiate_train_vectors
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Generate model request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Generate model request missing index")
        return {"statusCode": 400, "body": json.dumps({"error": "index is required"})}

    passed, attempted = initiate_train_vectors(index)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {"index": index, "attempted": attempted, "initiated": passed}
        ),
    }
