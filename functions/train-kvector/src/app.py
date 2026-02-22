import json
import logging

from main import train_and_upload_kvector
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Generate model request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Generate model request missing index")
        return {"statusCode": 400, "body": json.dumps({"error": "index is required"})}
    
    seed = event.get("seed")

    if seed is None:
        logger.warning("Generate model request missing seed")
        return {"statusCode": 400, "body": json.dumps({"error": "seed is required"})}

    train_and_upload_kvector(index, seed)
    return {"statusCode": 200, "body": json.dumps({"status": "ok", "index": index, "seed": seed})}
