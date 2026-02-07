import json
import logging

from generate_kvector import generate_kvector
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _extract_sentences(event):
    if not isinstance(event, dict):
        return None

    if "sentences" in event:
        return event["sentences"]

    body = event.get("body")
    if not body:
        return None

    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(payload, dict):
        return payload.get("sentences")

    return None


def handler(event, context):
    logger.info("Generate model request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Generate model request missing index")
        return {"statusCode": 400, "body": json.dumps({"error": "index is required"})}

    sentences = _extract_sentences(event)
    if not sentences:
        logger.warning("Generate model request missing sentences")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "sentences is required"}),
        }

    generate_kvector(index, sentences)
    return {"statusCode": 200, "body": json.dumps({"status": "ok", "index": index})}
