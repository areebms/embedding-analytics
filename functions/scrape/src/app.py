import json
import logging

from scrape import scrape
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info("Scrape request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Scrape request missing index")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "index is required"}),
        }

    logger.info("Starting scrape", extra={"index": index})
    scrape(index)
    logger.info("Scrape completed", extra={"index": index})

    return {
        "statusCode": 200,
        "body": json.dumps({"status": "ok", "index": index}),
    }
