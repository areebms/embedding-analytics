import logging

from main import scrape
from shared.lambda_event import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Scrape request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Scrape request missing index")
        raise ValueError("index is required")

    logger.info("Starting scrape", extra={"index": index})
    scrape(index)
    logger.info("Scrape completed", extra={"index": index})

    return {"index": index}
