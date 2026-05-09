import logging

from main import tokenize
from shared.aws import extract_index


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info("Tokenize request received", extra={"event": event})
    index = extract_index(event)
    if not index:
        logger.warning("Tokenize request missing index")
        raise ValueError("index is required")

    logger.info("Starting tokenize", extra={"index": index})
    tokenize(index)
    logger.info("Tokenize completed", extra={"index": index})

    return {"index": index}
