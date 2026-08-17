import logging

from scrape import scrape_book_content, scrape_book_metadata
from shared.commons import BookIndex
from shared.lambda_event import extract_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    """Run one scrape stage. The state machine invokes this once per stage."""
    logger.info("Scrape request received", extra={"event": event})

    source_id = extract_index(event)
    if not source_id:
        logger.warning("Scrape request missing index")
        raise ValueError("index is required")

    stage = (event or {}).get("stage")
    if stage not in ["content", "metadata"]:
        logger.warning("Scrape request has no runnable stage", extra={"stage": stage})
        raise ValueError("stage must be one of ['content', 'metadata']")

    index = BookIndex.parse(source_id)
    logger.info("Starting scrape", extra={"index": index, "stage": stage})

    if stage == "metadata":
        status = scrape_book_metadata(index)
    elif stage == "content":
        status = scrape_book_content(index)

    logger.info(
        "Scrape completed", extra={"index": index, "stage": stage, "status": status}
    )

    return {"index": index, "status": status}
