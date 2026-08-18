import logging

from submit import submit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    # No index: this is a corpus sweep over every book at SCRAPED, not a per-book
    # stage, so it takes no input beyond the event itself.
    logger.info("Standardize headings request received", extra={"event": event})
    summary = submit()
    logger.info("Standardize headings completed", extra=summary)
    return summary
