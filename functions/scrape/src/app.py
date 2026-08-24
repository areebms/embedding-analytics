import logging

from scrape import scrape_book_content, scrape_book_metadata, scrape_subject_book_list
from shared.commons import BookIndex
from shared.lambda_event import extract_field, extract_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STAGES = ("CONTENT", "SUBJECT", "METADATA")


def handler(event, context):
    """Run one scrape stage. The state machines invoke this once per stage."""
    logger.info("Scrape request received", extra={"event": event})

    stage = (event or {}).get("stage")
    if stage not in STAGES:
        logger.warning("Scrape request has no runnable stage", extra={"stage": stage})
        raise ValueError(f"stage must be one of {list(STAGES)}")

    if stage == "SUBJECT":
        subject = extract_field(event, "subject")
        if not subject:
            raise ValueError("subject is required")

        logger.info("Starting subject listing", extra={"subject": subject})
        result = scrape_subject_book_list(subject)
        logger.info(
            "Subject listing completed",
            extra={
                "subject": subject,
                "books_found": result["found"],
                "entries_created": result["created"],
            },
        )
        return result

    source_id = extract_index(event)
    if not source_id:
        logger.warning("Scrape request missing index")
        raise ValueError("index is required")

    book_id = BookIndex.parse(source_id)
    logger.info("Starting scrape", extra={"book_id": book_id, "stage": stage})

    if stage == "METADATA":
        status = scrape_book_metadata(book_id)
    elif stage == "CONTENT":
        status = scrape_book_content(book_id)

    logger.info(
        "Scrape completed", extra={"book_id": book_id, "stage": stage, "status": status}
    )

    return {"stage": stage, "book_id": book_id, "status": status}
