import logging

from llm_request.make_request import get_entries, submit
from llm_response.standardize import standardize_from_batch
from shared.lambda_event import extract_field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):

    logger.info("Standardize request received", extra={"event": event})

    batch_id = extract_field(event, "batch_id")
    book_ids = extract_field(event, "book_ids")

    # The field carries the work, so it also picks the stage: a book list submits, a batch
    # id collects. Truthiness rather than `is None` so that an empty `book_ids` -- a
    # caller with no work to hand over -- is refused here rather than falling through to
    # collect a batch that was never opened.
    if bool(batch_id) == bool(book_ids):
        raise ValueError("Exactly one of 'batch_id' or 'book_ids' must be provided.")

    if book_ids:
        status = submit(get_entries(book_ids))
    else:
        status = standardize_from_batch(batch_id)

    logger.info("Standardize completed", extra=status)
    return status
