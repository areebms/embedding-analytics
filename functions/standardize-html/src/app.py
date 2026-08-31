import logging

from llm_request.make_request import get_entries, resolve_subject, submit
from llm_response.standardize import standardize_from_batch
from shared.lambda_event import extract_field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):

    logger.info("Standardize request received", extra={"event": event})

    batch_id = extract_field(event, "batch_id")
    book_ids = extract_field(event, "book_ids")
    subject_id = extract_field(event, "subject_id")

    if sum([bool(batch_id), bool(book_ids), bool(subject_id)]) != 1:
        raise ValueError(
            "Exactly one of 'batch_id', 'book_ids' or 'subject_id' must be provided"
        )

    if batch_id:
        status = standardize_from_batch(batch_id)
    else:
        if subject_id:
            book_ids = resolve_subject(subject_id)
        status = submit(get_entries(book_ids))

    logger.info("Standardize completed", extra=status)
    return status
