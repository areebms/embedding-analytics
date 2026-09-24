import logging
from typing import Any

from publish import get_entries, publish_entries, resolve_subject
from shared.lambda_event import extract_field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    logger.info("Publish request received", extra={"event": event})

    book_ids = extract_field(event, "book_ids")
    subject_id = extract_field(event, "subject_id")

    if sum([bool(book_ids), bool(subject_id)]) != 1:
        raise ValueError("Exactly one of 'book_ids' or 'subject_id' must be provided")

    if subject_id:
        book_ids = resolve_subject(subject_id)

    status = publish_entries(get_entries(book_ids))

    logger.info("Publish completed", extra=status)
    return status
