import logging
from typing import Any

from create_embeddings import create_embeddings
from shared.commons import BookIndex
from shared.lambda_event import extract_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    logger.info("Create embeddings request received", extra={"event": event})

    index = extract_index(event)
    if not index:
        logger.warning("Create embeddings request missing index")
        raise ValueError("index is required")

    book_id = BookIndex.parse(index)
    status = create_embeddings(book_id) or {"book_id": book_id, "skipped": True}

    logger.info("Create embeddings completed", extra=status)
    return status
