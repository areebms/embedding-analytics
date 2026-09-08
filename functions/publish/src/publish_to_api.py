import logging

from shared.tables.pipeline_entries import get_pipeline_entries
from publish_utils import publish


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    book_ids = get_pipeline_entries().get_indexes()
    logger.info("Publishing %d books: %s", len(book_ids), book_ids)

    for idx in book_ids:
        publish(idx)
