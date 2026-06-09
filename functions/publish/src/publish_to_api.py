import logging

from shared.session import get_session
from shared.tables.pipeline import get_pipeline_table
from publish_utils import publish


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    session = get_session()
    table = get_pipeline_table()

    book_ids = sorted(
        [item["platform_data"] for item in table.get_all_entries(["platform_data"])]
    )
    logger.info("Publishing %d books: %s", len(book_ids), book_ids)

    for idx in book_ids:
        publish(idx)
