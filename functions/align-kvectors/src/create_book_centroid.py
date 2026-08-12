import logging
from decimal import Decimal

import numpy as np

from shared.session import get_session
from shared.tables.pipeline import get_pipeline_table
from procrustes_utils import (
    S3Kvectors,
    gradient_descent_alignment,
    build_centroid_kvector,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def align_kvectors(index):
    session = get_session()
    table = get_pipeline_table()

    s3_kvectors = S3Kvectors(session, index)

    file_names, kvector_stack = s3_kvectors.load("collected")

    if not kvector_stack:
        print(f"No models found for {index}.")
        return

    # Sort for repeatability
    file_names, kvector_stack = map(list, zip(*sorted(zip(file_names, kvector_stack))))
    terms = list(kvector_stack[0].key_to_index)

    centroid_vectors, residuals, mean_disparity, _ = gradient_descent_alignment(
        terms, kvector_stack
    )

    counts = np.array([kvector_stack[0].get_vecattr(t, "count") for t in terms])

    centroid = build_centroid_kvector(terms, counts, residuals, centroid_vectors)

    s3_kvectors.upload("aligned", centroid, kvector_stack, file_names)

    table.update_entries(
        index,
        {
            "mean_disparity": Decimal(str(mean_disparity)),
            "s3_prefix_models": f"kvectors/{index}/",
        },
    )

    return {"index": index, "mean_disparity": mean_disparity}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    session = get_session()
    table = get_pipeline_table()

    book_ids = sorted(
        [
            item["platform_data"]
            for item in table.get_all_entries(["platform_data"])
        ]
    )
    logger.info("Aligning %d books: %s", len(book_ids), book_ids)

    for idx in book_ids:
        align_kvectors(idx)
