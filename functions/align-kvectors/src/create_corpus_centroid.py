import logging
from collections import Counter

import numpy as np

from shared.aws import get_pipeline_table, get_session
from procrustes_utils import (
    CORPUS_CENTROID_KEY,
    load_book_centroid,
    gradient_descent_alignment,
    build_centroid_kvector,
    upload_kvector
)

logger = logging.getLogger(__name__)


def get_book_counts(kvector_stack) -> Counter:
    counts: Counter = Counter()
    for centroid in kvector_stack:
        counts.update(centroid.key_to_index.keys())
    return counts


def build_corpus_centroid(session, book_ids):
    kvector_stack = [load_book_centroid(session, idx) for idx in book_ids]
    terms, book_counts = zip(*get_book_counts(kvector_stack).most_common())
    terms = list(terms)
    book_counts = np.array([0 if count == 1 else 1 for count in book_counts]) # TODO: Revisit when corpus increases

    centroid_vectors, residuals, mean_disparity, iterations = (
        gradient_descent_alignment(terms, kvector_stack, counts=book_counts)
    )
    logger.info(
        "Corpus GPA complete: mean_disparity=%.4f after %d iterations",
        mean_disparity,
        iterations,
    )
 
    centroid_vectors /= np.linalg.norm(centroid_vectors, axis=1, keepdims=True)

    corpus_centroid = build_centroid_kvector(
        terms, book_counts, residuals, centroid_vectors
    )
    upload_kvector(session, corpus_centroid, CORPUS_CENTROID_KEY)
    logger.info("Uploaded corpus centroid")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    session = get_session()
    table = get_pipeline_table()

    book_ids = sorted(
        [
            item["platform_data"]
            for item in table.get_all_entries(["platform_data", "s3_prefix_models"])
            if item.get("s3_prefix_models")
        ]
    )
    logger.info("Building corpus centroid from %d books: %s", len(book_ids), book_ids)

    if len(book_ids) < 2:
        raise SystemExit("Need at least 2 aligned books to build a corpus centroid.")

    build_corpus_centroid(session, book_ids)
