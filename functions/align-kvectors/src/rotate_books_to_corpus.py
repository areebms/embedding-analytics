import logging
from decimal import Decimal

import numpy as np

from procrustes_utils import (
    S3Kvectors,
    load_book_centroid,
    load_corpus_centroid,
    rotate,
    weighted_orthogonal_procrustes,
)
from shared.aws import get_pipeline_table, get_session

logger = logging.getLogger(__name__)


def rotate_book_to_corpus(session, table, idx):
    book_centroid = load_book_centroid(session, idx)
    corpus_centroid = load_corpus_centroid(session)

    shared_terms = sorted(
        set(book_centroid.key_to_index) & set(corpus_centroid.key_to_index)
    )
    book_vectors = np.stack([book_centroid[w] for w in shared_terms]).astype(np.float32)
    corpus_vectors = np.stack([corpus_centroid[w] for w in shared_terms]).astype(
        np.float32
    )

    corpus_counts = np.array(
        [corpus_centroid.get_vecattr(w, "count") or 1 for w in shared_terms],
        dtype=np.float32,
    )

    result = weighted_orthogonal_procrustes(
        source=book_vectors,
        target=corpus_vectors,
        counts=corpus_counts
    )

    # Per-seed aligned/ files are in the previous frame (within-book or
    # old corpus). The rotation takes them to the new corpus frame.
    s3_kvectors = S3Kvectors(session, idx)
    file_names, kvector_stack = s3_kvectors.load(subdir="aligned")
    if not kvector_stack:
        logger.warning("No aligned per-seed kvectors for %s, skipping rotate.", idx)
        return

    rotate(book_centroid, result.R)
    for kvector in kvector_stack:
        rotate(kvector, result.R)

    s3_kvectors.upload(book_centroid, kvector_stack, file_names)

    table.update_entries(
        idx,
        {
            "corpus_disparity": Decimal(str(result.disparity)),
        },
    )
    logger.info("  %s: corpus_disparity=%.4f", idx, result.disparity)


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
    logger.info("Rotating %d books to corpus frame: %s", len(book_ids), book_ids)

    for idx in book_ids:
        rotate_book_to_corpus(session, table, idx)

    logger.info(
        "Corpus rotation complete. TermTable rows are now stale — "
        "re-run the publish step for every book to refresh per-seed vectors."
    )
