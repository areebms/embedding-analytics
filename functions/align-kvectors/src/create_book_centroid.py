import logging
from decimal import Decimal

import numpy as np

from shared.aws import get_session, get_pipeline_table
from procrustes_utils import (
    rotate,
    load_corpus_centroid,
    weighted_orthogonal_procrustes,
    S3Kvectors,
    gradient_descent_alignment,
    build_centroid_kvector
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def align_to_corpus_centroid(centroid, kvector_stack, session):
    """
    If a corpus_centroid exists, solve closed-form Procrustes between this
    book's centroid and the corpus, then apply the resulting rotation to
    the centroid AND every per-seed kvector in place. After this returns,
    everything in `kvector_stack` plus `centroid` is in the corpus frame.

    Returns the corpus disparity, or None if no corpus alignment was applied
    (bootstrap case, or insufficient stable anchors).
    """
    corpus_centroid = load_corpus_centroid(session)
    if corpus_centroid is None:
        logger.info(
            "No corpus centroid found — book stored in within-book frame. "
            "Run rebuild_corpus.py to establish or update the corpus frame."
        )
        return None

    anchor_list = sorted(set(centroid.key_to_index) & set(corpus_centroid.key_to_index))

    book_matrix = np.stack([centroid[w] for w in anchor_list]).astype(np.float32)
    corpus_matrix = np.stack([corpus_centroid[w] for w in anchor_list]).astype(
        np.float32
    )
    counts = np.array(
        [corpus_centroid.get_vecattr(w, "count") or 1 for w in anchor_list],
        dtype=np.float32,
    )

    result = weighted_orthogonal_procrustes(book_matrix, corpus_matrix, counts)
    rotate(centroid, result.R)
    for kv in kvector_stack:
        rotate(kv, result.R)

    logger.info(
        "Aligned to corpus on %d anchors, disparity=%.4f",
        len(anchor_list),
        result.disparity,
    )
    return result.disparity


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

    corpus_disparity = align_to_corpus_centroid(centroid, kvector_stack, session)

    s3_kvectors.upload("aligned", centroid, kvector_stack, file_names)

    updates = {
        "mean_disparity": Decimal(str(mean_disparity)),
        "s3_prefix_models": f"kvectors/{index}/",
    }
    if corpus_disparity is not None:
        updates["corpus_disparity"] = Decimal(str(corpus_disparity))
    table.update_entries(index, updates)

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
