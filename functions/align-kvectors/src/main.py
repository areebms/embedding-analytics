import logging
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes
from statistics import mean

from shared.aws import PipelineTable, get_session, upload_file, yield_s3_files
from shared.commons import get_index


MAX_ITERATIONS = 40
MIN_GRADIENT = 0.0001
VECTOR_SIZE = 200
S3_BUCKET = os.getenv("S3_BUCKET")

_session = None
_table = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_session_and_table():
    global _session, _table
    if _session is None:
        _session = get_session()
    if _table is None:
        _table = PipelineTable(_session)
    return _session, _table


class S3Kvectors:

    def __init__(self, session, index):
        self.session = session
        self.index = index

    def load(self):
        file_names = []
        kvector_stack = []

        for key, tmp_path in yield_s3_files(
            self.session, f"kvectors/{self.index}/collected/", ".model"
        ):
            kvector_stack.append(KeyedVectors.load(tmp_path))
            file_names.append(key.split("/")[-1])

        return file_names, kvector_stack 

    def upload(self, centroid, kvectors, file_names):
        models_dir = Path("/tmp/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(kvectors)):
            save_path = models_dir / file_names[i]
            kvectors[i].save(str(save_path))
            upload_file(
                self.session,
                f"kvectors/{self.index}/aligned/{save_path.name}",
                save_path,
            )

        save_path = models_dir / "centroid"
        centroid.save(str(save_path))
        upload_file(self.session, f"kvectors/{self.index}/centroid.model", save_path)


def normalized_disparity_alignment(terms, kvector, centroid_vectors_by_term):
    centroid_matrix = np.array(
        [centroid_vectors_by_term[w] for w in terms], dtype=np.float32
    )
    idx_kv = np.array([kvector.key_to_index[w] for w in terms])
    rotation, _ = orthogonal_procrustes(kvector.vectors[idx_kv], centroid_matrix)

    kvector.vectors = kvector.vectors @ rotation

    # Clear Cache
    if hasattr(kvector, "vectors_norm"):
        kvector.vectors_norm = None
    if hasattr(kvector, "norms"):
        kvector.norms = None
    kvector.fill_norms(force=True)

    # disparity = Frobenius Norm squared = Sum of Squared Errors
    disparity = np.linalg.norm(centroid_matrix - kvector.vectors, ord="fro") ** 2

    centroid_variance = (
        np.linalg.norm(centroid_matrix - centroid_matrix.mean(axis=0), ord="fro") ** 2
    )

    return (
        disparity / centroid_variance
    )  # equivalent to 1 - coefficient of determination


def compute_centroid_vectors(terms, kvector_stack):
    vectors_by_term = {}
    for raw_vectors in kvector_stack:
        for term in terms:
            vectors_by_term.setdefault(term, []).append(raw_vectors[term])
    return {
        term: np.mean(vectors_by_term[term], axis=0).astype(np.float32)
        for term in terms
    }


def build_centroid_kvector(terms, kvector_stack, centroid_vectors_by_term):
    centroid = KeyedVectors(vector_size=VECTOR_SIZE)
    centroid.add_vectors(
        terms,
        np.stack([centroid_vectors_by_term[term] for term in terms]),
    )
    centroid.fill_norms(force=True)

    term_disparities = np.mean(
        [
            np.sum((kvector.vectors - centroid.vectors) ** 2, axis=1)
            for kvector in kvector_stack
        ],
        axis=0,
    )

    term_variances = np.sum(
        (centroid.vectors - centroid.vectors.mean(axis=0)) ** 2, axis=1
    )

    for i, term in enumerate(terms):
        centroid.set_vecattr(term, "count", kvector_stack[0].get_vecattr(term, "count"))
        centroid.set_vecattr(term, "disparity", float(term_disparities[i]))
        centroid.set_vecattr(term, "variance", float(term_variances[i]))
        centroid.set_vecattr(
            term, "r_squared", float(1 - (term_disparities[i] / term_variances[i]))
        )

    return centroid


def gradient_descent_alignment(
    terms, kvector_stack, max_iterations, min_gradient=0.0001
):
    # Generalized Procrustes Analysis (Gower, 1975)
    prev_disparity = float("inf")
    for iteration in range(max_iterations):
        normalized_disparities = []
        centroid_vectors_by_term = compute_centroid_vectors(terms, kvector_stack)

        for kvector in kvector_stack:
            normalized_disparities.append(
                normalized_disparity_alignment(terms, kvector, centroid_vectors_by_term)
            )
        current_disparity = mean(normalized_disparities)

        if (prev_disparity - current_disparity) <= min_gradient:
            return centroid_vectors_by_term, current_disparity, iteration + 1
        prev_disparity = current_disparity

    raise Exception("Kvectors not aligned")


def perform_alignment(kvector_stack):

    terms = list(kvector_stack[0].key_to_index)

    # Align all kvectors with the first kvector for repeatability.
    initial_centroid_vectors = {term: kvector_stack[0][term] for term in terms}
    for kvector in kvector_stack:
        normalized_disparity_alignment(terms, kvector, initial_centroid_vectors)

    centroid_vectors_by_term, mean_disparity, _ = gradient_descent_alignment(
        terms, kvector_stack, MAX_ITERATIONS, MIN_GRADIENT
    )

    centroid = build_centroid_kvector(terms, kvector_stack, centroid_vectors_by_term)

    return mean_disparity, centroid


def align_kvectors(index):
    session, table = _get_session_and_table()

    s3_kvectors = S3Kvectors(session, index)

    file_names, kvector_stack = s3_kvectors.load()

    if not kvector_stack:
        print(f"No models found for {index}.")
        return

    # Sort for repeatability
    file_names, kvector_stack = map(list, zip(*sorted(zip(file_names, kvector_stack))))

    mean_disparity, centroid = perform_alignment(kvector_stack)

    s3_kvectors.upload(centroid, kvector_stack, file_names)

    table.update_entries(
        index,
        {
            "mean_disparity": Decimal(str(mean_disparity)),
            "s3_prefix_models": f"kvectors/{index}/",
        },
    )


if __name__ == "__main__":
    align_kvectors(get_index())
