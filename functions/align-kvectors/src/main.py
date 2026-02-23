import json
import logging
import os
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes
from statistics import mean

from shared.aws import PipelineTable, get_keys_with_prefix, get_session, upload_file
from shared.commons import get_index
from publish import publish

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


def _load_kvectors_from_s3(session, s3_prefix):
    s3_keys = sorted(
        key
        for key in get_keys_with_prefix(session, s3_prefix)
        if key.endswith(".model")
    )
    kvector_stack = []
    file_names = []
    s3_client = session.client("s3")
    for key in s3_keys:
        fd, tmp_path = tempfile.mkstemp(suffix=".model")
        os.close(fd)
        try:
            s3_client.download_file(S3_BUCKET, key, tmp_path)
            kvector_stack.append(KeyedVectors.load(tmp_path))
            file_names.append(key.split("/")[-1])
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
    return kvector_stack, file_names


def upload_kvectors_to_s3(session, index, kvectors, file_names):
    models_dir = Path("/tmp/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(kvectors)):
        save_path = models_dir / file_names[i]
        kvectors[i].save(str(save_path))
        upload_file(session, f"kvectors/{index}/aligned/{save_path.name}", save_path)


def normalized_disparity_alignment(terms, kvector, centroid):
    idx_ref = np.array([centroid.key_to_index[w] for w in terms])
    idx_kv = np.array([kvector.key_to_index[w] for w in terms])
    rotation, _ = orthogonal_procrustes(
        kvector.vectors[idx_kv], centroid.vectors[idx_ref]
    )

    kvector.vectors = kvector.vectors @ rotation

    # Calculate Disparity (Sum of Squared Errors) using Frobenius norm (ord='fro') squared
    disparity = np.linalg.norm(centroid.vectors - kvector.vectors, ord="fro") ** 2

    # Normalize by the inertia of the target set to get a 0-1 score
    total_variance = (
        np.linalg.norm(centroid.vectors - centroid.vectors.mean(axis=0), ord="fro") ** 2
    )

    if hasattr(kvector, "vectors_norm"):
        kvector.vectors_norm = None
    if hasattr(kvector, "norms"):
        kvector.norms = None
    kvector.fill_norms(force=True)

    return disparity / total_variance


def generate_centroid_kvector(terms, kvector_stack):
    vectors_by_term = {}
    for raw_vectors in kvector_stack:
        for term in terms:
            vectors_by_term.setdefault(term, []).append(raw_vectors[term])

    centroid_keyed_vector = KeyedVectors(vector_size=VECTOR_SIZE)
    centroid_keyed_vector.add_vectors(
        terms,
        np.stack([np.mean(vectors_by_term[term], axis=0) for term in terms]).astype(
            np.float32
        ),
    )
    centroid_keyed_vector.fill_norms(force=True)
    return centroid_keyed_vector


def gradient_descent_alignment(
    terms, kvector_stack, max_iterations, min_gradient=0.0001
):
    gradient = 1
    for _ in range(max_iterations):
        disparity = []
        centroid = generate_centroid_kvector(terms, kvector_stack)
        for kvector in kvector_stack:
            disparity.append(normalized_disparity_alignment(terms, kvector, centroid))
        gradient = gradient - mean(disparity)
        if gradient <= min_gradient:
            return mean(disparity)

    raise Exception("Kvectors not aligned")


def align_kvectors(index):
    session, table = _get_session_and_table()

    kvector_stack, file_names = _load_kvectors_from_s3(
        session, f"kvectors/{index}/collected/"
    )
    if not kvector_stack:
        print(f"No models found for {index}.")
        exit()

    terms = list(kvector_stack[0].key_to_index)

    for kvector in kvector_stack:
        normalized_disparity_alignment(terms, kvector, kvector_stack[0])

    mean_disparity = gradient_descent_alignment(terms, kvector_stack, 40, 0.0001)

    upload_kvectors_to_s3(session, index, kvector_stack, file_names)
    table.update_entry(index, "mean_disparity", Decimal(str(mean_disparity)))
    table.update_entry(index, "s3_aligned_data_prefix", f"kvectors/{index}/aligned/")
    table.update_entry(index, "terms", json.dumps(terms))

    # TODO: Move to new lambda
    publish(table, session, index)


if __name__ == "__main__":
    align_kvectors(get_index())
