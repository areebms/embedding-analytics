import numpy as np


def normalize_vectors(vectors, axis=-1):
    """L2-normalize along `axis`, guarding against zero-length vectors."""
    return vectors / np.maximum(
        np.linalg.norm(vectors, axis=axis, keepdims=True), 1e-12
    )
