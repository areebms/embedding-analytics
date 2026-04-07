import numpy as np

T_CRIT_95 = [
    0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
]


def extract_vectors(buffers):
    return np.stack(
        [
            np.frombuffer(bytes(buffer), dtype=np.float16).astype(np.float64)
            for buffer in buffers
        ]
    )


def normalize_vector_bytes(buffers):
    vectors = extract_vectors(buffers)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def get_confidence_intervals(primary_vectors, item_vectors):
    cosine_similarities = np.sum(primary_vectors * item_vectors, axis=1)
    n = len(cosine_similarities)
    cosine_similarity = float(np.mean(cosine_similarities))
    if n > 1:
        t_crit = T_CRIT_95[n - 1] if n - 1 < len(T_CRIT_95) else 1.96
        ci_half = float(t_crit * np.std(cosine_similarities, ddof=1) / np.sqrt(n))
    else:
        ci_half = 0.0
    return cosine_similarity, ci_half