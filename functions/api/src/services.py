import numpy as np

from shared.aws import TermTable
from schemas import OpNode, TermNode

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


def get_term_vectors(table, terms, platform_data):
    """
    Fetch and normalize vectors for 1-2 terms in a book.
    Returns normalized (n_seeds, dim) array, or None if any term is missing.
    """
    primary = table.get_entry(terms[0], platform_data, ["vectors"])
    if primary is None:
        return None
    vectors = extract_vectors(primary["vectors"])

    if len(terms) > 1:
        secondary = table.get_entry(terms[1], platform_data, ["vectors"])
        if secondary is None:
            return None
        secondary_vectors = extract_vectors(secondary["vectors"])
        vectors = np.mean(np.stack([vectors, secondary_vectors]), axis=0)

    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)



def resolve_term(term: str, table: TermTable, platform_data: str) -> np.ndarray:
    """Resolve a term to its per-seed vectors (n_seeds, dim), normalized."""
    entry = table.get_entry(term, platform_data, ["vectors"])
    if entry is None:
        raise ValueError(f"Unknown term: {term}")
    return normalize_vector_bytes(entry["vectors"])


def evaluate_tree(
    node: "TermNode | OpNode",
    table: TermTable,
    platform_data: str,
) -> np.ndarray:
    if isinstance(node, TermNode):
        return resolve_term(node.term, table, platform_data)

    left = evaluate_tree(node.args[0], table, platform_data)
    right = evaluate_tree(node.args[1], table, platform_data)
    result = left + right if node.op == "+" else left - right
    return result / np.linalg.norm(result, axis=1, keepdims=True)

