import numpy as np

from app.search.constants import T_CRIT_95
from app.search.schemas.search_expr import OpNode, TermNode
from shared.aws import TermTable, get_session


def get_term_table():
    return TermTable(get_session())


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


def get_confidence_intervals(query_vectors, item_vectors):
    cosine_similarities = np.sum(query_vectors * item_vectors, axis=1)
    n = len(cosine_similarities)
    cosine_similarity = float(np.mean(cosine_similarities))
    if n > 1:
        t_crit = T_CRIT_95[n - 1] if n - 1 < len(T_CRIT_95) else 1.96
        ci_half = float(t_crit * np.std(cosine_similarities, ddof=1) / np.sqrt(n))
    else:
        ci_half = 0.0
    return cosine_similarity, ci_half


def get_term_vectors(term: str, table: TermTable, platform_data: str) -> np.ndarray:
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
        return get_term_vectors(node.term, table, platform_data)

    left = evaluate_tree(node.args[0], table, platform_data)
    right = evaluate_tree(node.args[1], table, platform_data)
    result = left + right if node.op == "+" else left - right
    return result / np.linalg.norm(result, axis=1, keepdims=True)
