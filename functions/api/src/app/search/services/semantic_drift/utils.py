from __future__ import annotations

from typing import NamedTuple

import numpy as np

from app.search.constants import BARE_TERM_PATTERN
from app.search.schemas.semantic_drift import OpNode, TermNode
from app.search.services.utils import extract_terms, serialize_expression


def normalize_vectors(vectors, axis=-1):
    """L2-normalize along `axis`, guarding against zero-length vectors."""
    return vectors / np.maximum(
        np.linalg.norm(vectors, axis=axis, keepdims=True), 1e-12
    )


def center_vectors(vectors, axis=-1):
    return vectors - vectors.mean(axis=axis, keepdims=True)


class SearchExpr(NamedTuple):

    tree: TermNode | OpNode
    terms: list[str]
    serialized: str

    @classmethod
    def from_query(cls, query: TermNode | OpNode | str) -> SearchExpr:
        if isinstance(query, str):

            if not BARE_TERM_PATTERN.fullmatch(query):
                raise ValueError(f"expected a single bare term, got {query!r}")
            query = TermNode(term=query)
        return cls(
            query,
            extract_terms(query),
            serialize_expression(query, strip_outer=True),
        )
