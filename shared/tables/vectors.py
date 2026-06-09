import os
import logging
from typing import Iterable

from shared.schemas import PineconeEntry

import numpy as np
from pinecone import Pinecone
from pinecone.errors.exceptions import NotFoundError, PineconeValueError

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "embedding-analytics")
PINECONE_BATCH_SIZE_LIMIT = 100
NAMESPACE = "__default__"
DELIMITER = "::"

_pinecone_table = None


def get_pinecone_table():
    global _pinecone_table
    if _pinecone_table is None:
        _pinecone_table = PineconeTable()
    return _pinecone_table


def get_vector_id(book_id: str, term: str) -> str:
    return f"{book_id}{DELIMITER}{term}"


def _split_vector_id(vector_id: str) -> tuple[str, str]:
    book_id, _, term = vector_id.partition(DELIMITER)
    return book_id, term


logger = logging.getLogger(__name__)


class PineconeTable:

    def __init__(self):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")

        self.index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)

    def batch_upsert_unthrottled(self, batch):
        try:
            self.index.upsert(vectors=batch, namespace=NAMESPACE)
        except PineconeValueError as e:
            logger.warning(str(e))

    def batch_upsert(self, items: Iterable[dict]):
        batch = []
        for item in items:
            batch.append(
                PineconeEntry(
                    id=get_vector_id(item["book_id"], item["term"]),
                    values=item["vector"].astype(np.float32).tolist(),
                    metadata={
                        "book_id": item["book_id"],
                        "pos": item["pos"],
                        "count": item["count"],
                    },
                ).model_dump()
            )

            if len(batch) >= PINECONE_BATCH_SIZE_LIMIT:
                self.batch_upsert_unthrottled(batch)
                batch = []

        if batch:
            self.batch_upsert_unthrottled(batch)

    def delete_book(self, book_id: str):
        """Remove all vectors for a book"""

        try:
            self.index.delete(
                filter={"book_id": {"$eq": book_id}},
                namespace=NAMESPACE,
            )
        except NotFoundError:
            pass

    def delete_book_terms(self, book_id: str, terms: Iterable[str]):
        """Remove all vectors for a book"""

        try:
            self.index.delete(
                ids=[get_vector_id(book_id, term) for term in terms],
                namespace=NAMESPACE,
            )
        except NotFoundError:
            pass

    def query_book(
        self,
        book_id: str,
        query_vector: np.ndarray,
        top_k: int = 100,
    ) -> list[dict]:

        matches = self.index.query(
            vector=query_vector.astype(np.float32).tolist(),
            top_k=top_k,
            namespace=NAMESPACE,
            filter={"book_id": {"$eq": book_id}},
            include_metadata=True,
        ).get("matches", [])

        return [
            {
                "term": _split_vector_id(match["id"])[1],
                "book_id": book_id,
                "score": float(match["score"]),
                "pos": match["metadata"]["pos"],
                "count": int(match["metadata"]["count"]),
            }
            for match in matches
        ]
