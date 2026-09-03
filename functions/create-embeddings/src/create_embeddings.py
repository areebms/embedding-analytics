import logging
import tempfile
from typing import NamedTuple

import numpy as np
from botocore.exceptions import ClientError

from shared.commons import BookIndex
from shared.s3 import load_csv, upload_file
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)
from constants import MIN_TOKEN_SIZE, VECTOR_SIZE
from ppmi_svd import (
    Passages,
    build_vocab,
    positive_pointwise_mutual_info,
    term_cooccurrence_in_window,
    truncated_svd,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingData(NamedTuple):
    terms: list[str]
    vectors: np.ndarray
    term_counts: np.ndarray


def load_passages(entry: PipelineEntry) -> list[list[str]] | None:
    try:
        passages = list(load_csv(entry.s3_token_lemmas_key))
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchKey":
            raise
        return None

    return [
        [token for token in passage if token.isalpha() and len(token) >= MIN_TOKEN_SIZE]
        for passage in passages
    ]


def get_embedding_data(passages: Passages) -> EmbeddingData | None:
    terms, term_indexes, counts = build_vocab(passages)
    if len(terms) <= VECTOR_SIZE:
        return None

    term_counts = np.array([counts[term] for term in terms])
    book_counts = term_cooccurrence_in_window(passages, term_indexes)

    vectors = truncated_svd(positive_pointwise_mutual_info(book_counts))

    does_cooccur = (
        np.asarray(book_counts.sum(axis=1)).ravel() - book_counts.diagonal() > 0
    )  # term exists off diagonal ie co-occurs with other terms.
    if not does_cooccur.all():
        logger.info(
            "dropping %d terms that co-occur with nothing",
            int((~does_cooccur).sum()),
        )
        terms = [term for term, keeping in zip(terms, does_cooccur) if keeping]
        term_counts = term_counts[does_cooccur]
        vectors = vectors[does_cooccur]

    return EmbeddingData(
        terms,
        np.asarray(vectors, dtype=np.float32),
        np.asarray(term_counts, dtype=np.int64),
    )


def upload_embedding_data(book_id: BookIndex, embedding_data: EmbeddingData) -> None:
    with tempfile.NamedTemporaryFile(suffix=".npz") as file:
        np.savez(
            file,
            terms=np.asarray(embedding_data.terms, dtype=np.str_),
            vectors=embedding_data.vectors,
            attr_count=embedding_data.term_counts,
        )
        file.flush()
        upload_file(f"embeddings/{book_id}.npz", file.name)


def create_embeddings(book_id: BookIndex) -> dict[str, BookIndex] | None:
    pipeline_entries = get_pipeline_entries()

    try:
        entry = pipeline_entries.get_entry(book_id, ["book_id", "status"])
    except LookupError:
        logger.warning("%s has no pipeline entry.", book_id)
        return None

    if entry.status != EntryStatus.TOKENIZED:
        logger.warning(
            "%s is %s, not %s.", book_id, entry.status, EntryStatus.TOKENIZED
        )
        return None

    passages = load_passages(entry)
    if passages is None:
        logger.warning("%s has not been tokenized.", book_id)
        return None

    embedding_data = get_embedding_data(passages)
    if embedding_data is None:
        logger.warning("%s has too few terms for %d dimensions.", book_id, VECTOR_SIZE)
        return None

    upload_embedding_data(book_id, embedding_data)

    if not pipeline_entries.set_status(book_id, EntryStatus.EMBEDDED):
        logger.warning("%s was not advanced to %s.", book_id, EntryStatus.EMBEDDED)

    return {"book_id": book_id}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    book_ids = get_pipeline_entries().get_indexes(EntryStatus.TOKENIZED)
    logger.info("Embedding %d books: %s", len(book_ids), book_ids)

    for book_id in book_ids:
        create_embeddings(book_id)
