import json
import logging
import tempfile
from typing import NamedTuple

import numpy as np
from botocore.exceptions import ClientError

from shared.commons import BookIndex
from shared.s3 import load_csv, upload_file
from shared.session import get_session
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)
from constants import MIN_TOKEN_SIZE, VECTOR_SIZE, MAX_BOOKS_PER_SUBJECT
from ppmi_svd import (
    Passages,
    build_vocab,
    positive_pointwise_mutual_info,
    term_cooccurrence_in_window,
    truncated_svd,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PENDING = (EntryStatus.TOKENIZED,)
ANNOUNCE_SOURCE = "embedding-analytics.create-embeddings"
ANNOUNCE_DETAIL_TYPE = "Books Embedded"


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


def upload_embedding_data(entry: PipelineEntry, embedding_data: EmbeddingData) -> None:
    with tempfile.NamedTemporaryFile(suffix=".npz") as file:
        np.savez(
            file,
            terms=np.asarray(embedding_data.terms, dtype=np.str_),
            vectors=embedding_data.vectors,
            attr_count=embedding_data.term_counts,
        )
        file.flush()
        upload_file(entry.s3_embeddings_key, file.name)


def set_status(book_id, status):
    if not get_pipeline_entries().set_status(book_id, status):
        logger.warning("%s: the status guard refused the write to %s.", book_id, status)


def resolve_subject(subject_id: str) -> list[BookIndex]:
    book_ids = get_pipeline_entries().get_indexes(
        status=EntryStatus.TOKENIZED, subject_id=subject_id
    )

    if len(book_ids) > MAX_BOOKS_PER_SUBJECT:
        logger.info(
            "%s resolved %d books; taking the first %d, the rest keep %s "
            "for the next run.",
            subject_id,
            len(book_ids),
            MAX_BOOKS_PER_SUBJECT,
            EntryStatus.TOKENIZED,
        )
        book_ids = book_ids[:MAX_BOOKS_PER_SUBJECT]

    return book_ids


def get_entries(book_ids: list[str]) -> list[PipelineEntry]:
    entries = get_pipeline_entries().get_entries(list(book_ids))

    if len(entries) != len(book_ids):
        logger.warning(
            "%d of %d book(s) have no pipeline entry.",
            len(book_ids) - len(entries),
            len(book_ids),
        )

    pending = []
    for entry in entries:
        if entry.status in PENDING:
            pending.append(entry)
        else:
            logger.info(
                "%s is at %s, not %s; skipping embedding creation.",
                entry.book_id,
                entry.status,
                ", ".join(PENDING),
            )

    return pending


def announce_embeddings_creation(book_ids: list[str]) -> None:
    response = (
        get_session()
        .client("events")
        .put_events(
            Entries=[
                {
                    "Source": ANNOUNCE_SOURCE,
                    "DetailType": ANNOUNCE_DETAIL_TYPE,
                    "Detail": json.dumps(
                        {"book_ids": book_ids, "embedded": len(book_ids)}
                    ),
                }
            ]
        )
    )

    if response["FailedEntryCount"]:
        raise RuntimeError(
            f"the bus rejected '{ANNOUNCE_DETAIL_TYPE}' for {book_ids}: "
            f"{response['Entries']}"
        )


def create_embeddings(entries: list[PipelineEntry]) -> dict:
    embedded = []

    for entry in entries:
        passages = load_passages(entry)
        if passages is None:
            logger.warning("%s has not been tokenized.", entry.book_id)
            continue

        embedding_data = get_embedding_data(passages)
        if embedding_data is None:
            logger.warning(
                "%s has too few terms for %d dimensions.", entry.book_id, VECTOR_SIZE
            )
            set_status(entry.book_id, EntryStatus.EMBEDDINGS_CREATION_FAILED)
            continue

        upload_embedding_data(entry, embedding_data)
        set_status(entry.book_id, EntryStatus.EMBEDDINGS_CREATED)

        embedded.append(str(entry.book_id))

    if embedded:
        announce_embeddings_creation(embedded)

    logger.info("%d of %d book(s) embedded.", len(embedded), len(entries))
    return {"found": len(entries), "embedded": len(embedded)}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    book_ids = get_pipeline_entries().get_indexes(EntryStatus.TOKENIZED)
    logger.info("Embedding %d books: %s", len(book_ids), book_ids)

    create_embeddings(get_entries(book_ids))
