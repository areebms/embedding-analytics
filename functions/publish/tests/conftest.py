import json
import os
import tempfile

import numpy as np
import pytest


# ── Environment ───────────────────────────────────────────────────────

os.environ.update(
    AWS_REGION="us-east-1",
    AWS_DEFAULT_REGION="us-east-1",
    AWS_ACCESS_KEY_ID="testing",
    AWS_SECRET_ACCESS_KEY="testing",
    AWS_SESSION_TOKEN="testing",
    S3_BUCKET="test-bucket",
    PIPELINE_TABLE="pipeline-test",
    BOOK_TERM_TABLE="term-test",
    TERM_CORPUS_TABLE="corpus-term-test",
)
os.environ.pop("AWS_PROFILE", None)

from shared.commons import BookIndex
from shared.s3 import upload_csv, upload_file, upload_json
from shared.tables.book_terms import get_book_term_table
from shared.tables.corpus_terms import get_corpus_term_table
from shared.tables.pipeline_entries import (
    BookMetadata,
    EntryStatus,
    PipelineEntry,
    get_pipeline_entries,
)
from shared.tests_utils import aws, bucket, entries  # noqa: F401


# ── Test data constants ───────────────────────────────────────────────

BOOK_SMITH = BookIndex(3300)
BOOK_RICARDO = BookIndex(33310)
SUBJECT = BookIndex(42)
VECTOR_DIM = 10

SMITH_TERM_COUNTS = {
    "labour": 42,
    "value": 30,
    "rent": 10,
}

SMITH_TERMS = [
    {"term": "labour", "count_": 42, "tags": {"N", "V"}},
    {"term": "value", "count_": 30, "tags": {"N"}},
    {"term": "rent", "count_": 10, "tags": {"N"}},
]

RICARDO_TERMS = [
    {"term": "labour", "count_": 20, "tags": {"N"}},
    {"term": "profit", "count_": 15, "tags": {"N"}},
]

SMITH_METADATA = {
    "author": ["Smith, Adam"],
    "title": ["The Wealth of Nations"],
}

SMITH_PUBLISHED_YEAR = 1776

SMITH_STALE_METADATA = BookMetadata(
    author="Anon", title="Untitled", published_year=SMITH_PUBLISHED_YEAR
)

SMITH_TOKEN_LEMMAS = [
    ["labour", "value", "rent"],
    ["labour", "value"],
]

SMITH_TOKEN_TAGS = [
    ["NN", "NN", "NN"],
    ["VB", "NN"],
]


# ── Helpers ───────────────────────────────────────────────────────────


def _upload_embeddings(book_id, term_counts, rng):
    terms = list(term_counts)
    with tempfile.NamedTemporaryFile(suffix=".npz") as file:
        np.savez(
            file,
            terms=np.asarray(terms, dtype=np.str_),
            vectors=rng.randn(len(terms), VECTOR_DIM).astype(np.float32),
            attr_count=np.asarray(
                [term_counts[term] for term in terms], dtype=np.int64
            ),
        )
        file.flush()
        upload_file(f"embeddings/{book_id}.npz", file.name)


def _upload_pos_data(entry, token_lemmas, token_tags):
    upload_csv(entry.s3_token_lemmas_key, token_lemmas)
    upload_csv(entry.s3_token_tags_key, token_tags)


def _seed_book(term_table, corpus_term_table, book_id, terms):
    """Write terms to BookTermTable and CorpusTermTable as if publish
    had already run for this book."""
    for entry in terms:
        term_table.update_entries(
            entry["term"],
            book_id,
            {"count_": entry["count_"], "tags": entry["tags"]},
        )
        corpus_term_table.add_book(entry["term"], book_id)


# ── Table creation ────────────────────────────────────────────────────


def _create_term_table(dynamodb):
    dynamodb.create_table(
        TableName=os.environ["BOOK_TERM_TABLE"],
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "term", "AttributeType": "S"},
            {"AttributeName": "book_id", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "term", "KeyType": "HASH"},
            {"AttributeName": "book_id", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "book_id-index",
                "KeySchema": [
                    {"AttributeName": "book_id", "KeyType": "HASH"},
                    {"AttributeName": "term", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    )


def _create_corpus_term_table(dynamodb):
    dynamodb.create_table(
        TableName=os.environ["TERM_CORPUS_TABLE"],
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "partition", "AttributeType": "S"},
            {"AttributeName": "term", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "partition", "KeyType": "HASH"},
            {"AttributeName": "term", "KeyType": "RANGE"},
        ],
    )


# ── Fixtures: infrastructure ──────────────────────────────────────────


@pytest.fixture
def moto_dynamo(aws):
    import shared.tables.book_terms as book_terms_module
    import shared.tables.corpus_terms as corpus_terms_module

    book_terms_module._book_term_table = None
    corpus_terms_module._corpus_term_table = None

    dynamodb = aws.resource("dynamodb")
    _create_term_table(dynamodb)
    _create_corpus_term_table(dynamodb)

    return aws


@pytest.fixture
def term_table(moto_dynamo):
    return get_book_term_table()


@pytest.fixture
def corpus_term_table(moto_dynamo):
    return get_corpus_term_table()


@pytest.fixture
def pipeline_entries(moto_dynamo):
    return get_pipeline_entries()


# ── Fixtures: seeded data ─────────────────────────────────────────────


@pytest.fixture
def seeded_smith(term_table, corpus_term_table):
    """Seed BookTermTable and CorpusTermTable with Smith's terms."""
    _seed_book(term_table, corpus_term_table, BOOK_SMITH, SMITH_TERMS)


@pytest.fixture
def seeded_ricardo(term_table, corpus_term_table):
    """Seed BookTermTable and CorpusTermTable with Ricardo's terms."""
    _seed_book(term_table, corpus_term_table, BOOK_RICARDO, RICARDO_TERMS)


@pytest.fixture
def book_s3_data(moto_dynamo):

    def _book_s3_data(book_id=BOOK_SMITH, term_counts=SMITH_TERM_COUNTS):
        _upload_embeddings(book_id, term_counts, np.random.RandomState(42))

        # POS data: token lemmas and tags as CSVs.
        # Sentences are rows. Each term appears with a noun tag (NN),
        # "labour" also appears as a verb (VB) to match SMITH_TERMS tags.
        entry = PipelineEntry(
            book_id=book_id,
            subject_ids={SUBJECT},
            status=EntryStatus.EMBEDDINGS_CREATED,
            metadata=SMITH_STALE_METADATA,
        )
        _upload_pos_data(entry, SMITH_TOKEN_LEMMAS, SMITH_TOKEN_TAGS)

        # Metadata JSON.
        upload_json(entry.s3_metadata_key, json.dumps(SMITH_METADATA))

        get_pipeline_entries().put_entry(entry)

        return entry

    return _book_s3_data


@pytest.fixture
def smith_s3_data(book_s3_data):
    return book_s3_data()
