import csv
import io
import json
import os
import tempfile

import numpy as np
import pytest
from gensim.models import KeyedVectors
from moto import mock_aws
from unittest.mock import MagicMock


# ── Environment ───────────────────────────────────────────────────────

os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("PIPELINE_TABLE", "pipeline-test")
os.environ.setdefault("BOOK_TERM_TABLE", "term-test")
os.environ.setdefault("TERM_CORPUS_TABLE", "corpus-term-test")

from shared.tables.book_terms import get_book_term_table
from shared.tables.corpus_terms import get_corpus_term_table
from shared.tables.pipeline import get_pipeline_table
from shared.tables.pipeline_entries import EntryStatus, metadata_key
from shared.session import get_session


# ── Test data constants ───────────────────────────────────────────────

BOOK_SMITH = "gutenberg-3300"
BOOK_RICARDO = "gutenberg-33310"
VECTOR_DIM = 10
NUM_SEEDS = 2

SMITH_TERM_ATTRS = {
    "labour": {"count": 42, "variance": 0.1, "disparity": 0.05, "r_squared": 0.95},
    "value": {"count": 30, "variance": 0.15, "disparity": 0.08, "r_squared": 0.90},
    "rent": {"count": 10, "variance": 0.2, "disparity": 0.1, "r_squared": 0.85},
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


# ── Helpers ───────────────────────────────────────────────────────────


def _create_keyed_vectors(terms_attrs, vector_dim, rng):
    """Build a gensim KeyedVectors with deterministic random vectors
    and the given term attributes (count, variance, disparity, r_squared)."""
    kv = KeyedVectors(vector_size=vector_dim)
    terms = list(terms_attrs.keys())
    vectors = rng.randn(len(terms), vector_dim).astype(np.float32)
    kv.add_vectors(terms, vectors)
    for term, attrs in terms_attrs.items():
        for attr_name, attr_value in attrs.items():
            kv.set_vecattr(term, attr_name, attr_value)
    return kv


def _save_keyed_vectors_to_s3(s3_resource, bucket, s3_key, kv):
    """Save a KeyedVectors to a temp file, upload to moto S3."""
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as tmp:
        kv.save(tmp.name)
        tmp_path = tmp.name
    try:
        s3_resource.Object(bucket, s3_key).upload_file(tmp_path)
    finally:
        os.unlink(tmp_path)


def _upload_csv_to_s3(s3_resource, bucket, s3_key, rows):
    """Write a list of lists as CSV and upload to moto S3."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    body = buf.getvalue().encode("utf-8")
    s3_resource.Object(bucket, s3_key).put(Body=body)


def _upload_json_to_s3(s3_resource, bucket, s3_key, data):
    """Upload a JSON object to moto S3."""
    body = json.dumps(data).encode("utf-8")
    s3_resource.Object(bucket, s3_key).put(Body=body)


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


def _create_pipeline_table(dynamodb):
    dynamodb.create_table(
        TableName=os.environ["PIPELINE_TABLE"],
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "platform_data", "AttributeType": "S"},
            {"AttributeName": "pipeline_status", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "platform_data", "KeyType": "HASH"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "pipeline_status-index",
                "KeySchema": [
                    {"AttributeName": "pipeline_status", "KeyType": "HASH"},
                    {"AttributeName": "platform_data", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "KEYS_ONLY"},
            }
        ],
    )


def _create_term_table(dynamodb):
    dynamodb.create_table(
        TableName=os.environ["BOOK_TERM_TABLE"],
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "term", "AttributeType": "S"},
            {"AttributeName": "platform_data", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "term", "KeyType": "HASH"},
            {"AttributeName": "platform_data", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "platform_data-index",
                "KeySchema": [
                    {"AttributeName": "platform_data", "KeyType": "HASH"},
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
def moto_dynamo():
    """Spin up moto AWS (DynamoDB + S3) with all tables and a bucket.
    Resets module-level singletons each run."""

    with mock_aws():
        session = get_session()
        dynamodb = session.resource("dynamodb")
        s3 = session.resource("s3")

        _create_pipeline_table(dynamodb)
        _create_term_table(dynamodb)
        _create_corpus_term_table(dynamodb)
        s3.create_bucket(
            Bucket=os.environ["S3_BUCKET"],
            CreateBucketConfiguration={
                "LocationConstraint": os.environ["AWS_REGION"],
            },
        )

        yield session


@pytest.fixture
def term_table(moto_dynamo):
    return get_book_term_table()


@pytest.fixture
def corpus_term_table(moto_dynamo):
    return get_corpus_term_table()


@pytest.fixture
def pipeline_table(moto_dynamo):
    return get_pipeline_table()


@pytest.fixture
def pinecone_table():
    mock = MagicMock()
    mock.delete_book = MagicMock()
    mock.batch_upsert = MagicMock()
    return mock


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
def smith_s3_data(moto_dynamo):
    """Upload all S3 artifacts needed for publish(BOOK_SMITH):
    centroid model, seed models, POS CSVs, and metadata JSON.
    Also populates the PipelineTable entry."""
    s3 = moto_dynamo.resource("s3")
    bucket = os.environ["S3_BUCKET"]
    rng = np.random.RandomState(42)

    # Centroid model (aligned, mean across seeds).
    centroid_kv = _create_keyed_vectors(SMITH_TERM_ATTRS, VECTOR_DIM, rng)
    _save_keyed_vectors_to_s3(
        s3, bucket, f"kvectors/{BOOK_SMITH}/aligned/centroid.model", centroid_kv
    )

    # Per-seed models (each seed has the same terms, different vectors).
    for seed_idx in range(NUM_SEEDS):
        seed_kv = _create_keyed_vectors(SMITH_TERM_ATTRS, VECTOR_DIM, rng)
        _save_keyed_vectors_to_s3(
            s3,
            bucket,
            f"kvectors/{BOOK_SMITH}/aligned/{seed_idx}-seed.model",
            seed_kv,
        )

    # POS data: token lemmas and tags as CSVs.
    # Sentences are rows. Each term appears with a noun tag (NN),
    # "labour" also appears as a verb (VB) to match SMITH_TERMS tags.
    token_lemmas = [
        ["labour", "value", "rent"],
        ["labour", "value"],
    ]
    token_tags = [
        ["NN", "NN", "NN"],
        ["VB", "NN"],
    ]
    lemmas_key = f"tokens/{BOOK_SMITH}/lemmas.csv"
    tags_key = f"tokens/{BOOK_SMITH}/tags.csv"
    _upload_csv_to_s3(s3, bucket, lemmas_key, token_lemmas)
    _upload_csv_to_s3(s3, bucket, tags_key, token_tags)

    # Metadata JSON, at the key publish derives from the index.
    _upload_json_to_s3(s3, bucket, metadata_key(BOOK_SMITH), SMITH_METADATA)

    # PipelineTable entry with S3 keys.

    pt = get_pipeline_table()
    pt.table.put_item(
        Item={
            "platform_data": BOOK_SMITH,
            "pipeline_status": EntryStatus.SCRAPED_HTML,
            "s3_token_lemmas_key": lemmas_key,
            "s3_token_tags_key": tags_key,
            "published_year": SMITH_PUBLISHED_YEAR,
        }
    )
