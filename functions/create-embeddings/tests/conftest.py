import csv
import io
import os
import random
import tempfile

import boto3
import numpy as np
import pytest
from moto import mock_aws


os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("PIPELINE_TABLE", "pipeline-test")

from shared.commons import BookIndex

from constants import MIN_COUNT, VECTOR_SIZE


INDEX = BookIndex(3300)
SUBJECT = BookIndex(42)

TOKEN_LEMMAS = [
    ["labour", "the", "rent", "productive", "1776"],
    ["value", "of", "commodity", "wealth"],
]

KEPT_LEMMAS = [
    ["labour", "rent", "productive"],
    ["value", "commodity", "wealth"],
]

VOCABULARY = [
    f"term{chr(97 + position // 26)}{chr(97 + position % 26)}"
    for position in range(VECTOR_SIZE + 30)
]
SOLITARY_TERM = "solitary"

_rng = random.Random(0)
SYNTHETIC_PASSAGES = [
    [_rng.choice(VOCABULARY) for _ in range(24)] for _ in range(600)
]
SOLITARY_PASSAGES = SYNTHETIC_PASSAGES + [[SOLITARY_TERM]] * MIN_COUNT


def _create_pipeline_table(dynamodb):
    dynamodb.create_table(
        TableName=os.environ["PIPELINE_TABLE"],
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "book_id", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "book_id", "KeyType": "HASH"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "status-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "book_id", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "KEYS_ONLY"},
            }
        ],
    )


def _create_bucket(session):
    region = os.environ["AWS_REGION"]
    constraint = (
        {}
        if region == "us-east-1"
        else {"CreateBucketConfiguration": {"LocationConstraint": region}}
    )
    session.resource("s3").create_bucket(
        Bucket=os.environ["S3_BUCKET"], **constraint
    )


@pytest.fixture
def moto_dynamo():
    import shared.s3 as s3_module
    import shared.session as session_module
    import shared.tables.pipeline_entries as pipeline_entries_module

    session_module._session = None
    pipeline_entries_module._pipeline_entries = None
    s3_module._s3_resource = None

    with mock_aws():
        session = boto3.Session(region_name=os.environ["AWS_REGION"])
        _create_pipeline_table(session.resource("dynamodb"))
        _create_bucket(session)

        yield session


@pytest.fixture
def pipeline_entries(moto_dynamo):
    from shared.tables.pipeline_entries import get_pipeline_entries

    return get_pipeline_entries()


@pytest.fixture
def pipeline_item(moto_dynamo):
    def read(index=INDEX):
        return moto_dynamo.resource("dynamodb").Table(
            os.environ["PIPELINE_TABLE"]
        ).get_item(Key={"book_id": str(index)})["Item"]

    return read


@pytest.fixture
def token_lemmas(moto_dynamo):
    def write(index=INDEX, passages=TOKEN_LEMMAS):
        buffer = io.StringIO()
        csv.writer(buffer).writerows(passages)
        moto_dynamo.resource("s3").Object(
            os.environ["S3_BUCKET"], f"token_lemmas/{index}.csv"
        ).put(Body=buffer.getvalue().encode("utf-8"))

    return write


@pytest.fixture
def uploaded_embeddings(moto_dynamo):
    def read(index=INDEX):
        with tempfile.NamedTemporaryFile(suffix=".npz") as file:
            moto_dynamo.resource("s3").Object(
                os.environ["S3_BUCKET"], f"embeddings/{index}.npz"
            ).download_file(file.name)
            with np.load(file.name, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}

    return read
