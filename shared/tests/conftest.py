import os

import boto3
import pytest
from moto import mock_aws


os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("TERM_CORPUS_TABLE", "corpus-term-test")


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


@pytest.fixture
def moto_dynamo():
    import shared.session as session_module
    import shared.tables.corpus_terms as corpus_terms_module
    session_module._session = None
    corpus_terms_module._corpus_term_table = None

    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        dynamodb = session.resource("dynamodb")
        s3 = session.resource("s3")

        _create_corpus_term_table(dynamodb)
        s3.create_bucket(Bucket=os.environ["S3_BUCKET"])

        yield session


@pytest.fixture
def corpus_term_table(moto_dynamo):
    from shared.tables.corpus_terms import get_corpus_term_table
    return get_corpus_term_table()
