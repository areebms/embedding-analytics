import os

import boto3
import pytest
from moto import mock_aws


# Set, not setdefault: the deploy gate runs this suite inside the image with
# `docker run --env-file .env` (infra/deploy_lambdas.sh), so the real deployment
# config is on the environment. Inheriting it pointed the suite at the production
# bucket and table names, and at us-west-2 -- where moto's create_bucket fails with
# IllegalLocationConstraintException, because a bucket outside us-east-1 needs an
# explicit CreateBucketConfiguration. These are moto tests; they must not vary with
# whatever .env happens to hold.
os.environ.update(
    AWS_REGION="us-east-1",
    AWS_DEFAULT_REGION="us-east-1",
    AWS_ACCESS_KEY_ID="testing",
    AWS_SECRET_ACCESS_KEY="testing",
    AWS_SESSION_TOKEN="testing",
    S3_BUCKET="test-bucket",
    PIPELINE_TABLE="pipeline-test",
    TERM_CORPUS_TABLE="corpus-term-test",
)
# shared.session builds Session(profile_name=AWS_PROFILE); a profile named in .env
# does not exist inside the image.
os.environ.pop("AWS_PROFILE", None)


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
    import shared.tables.pipeline_entries as pipeline_entries_module
    session_module._session = None
    corpus_terms_module._corpus_term_table = None
    pipeline_entries_module._pipeline_entries = None

    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        dynamodb = session.resource("dynamodb")
        s3 = session.resource("s3")

        _create_corpus_term_table(dynamodb)
        _create_pipeline_table(dynamodb)
        s3.create_bucket(Bucket=os.environ["S3_BUCKET"])

        yield session


@pytest.fixture
def corpus_term_table(moto_dynamo):
    from shared.tables.corpus_terms import get_corpus_term_table
    return get_corpus_term_table()


@pytest.fixture
def pipeline_entries(moto_dynamo):
    from shared.tables.pipeline_entries import get_pipeline_entries
    return get_pipeline_entries()
