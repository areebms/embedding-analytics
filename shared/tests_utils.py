"""moto fixtures for the pipeline's AWS plumbing.

Importing a fixture here into a conftest makes it available to that suite:

    from shared.testing import aws, bucket, entries  # noqa: F401
"""

import os

import pytest
from moto import mock_aws


def create_pipeline_table(dynamodb):
    dynamodb.create_table(
        TableName=os.environ["PIPELINE_TABLE"],
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "book_id", "AttributeType": "S"},
            {"AttributeName": "status", "AttributeType": "S"},
        ],
        KeySchema=[{"AttributeName": "book_id", "KeyType": "HASH"}],
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


@pytest.fixture
def aws():
    """moto S3 + DynamoDB, with the module-level singletons reset each run."""
    import boto3

    import shared.s3 as s3_module
    import shared.session as session_module
    import shared.tables.pipeline_entries as pipeline_entries_module

    session_module._session = None
    pipeline_entries_module._pipeline_entries = None
    s3_module._s3_resource = None

    with mock_aws():
        session = boto3.Session(region_name=os.environ["AWS_REGION"])
        create_pipeline_table(session.resource("dynamodb"))
        # No CreateBucketConfiguration: us-east-1 is the one region CreateBucket must
        # not be told about, and the conftest environment pins the suite there.
        session.resource("s3").create_bucket(Bucket=os.environ["S3_BUCKET"])

        yield session


@pytest.fixture
def entries(aws):
    from shared.tables.pipeline_entries import get_pipeline_entries

    return get_pipeline_entries()


@pytest.fixture
def bucket(aws):
    return aws.resource("s3").Bucket(os.environ["S3_BUCKET"])


def s3_body(bucket, key):
    return bucket.Object(key).get()["Body"].read().decode("utf-8")


def s3_content_type(bucket, key):
    return bucket.Object(key).get()["ContentType"]
