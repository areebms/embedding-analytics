import os

import pytest
from moto import mock_aws


os.environ.update(
    AWS_REGION="us-east-1",
    AWS_DEFAULT_REGION="us-east-1",
    AWS_ACCESS_KEY_ID="testing",
    AWS_SECRET_ACCESS_KEY="testing",
    AWS_SESSION_TOKEN="testing",
    S3_BUCKET="test-bucket",
    PIPELINE_TABLE="pipeline-test",
)

os.environ.pop("AWS_PROFILE", None)

from shared.commons import BookIndex


INDEX = BookIndex(3300)
SUBJECT = BookIndex(12345)

# A bibrec table trimmed to the rows get_metadata actually reads. The <a> in the
# language row is what drives the "*_link" key.
BIBREC_ENGLISH = """
<table class="bibrec">
  <tr><th>Author</th><td><a href="/ebooks/author/1">Smith, Adam</a></td></tr>
  <tr><th>Title</th><td>The Wealth of Nations</td></tr>
  <tr><th>Language</th><td><a href="/browse/languages/en">English</a></td></tr>
  <tr><td>no header, skipped</td></tr>
</table>
"""

BIBREC_FRENCH = BIBREC_ENGLISH.replace(">English<", ">French<")


def _create_pipeline_table(dynamodb):
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
    s3_module._s3_loader = None

    with mock_aws():
        session = boto3.Session(region_name=os.environ["AWS_REGION"])
        _create_pipeline_table(session.resource("dynamodb"))
        session.resource("s3").create_bucket(Bucket=os.environ["S3_BUCKET"])

        yield session


@pytest.fixture
def entries(aws):
    from shared.tables.pipeline_entries import get_pipeline_entries

    return get_pipeline_entries()


@pytest.fixture
def bucket(aws):
    return aws.resource("s3").Bucket(os.environ["S3_BUCKET"])


@pytest.fixture
def seed(entries):
    """Put one pipeline row at a given status, the way the SUBJECT stage would."""
    from shared.tables.pipeline_entries import PipelineEntry

    def _seed(status, index=INDEX, subject_ids={SUBJECT}):
        entries.put_entry(
            PipelineEntry(book_id=index, subject_ids=subject_ids, status=status)
        )
        return index

    return _seed
