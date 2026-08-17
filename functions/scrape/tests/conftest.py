import os

import pytest
from moto import mock_aws


os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("S3_BUCKET", "test-bucket")
os.environ.setdefault("PIPELINE_TABLE", "pipeline-test")

from shared.commons import BookIndex


INDEX = BookIndex(3300)

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
            {"AttributeName": "platform_data", "AttributeType": "S"},
            {"AttributeName": "pipeline_status", "AttributeType": "S"},
        ],
        KeySchema=[{"AttributeName": "platform_data", "KeyType": "HASH"}],
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
    """Put one pipeline row at a given status, the way `scrape.py list` would."""
    from shared.tables.pipeline_entries import PipelineEntry

    def _seed(status, index=INDEX):
        entries.put_entry(PipelineEntry(platform_data=index, pipeline_status=status))
        return index

    return _seed
