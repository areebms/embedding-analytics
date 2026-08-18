import os
from unittest.mock import MagicMock

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
INDEX_2 = BookIndex(11)

BATCH_ID = "msgbatch_test123"

# A book shaped the way scrape leaves one: the Project Gutenberg licence wrapper
# around the real text, three headings, and prose between them. Reduces to
# h1/p/h2/h2/p once the wrapper is stripped.
BOOK_HTML = """<!DOCTYPE html>
<html>
  <body>
    <section id="pg-header"><p>The Project Gutenberg eBook of Everything</p></section>
    <h1>The Wealth of Nations</h1>
    <p>An inquiry into the nature and causes.</p>
    <h2>BOOK I.</h2>
    <h2>OF THE CAUSES OF IMPROVEMENT.</h2>
    <p>The greatest improvement in the productive powers of labour.</p>
    <section id="pg-footer"><p>End of the Project Gutenberg eBook</p></section>
  </body>
</html>
"""

# The same page with every heading removed: what a book of pure prose looks like.
PROSE_ONLY_HTML = """<html><body>
  <p>An inquiry into the nature and causes.</p>
  <p>The greatest improvement in the productive powers of labour.</p>
</body></html>
"""


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


def _create_bucket(session):
    """us-east-1 is the one region CreateBucket must not be told about; every other
    region requires the constraint.

    Both arms are live: the setdefault above only applies when the variable is
    unset, and the deploy gate runs this suite as `docker run --env-file .env`,
    where .env sets AWS_REGION=us-west-2. Creating the bucket unconditionally —
    as scrape's conftest does — raises IllegalLocationConstraintException there,
    so the suite would pass locally and error in the gate it exists to clear.
    """
    region = os.environ["AWS_REGION"]
    constraint = (
        {}
        if region == "us-east-1"
        else {"CreateBucketConfiguration": {"LocationConstraint": region}}
    )
    session.resource("s3").create_bucket(Bucket=os.environ["S3_BUCKET"], **constraint)


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
        _create_bucket(session)

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
    """Put one pipeline row at a given status, the way the scrape stages would."""
    from shared.tables.pipeline_entries import PipelineEntry

    def _seed(status, index=INDEX):
        entries.put_entry(PipelineEntry(platform_data=index, pipeline_status=status))
        return index

    return _seed


@pytest.fixture
def scraped_book(seed, bucket):
    """A book at SCRAPED_HTML with its raw html in the bucket: what submit sweeps."""
    from shared.tables.pipeline_entries import EntryStatus, html_key

    def _scraped_book(index=INDEX, html=BOOK_HTML):
        seed(EntryStatus.SCRAPED_HTML, index)
        bucket.put_object(Key=html_key(index), Body=html.encode("utf-8"))
        return index

    return _scraped_book


@pytest.fixture
def anthropic_client():
    """The Anthropic SDK, mocked the way publish mocks Pinecone: moto covers S3 and
    DynamoDB, but nothing fakes the Batches API, so the client is a MagicMock."""
    client = MagicMock()
    client.messages.batches.create.return_value = MagicMock(id=BATCH_ID)
    return client


@pytest.fixture
def statuses(entries):
    """Read a book's current pipeline_status back out of the table."""

    def _status(index=INDEX):
        entry = entries.get_entry(index, ["platform_data", "pipeline_status"])
        return None if entry is None else entry.pipeline_status

    return _status
