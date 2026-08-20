import datetime
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

# What BOOK_HTML flattens to. Three headings, so a classification reply for this
# book is three lines.
BOOK_PAIRS = [
    ("h1", "The Wealth of Nations"),
    ("p", "An inquiry into the nature and causes."),
    ("h2", "BOOK I."),
    ("h2", "OF THE CAUSES OF IMPROVEMENT."),
    ("p", "The greatest improvement in the productive powers of labour."),
]

# The same page with every heading removed: what a book of pure prose looks like.
PROSE_ONLY_HTML = """<html><body>
  <p>An inquiry into the nature and causes.</p>
  <p>The greatest improvement in the productive powers of labour.</p>
</body></html>
"""


# ── AWS ───────────────────────────────────────────────────────────────


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
    where .env sets AWS_REGION=us-west-2. Creating the bucket unconditionally
    raises IllegalLocationConstraintException there, so the suite would pass
    locally and error in the gate it exists to clear.
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
def statuses(entries):
    """Read a book's current pipeline_status back out of the table."""

    def _status(index=INDEX):
        entry = entries.get_entry(index, ["platform_data", "pipeline_status"])
        return None if entry is None else entry.pipeline_status

    return _status


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
def book_manifest(bucket):
    """The per-book manifest submit leaves behind for collect to render from."""
    from book_records.schemas import BookTagTextPairs
    from book_records.utils import sanitize_llm_index

    def _book_manifest(index=INDEX, tag_text_pairs=None):
        book_tag_text_pairs = BookTagTextPairs(
            llm_index=sanitize_llm_index(index),
            index=index,
            tag_text_pairs=BOOK_PAIRS if tag_text_pairs is None else tag_text_pairs,
        )
        bucket.put_object(
            Key=f"standardize-headings/books/{index}.json",
            Body=book_tag_text_pairs.model_dump_json().encode("utf-8"),
        )
        return book_tag_text_pairs

    return _book_manifest


def s3_body(bucket, key):
    return bucket.Object(key).get()["Body"].read().decode("utf-8")


def s3_content_type(bucket, key):
    return bucket.Object(key).get()["ContentType"]


# ── Anthropic ─────────────────────────────────────────────────────────


@pytest.fixture
def anthropic_client():
    """The Anthropic SDK, mocked the way publish mocks Pinecone: moto covers S3 and
    DynamoDB, but nothing fakes the Batches API, so the client is a MagicMock."""
    client = MagicMock()
    client.messages.batches.create.return_value = MagicMock(id=BATCH_ID)
    return client


def succeeded_response(custom_id, text, stop_reason="end_turn"):
    """One finished batch result, built from the real SDK types.

    Not a MagicMock: yield_anthropic_content calls response.to_json() before it
    looks at the result, and serialize_content_block branches on isinstance. A
    mock would satisfy both without proving either works against the SDK.
    """
    from anthropic.types import Message, TextBlock, Usage
    from anthropic.types.messages import (
        MessageBatchIndividualResponse,
        MessageBatchSucceededResult,
    )

    return MessageBatchIndividualResponse(
        custom_id=custom_id,
        result=MessageBatchSucceededResult(
            type="succeeded",
            message=Message(
                id="msg_test",
                type="message",
                role="assistant",
                model="claude-sonnet-5",
                content=[TextBlock(type="text", text=text)],
                stop_reason=stop_reason,
                stop_sequence=None,
                usage=Usage(input_tokens=10, output_tokens=10),
            ),
        ),
    )


def errored_response(custom_id, message="request too large"):
    from anthropic.types.messages import (
        MessageBatchErroredResult,
        MessageBatchIndividualResponse,
    )
    from anthropic.types.shared import ErrorResponse, InvalidRequestError

    return MessageBatchIndividualResponse(
        custom_id=custom_id,
        result=MessageBatchErroredResult(
            type="errored",
            error=ErrorResponse(
                type="error",
                error=InvalidRequestError(
                    type="invalid_request_error", message=message
                ),
            ),
        ),
    )


@pytest.fixture
def batch_client():
    """An Anthropic client whose batch retrieve/results answer with real SDK objects."""
    from anthropic.types.messages import MessageBatch, MessageBatchRequestCounts

    def _batch_client(processing_status="ended", responses=()):
        responses = list(responses)
        batch = MessageBatch(
            id=BATCH_ID,
            type="message_batch",
            processing_status=processing_status,
            created_at=datetime.datetime(2026, 1, 1),
            expires_at=datetime.datetime(2026, 2, 1),
            request_counts=MessageBatchRequestCounts(
                processing=0,
                succeeded=len(responses),
                errored=0,
                canceled=0,
                expired=0,
            ),
            archived_at=None,
            cancel_initiated_at=None,
            ended_at=None,
            results_url=None,
        )
        client = MagicMock()
        client.messages.batches.retrieve.return_value = batch
        client.messages.batches.results.return_value = iter(responses)
        return client

    return _batch_client
