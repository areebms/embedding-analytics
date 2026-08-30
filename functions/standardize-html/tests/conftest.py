import datetime
import os
from unittest.mock import MagicMock

import pytest


# Set, not setdefault: the deploy gate runs this suite inside the image with
# `docker run --env-file .env` (infra/deploy_lambdas.sh), so the real deployment config
# is on the environment. Inheriting it points the suite at the production bucket and
# table, and at a region where create_bucket needs the CreateBucketConfiguration these
# fixtures deliberately do not pass. These are moto tests; they must not vary with
# whatever .env happens to hold.
os.environ.update(
    AWS_REGION="us-east-1",
    AWS_DEFAULT_REGION="us-east-1",
    AWS_ACCESS_KEY_ID="testing",
    AWS_SECRET_ACCESS_KEY="testing",
    AWS_SESSION_TOKEN="testing",
    S3_BUCKET="test-bucket",
    PIPELINE_TABLE="pipeline-test",
    ANTHROPIC_API_KEY="test-key",
)
# shared.session builds Session(profile_name=AWS_PROFILE); a profile named in .env does
# not exist inside the image.
os.environ.pop("AWS_PROFILE", None)

from anthropic.types import Message, TextBlock, Usage
from anthropic.types.messages import (
    MessageBatch,
    MessageBatchErroredResult,
    MessageBatchIndividualResponse,
    MessageBatchRequestCounts,
    MessageBatchSucceededResult,
)
from anthropic.types.shared import ErrorResponse, InvalidRequestError

from shared.commons import BookIndex
from shared.tests_utils import (  # noqa: F401
    aws,
    bucket,
    entries,
    s3_body,
    s3_content_type,
)


INDEX = BookIndex(3300)
INDEX_2 = BookIndex(11)
SUBJECT = BookIndex(12345)

BATCH_ID = "msgbatch_test123"

# A book shaped the way scrape leaves one: the Project Gutenberg licence wrapper around
# the real text, three headings, and prose between them.
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

# What BOOK_HTML flattens to. Three headings, so a classification reply for this book is
# three lines.
BOOK_PAIRS = [
    ("h1", "The Wealth of Nations"),
    ("p", "An inquiry into the nature and causes."),
    ("h2", "BOOK I."),
    ("h2", "OF THE CAUSES OF IMPROVEMENT."),
    ("p", "The greatest improvement in the productive powers of labour."),
]

# The reply BOOK_PAIRS earns: one line per heading, in order. The first is a title
# page heading, which is paratext now that the library record carries the title.
BOOK_REPLY = "0|drop\n1|chapter\n2|section\n"

# The same page with every heading removed: what a book of pure prose looks like.
PROSE_ONLY_HTML = """<html><body>
  <p>An inquiry into the nature and causes.</p>
  <p>The greatest improvement in the productive powers of labour.</p>
</body></html>
"""


# ── AWS ───────────────────────────────────────────────────────────────


@pytest.fixture
def seed(entries):
    """Put one pipeline row at a given status, the way the scrape stages would."""
    from shared.tables.pipeline_entries import PipelineEntry

    def _seed(status, index=INDEX, subject_ids={SUBJECT}):
        entries.put_entry(
            PipelineEntry(book_id=index, subject_ids=subject_ids, status=status)
        )
        return index

    return _seed


@pytest.fixture
def scraped_book(seed, bucket):
    """A book at SCRAPED_HTML with its raw html in the bucket: what SEND is handed."""
    from shared.tables.pipeline_entries import EntryStatus

    def _scraped_book(index=INDEX, html=BOOK_HTML):
        seed(EntryStatus.SCRAPED_HTML, index)
        bucket.put_object(Key=f"html/{index}.html", Body=html.encode("utf-8"))
        return index

    return _scraped_book


def status_of(entries, index=INDEX):
    return entries.get_entry(index).status


# ── Anthropic ─────────────────────────────────────────────────────────


def succeeded_response(custom_id, text=BOOK_REPLY, stop_reason="end_turn"):
    """One finished batch result, built from the real SDK types.

    Not a MagicMock: yield_anthropic_content calls response.to_json() before it looks at
    the result, and serialize_content_block branches on isinstance. A mock would satisfy
    both without proving either works against the SDK.
    """
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
def send_client(monkeypatch):
    """The client SEND opens its batch with.

    moto covers S3 and DynamoDB, but nothing fakes the Batches API, so the client itself
    is a MagicMock -- patched over get_client, which would otherwise build a real one.
    """
    client = MagicMock()
    client.messages.batches.create.return_value = MagicMock(
        id=BATCH_ID, processing_status="in_progress"
    )
    monkeypatch.setattr(
        "llm_request.send_anthropic_request.get_client", lambda: client
    )
    return client


@pytest.fixture
def submitted_batch(scraped_book, send_client):
    """Books taken through SEND: manifests written, status STANDARDIZE_SUBMITTED.

    Going through the real stage rather than planting a manifest, so what RETRIEVE reads
    back is what SEND actually wrote.
    """
    import app

    def _submitted_batch(indexes=(INDEX,), html=BOOK_HTML):
        for index in indexes:
            scraped_book(index, html)
        return app.handler({"book_ids": [str(index) for index in indexes]}, None)

    return _submitted_batch


@pytest.fixture
def collect_client(monkeypatch):
    """The client RETRIEVE settles a batch with, answering with real SDK objects."""

    def _collect_client(processing_status="ended", responses=()):
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
        monkeypatch.setattr(
            "llm_response.standardize.get_client", lambda: client
        )
        return client

    return _collect_client
