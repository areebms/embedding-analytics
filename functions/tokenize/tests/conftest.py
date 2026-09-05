import os

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
)
# shared.session builds Session(profile_name=AWS_PROFILE); a profile named in .env does
# not exist inside the image.
os.environ.pop("AWS_PROFILE", None)

from shared.commons import BookIndex
from shared.tests_utils import (  # noqa: F401
    aws,
    bucket,
    entries,
    s3_body,
)


INDEX = BookIndex(3300)
INDEX_2 = BookIndex(11)
SUBJECT = BookIndex(12345)

# What standardize-html leaves at text/{index}.txt: one passage per block-level element,
# passages separated by a blank line. A heading, then two paragraphs -- the heading is
# the reason a passage is the unit, since a segmenter would weld it to the prose below.
BOOK_TEXT = """OF THE CAUSES OF IMPROVEMENT.

The greatest improvement in the productive powers of labour.

An inquiry into the nature and causes."""

BOOK_PASSAGES = [
    "OF THE CAUSES OF IMPROVEMENT.",
    "The greatest improvement in the productive powers of labour.",
    "An inquiry into the nature and causes.",
]


@pytest.fixture
def seed(entries):
    """Put one pipeline row at a given status, the way standardize-html would."""
    from shared.tables.pipeline_entries import PipelineEntry

    def _seed(status, index=INDEX, subject_ids={SUBJECT}):
        entries.put_entry(
            PipelineEntry(book_id=index, subject_ids=subject_ids, status=status)
        )
        return index

    return _seed


@pytest.fixture
def standardized_book(seed, bucket):
    """A book at STANDARDIZED with its text in the bucket: what tokenize is handed."""
    from shared.tables.pipeline_entries import EntryStatus

    def _standardized_book(index=INDEX, text=BOOK_TEXT):
        seed(EntryStatus.STANDARDIZED, index)
        bucket.put_object(Key=f"text/{index}.txt", Body=text.encode("utf-8"))
        return index

    return _standardized_book


@pytest.fixture
def events_client(mocker):
    import main

    client = mocker.Mock()
    client.put_events.return_value = {"FailedEntryCount": 0, "Entries": [{}]}
    mocker.patch.object(main, "get_session").return_value.client.return_value = client
    return client


def status_of(entries, index=INDEX):
    return entries.get_entry(index).status


def csv_rows(bucket, key):
    import csv
    import io

    return list(csv.reader(io.StringIO(s3_body(bucket, key))))
