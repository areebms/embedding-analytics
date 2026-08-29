import os

import pytest


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
from shared.tests_utils import aws, bucket, entries, s3_body  # noqa: F401


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
