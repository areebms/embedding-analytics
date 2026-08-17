"""Tests for the handler's stage dispatch.

One Lambda runs both stages; the event's `stage` picks which. No test reached this
layer before, which is how the handler shipped calling a signature that no longer
existed.
"""

import json

import pytest

from conftest import INDEX
from shared.tables.pipeline_entries import EntryStatus, html_key, metadata_key


def test_metadata_stage_runs_and_returns_its_status(seed, bucket, mocker):
    import app
    import scrape

    seed(EntryStatus.CREATED)
    mocker.patch.object(scrape, "get_metadata", return_value={"language": ["English"]})

    result = app.handler({"index": "gutenberg-3300", "stage": "metadata"}, None)

    assert result == {"index": INDEX, "status": EntryStatus.SCRAPED_METADATA}
    assert json.loads(bucket.Object(metadata_key(INDEX)).get()["Body"].read())


def test_content_stage_runs_the_other_half(seed, bucket, mocker):
    import app
    import scrape

    seed(EntryStatus.SCRAPED_METADATA)
    mocker.patch.object(scrape, "get_html", return_value="<html>raw</html>")

    result = app.handler({"index": "gutenberg-3300", "stage": "content"}, None)

    assert result == {"index": INDEX, "status": EntryStatus.SCRAPED_HTML}
    assert bucket.Object(html_key(INDEX)).get()["Body"].read() == b"<html>raw</html>"


def test_the_returned_status_is_json_serialisable(seed, mocker):
    """Step Functions branches on this payload, so it has to survive serialisation."""
    import app
    import scrape

    seed(EntryStatus.CREATED)
    mocker.patch.object(scrape, "get_metadata", return_value={"language": ["French"]})

    result = app.handler({"index": "gutenberg-3300", "stage": "metadata"}, None)

    assert json.loads(json.dumps(result)) == {
        "index": "gutenberg-3300",
        "status": "SCRAPED_SKIPPED_NON_ENGLISH",
    }


def test_an_event_with_no_stage_is_rejected(aws):
    import app

    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler({"index": "gutenberg-3300"}, None)


def test_the_old_subcommand_name_is_not_a_stage(aws):
    """`html` was the CLI subcommand before it was renamed to `content`."""
    import app

    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler({"index": "gutenberg-3300", "stage": "html"}, None)


def test_an_event_with_no_index_is_rejected(aws):
    import app

    with pytest.raises(ValueError, match="index is required"):
        app.handler({"stage": "metadata"}, None)


def test_the_index_arrives_as_a_book_index(seed, mocker):
    """Stages call index.source_id, which a plain string does not have."""
    import app
    import scrape

    seed(EntryStatus.CREATED)
    get_metadata = mocker.patch.object(
        scrape, "get_metadata", return_value={"language": ["English"]}
    )

    app.handler({"index": "gutenberg-3300", "stage": "metadata"}, None)

    get_metadata.assert_called_once_with(3300)
