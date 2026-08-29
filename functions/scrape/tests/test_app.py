"""Tests for the handler's stage dispatch.

One Lambda runs all three stages; the event's `stage` picks which. No test reached this
layer before, which is how the handler shipped calling a signature that no longer
existed.

`SUBJECT` takes a subject rather than an index, which is why the handler validates
`stage` before it validates either argument.
"""

import json

import pytest

from conftest import INDEX, s3_body
from shared.tables.pipeline_entries import EntryStatus, html_key, metadata_key


def test_metadata_stage_runs_and_returns_its_status(seed, bucket, mocker):
    import app
    import scrape

    seed(EntryStatus.LISTED)
    mocker.patch.object(scrape, "get_metadata", return_value={"language": ["English"]})

    result = app.handler({"index": "gutenberg-3300", "stage": "METADATA"}, None)

    assert result == {
        "stage": "METADATA",
        "book_id": INDEX,
        "status": EntryStatus.SCRAPED_METADATA,
    }
    assert json.loads(s3_body(bucket, metadata_key(INDEX)))


def test_content_stage_runs_the_other_half(seed, bucket, mocker):
    import app
    import scrape

    seed(EntryStatus.SCRAPED_METADATA)
    mocker.patch.object(scrape, "get_html", return_value="<html>raw</html>")

    result = app.handler({"index": "gutenberg-3300", "stage": "CONTENT"}, None)

    assert result == {
        "stage": "CONTENT",
        "book_id": INDEX,
        "status": EntryStatus.SCRAPED_HTML,
    }
    assert s3_body(bucket, html_key(INDEX)) == "<html>raw</html>"


def test_the_returned_status_is_json_serialisable(seed, mocker):
    """Step Functions branches on this payload, so it has to survive serialisation.
    The ranked status value is what `$contains(status, 'SCRAPED_HTML')` matches on."""
    import app
    import scrape

    seed(EntryStatus.LISTED)
    mocker.patch.object(scrape, "get_metadata", return_value={"language": ["French"]})

    result = app.handler({"index": "gutenberg-3300", "stage": "METADATA"}, None)

    assert json.loads(json.dumps(result)) == {
        "stage": "METADATA",
        "book_id": "gutenberg-3300",
        "status": "0101T_SCRAPED_SKIPPED_NON_ENGLISH",
    }


def test_an_event_with_no_stage_is_rejected(aws):
    import app

    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler({"index": "gutenberg-3300"}, None)


@pytest.mark.parametrize("stage", ["metadata", "content", "list", "html"])
def test_the_old_lowercase_stage_names_are_not_stages(aws, stage):
    """The stages were renamed to upper case, and `list` to `SUBJECT`, after `html`
    had already been renamed to `content`. Every earlier spelling is rejected rather
    than run."""
    import app

    with pytest.raises(ValueError, match="stage must be one of"):
        app.handler({"index": "gutenberg-3300", "subject": "12345", "stage": stage}, None)


def test_an_event_with_no_index_is_rejected(aws):
    import app

    with pytest.raises(ValueError, match="index is required"):
        app.handler({"stage": "METADATA"}, None)


def test_the_index_arrives_as_a_book_index(seed, mocker):
    """Stages call book_id.source_id, which a plain string does not have."""
    import app
    import scrape

    seed(EntryStatus.LISTED)
    get_metadata = mocker.patch.object(
        scrape, "get_metadata", return_value={"language": ["English"]}
    )

    app.handler({"index": "gutenberg-3300", "stage": "METADATA"}, None)

    get_metadata.assert_called_once_with(3300)


# ── SUBJECT stage ─────────────────────────────────────────────────────


def test_subject_stage_seeds_the_subject_and_returns_its_books(entries, mocker):
    import app
    import scrape

    mocker.patch.object(scrape, "get_book_ids", return_value=["3300", "846"])

    result = app.handler({"subject": "12345", "stage": "SUBJECT"}, None)

    assert result == {
        "subject": "12345",
        "found": 2,
        "created": 2,
        "indexes": ["gutenberg-3300", "gutenberg-846"],
    }
    assert entries.get_indexes(EntryStatus.LISTED) == [
        "gutenberg-3300",
        "gutenberg-846",
    ]


def test_subject_stage_needs_no_index(entries, mocker):
    """The other two stages reject an event with no index; this one must not."""
    import app
    import scrape

    mocker.patch.object(scrape, "get_book_ids", return_value=["3300"])

    assert app.handler({"subject": "12345", "stage": "SUBJECT"}, None)["found"] == 1


def test_a_subject_event_with_no_subject_is_rejected(aws):
    import app

    with pytest.raises(ValueError, match="subject is required"):
        app.handler({"stage": "SUBJECT"}, None)


def test_the_index_is_not_accepted_in_place_of_a_subject(aws):
    """A per-book event sent to the SUBJECT stage is a mistake, not a one-book subject."""
    import app

    with pytest.raises(ValueError, match="subject is required"):
        app.handler({"index": "gutenberg-3300", "stage": "SUBJECT"}, None)


def test_the_subject_is_read_from_a_json_body(entries, mocker):
    import app
    import scrape

    mocker.patch.object(scrape, "get_book_ids", return_value=["3300"])

    result = app.handler(
        {"body": json.dumps({"subject": "12345"}), "stage": "SUBJECT"}, None
    )

    assert result["subject"] == "12345"


def test_the_listed_indexes_are_json_serialisable(entries, mocker):
    """The Map iterates this array, so BookIndex has to come back out as plain strings."""
    import app
    import scrape

    mocker.patch.object(scrape, "get_book_ids", return_value=["3300", "846"])

    result = app.handler({"subject": "12345", "stage": "SUBJECT"}, None)

    assert json.loads(json.dumps(result))["indexes"] == [
        "gutenberg-3300",
        "gutenberg-846",
    ]
