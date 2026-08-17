"""Tests for the two scrape stages and the status transitions they drive."""

import json

import pytest

from conftest import INDEX
from shared.tables.pipeline_entries import EntryStatus, html_key, metadata_key


ENGLISH_METADATA = {"language": ["English"], "author": ["Smith, Adam"]}
FRENCH_METADATA = {"language": ["French"], "author": ["Rousseau, Jean-Jacques"]}


def _body(bucket, key):
    return bucket.Object(key).get()["Body"].read().decode("utf-8")


# ── metadata stage ────────────────────────────────────────────────────


def test_metadata_uploads_json_and_advances_the_status(seed, bucket, mocker):
    import scrape

    seed(EntryStatus.CREATED)
    mocker.patch.object(scrape, "get_metadata", return_value=ENGLISH_METADATA)

    status = scrape.scrape_book_metadata(INDEX)

    assert status == EntryStatus.SCRAPED_METADATA
    assert json.loads(_body(bucket, metadata_key(INDEX))) == ENGLISH_METADATA
    assert scrape.get_status(INDEX) == EntryStatus.SCRAPED_METADATA


def test_metadata_marks_a_non_english_book_skipped(seed, bucket, mocker):
    """The returned status is what the state machine's Choice branches on."""
    import scrape

    seed(EntryStatus.CREATED)
    mocker.patch.object(scrape, "get_metadata", return_value=FRENCH_METADATA)

    status = scrape.scrape_book_metadata(INDEX)

    assert status == EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH
    assert scrape.get_status(INDEX) == EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH
    # The metadata still lands: knowing why a book was skipped is worth the object.
    assert json.loads(_body(bucket, metadata_key(INDEX))) == FRENCH_METADATA


def test_metadata_is_idempotent_and_reports_the_current_status(seed, mocker):
    import scrape

    seed(EntryStatus.SCRAPED_HTML)
    get_metadata = mocker.patch.object(scrape, "get_metadata")

    status = scrape.scrape_book_metadata(INDEX)

    assert status == EntryStatus.SCRAPED_HTML
    get_metadata.assert_not_called()


# ── content stage ─────────────────────────────────────────────────────


def test_content_uploads_raw_html_and_advances_the_status(seed, bucket, mocker):
    import scrape

    seed(EntryStatus.SCRAPED_METADATA)
    mocker.patch.object(scrape, "get_html", return_value="<html>raw</html>")

    status = scrape.scrape_book_content(INDEX)

    assert status == EntryStatus.SCRAPED_HTML
    assert _body(bucket, html_key(INDEX)) == "<html>raw</html>"
    assert scrape.get_status(INDEX) == EntryStatus.SCRAPED_HTML


def test_content_will_not_run_before_metadata(seed, mocker):
    import scrape

    seed(EntryStatus.CREATED)
    get_html = mocker.patch.object(scrape, "get_html")

    status = scrape.scrape_book_content(INDEX)

    assert status == EntryStatus.CREATED
    get_html.assert_not_called()


def test_content_will_not_run_for_a_skipped_book(seed, mocker):
    import scrape

    seed(EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH)
    get_html = mocker.patch.object(scrape, "get_html")

    assert scrape.scrape_book_content(INDEX) == EntryStatus.SCRAPED_SKIPPED_NON_ENGLISH
    get_html.assert_not_called()


# ── seeding is a prerequisite ─────────────────────────────────────────


def test_an_unseeded_book_names_itself_and_the_remedy(aws):
    import scrape

    with pytest.raises(LookupError, match="gutenberg-404"):
        scrape.get_status("gutenberg-404")


def test_a_row_with_no_status_is_a_distinct_error(entries):
    import scrape
    from shared.tables.pipeline_entries import PipelineEntry

    entries.put_entry(PipelineEntry(platform_data=INDEX))

    with pytest.raises(LookupError, match="no status"):
        scrape.get_status(INDEX)


# ── bulk CLI helpers ──────────────────────────────────────────────────


def test_subject_list_seeds_every_book_once(entries, mocker):
    import scrape

    mocker.patch.object(scrape, "get_book_ids", return_value=["3300", "846"])

    scrape.scrape_subject_book_list("subject-42")
    scrape.scrape_subject_book_list("subject-42")  # put_entry is a conditional create

    assert entries.get_indexes(EntryStatus.CREATED) == ["gutenberg-3300", "gutenberg-846"]


def test_subject_list_survives_one_book_failing_to_seed(entries, mocker):
    import scrape

    mocker.patch.object(scrape, "get_book_ids", return_value=["3300", "846"])
    mocker.patch.object(
        entries, "put_entry", side_effect=[RuntimeError("throttled"), True]
    )

    scrape.scrape_subject_book_list("subject-42")  # must not raise

    assert entries.put_entry.call_count == 2


def test_sleepy_map_keeps_going_after_one_book_fails(mocker):
    import scrape

    mocker.patch.object(scrape, "sleep")
    seen = []

    def stage(index):
        seen.append(index)
        if index == "b":
            raise RuntimeError("gutenberg said no")

    scrape.sleepy_map(stage, ["a", "b", "c"])

    assert seen == ["a", "b", "c"]
