"""Tests for the submit stage: opening one batch over the whole corpus.

The ordering inside submit is the part worth pinning down. A book marked
STANDARDIZE_SUBMITTED with no manifest to render from is stuck out of reach of both
stages, and STANDARDIZE_SUBMITTED is also the only thing stopping a second run from
resubmitting — and paying for — a corpus already in flight.
"""

import json

import pytest

from conftest import BATCH_ID, INDEX, INDEX_2, PROSE_ONLY_HTML, s3_body
from shared.tables.pipeline_entries import EntryStatus

from book_records.batch_index import BATCH_INDEX_KEY
from submit import submit


@pytest.fixture
def sending(mocker, anthropic_client):
    """The real send path, with only the SDK client faked."""
    import llm_classify_request.send_request as send_request

    mocker.patch.object(send_request, "get_client", return_value=anthropic_client)
    return anthropic_client


def test_an_empty_corpus_opens_no_batch(sending, entries, bucket):
    assert submit() == {"batch_id": None, "book_count": 0}
    sending.messages.batches.create.assert_not_called()


def test_the_swept_books_go_up_as_one_batch(sending, scraped_book, bucket):
    scraped_book(INDEX)
    scraped_book(INDEX_2)

    assert submit() == {"batch_id": BATCH_ID, "book_count": 2}
    sending.messages.batches.create.assert_called_once()


def test_every_submitted_book_is_marked_in_flight(sending, scraped_book, statuses):
    scraped_book(INDEX)
    scraped_book(INDEX_2)

    submit()

    assert statuses(INDEX) == EntryStatus.STANDARDIZE_SUBMITTED
    assert statuses(INDEX_2) == EntryStatus.STANDARDIZE_SUBMITTED


def test_the_manifest_names_the_batch_and_every_book_in_it(
    sending, scraped_book, bucket
):
    scraped_book(INDEX)
    scraped_book(INDEX_2)

    submit()

    manifest = json.loads(s3_body(bucket, BATCH_INDEX_KEY))
    assert manifest["llm_batch_id"] == BATCH_ID
    assert manifest["llm_index_mapping"] == {
        "gutenberg-3300": "gutenberg-3300",
        "gutenberg-11": "gutenberg-11",
    }


def test_a_book_with_no_headings_is_neither_submitted_nor_swept_again(
    sending, scraped_book, statuses
):
    scraped_book(INDEX, PROSE_ONLY_HTML)

    assert submit() == {"batch_id": None, "book_count": 0}
    assert statuses(INDEX) == EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS
    sending.messages.batches.create.assert_not_called()


def test_a_second_run_over_a_corpus_in_flight_submits_nothing(
    sending, scraped_book, bucket
):
    """The manifest sits at one fixed key, so a second batch would overwrite the
    index the first one still needs in order to be collected."""
    scraped_book(INDEX)
    submit()
    sending.messages.batches.create.reset_mock()

    scraped_book(INDEX_2)

    assert submit() == {"batch_id": None, "book_count": 0}
    sending.messages.batches.create.assert_not_called()
    # The first batch's manifest is untouched.
    assert json.loads(s3_body(bucket, BATCH_INDEX_KEY))["llm_index_mapping"] == {
        "gutenberg-3300": "gutenberg-3300"
    }


# ── the manifest is written before the status changes ─────────────────


def test_no_book_is_in_flight_at_the_moment_the_manifest_is_written(
    sending, scraped_book, entries, mocker
):
    """Reversing these two would leave a book marked STANDARDIZE_SUBMITTED with no
    manifest to render from: unreachable by submit, which skips it, and by collect,
    which cannot resolve it."""
    import submit as submit_module

    scraped_book(INDEX)
    scraped_book(INDEX_2)
    in_flight_when_written = []
    real_save_batch_index = submit_module.save_batch_index

    def recording_save(batch_id, book_tag_text_pairs):
        in_flight_when_written.extend(
            entries.get_indexes(EntryStatus.STANDARDIZE_SUBMITTED)
        )
        return real_save_batch_index(batch_id, book_tag_text_pairs)

    mocker.patch.object(submit_module, "save_batch_index", side_effect=recording_save)

    submit()

    assert in_flight_when_written == []
    assert entries.get_indexes(EntryStatus.STANDARDIZE_SUBMITTED) == [INDEX_2, INDEX]


def test_a_failed_manifest_write_leaves_no_book_stranded(
    sending, scraped_book, entries, statuses, mocker
):
    import submit as submit_module

    scraped_book(INDEX)
    mocker.patch.object(
        submit_module, "save_batch_index", side_effect=RuntimeError("s3 is down")
    )

    with pytest.raises(RuntimeError, match="s3 is down"):
        submit()

    assert entries.get_indexes(EntryStatus.STANDARDIZE_SUBMITTED) == []
    assert statuses(INDEX) == EntryStatus.SCRAPED_HTML


def test_the_status_write_touches_nothing_but_the_status(
    sending, scraped_book, entries, bucket
):
    """update_entries writes only the fields the caller set, so submit cannot clobber
    a column another stage owns."""
    from shared.tables.pipeline import get_pipeline_table

    scraped_book(INDEX)
    get_pipeline_table().table.update_item(
        Key={"platform_data": str(INDEX)},
        UpdateExpression="SET published_year = :year",
        ExpressionAttributeValues={":year": 1776},
    )

    submit()

    item = get_pipeline_table().table.get_item(Key={"platform_data": str(INDEX)})["Item"]
    assert item["pipeline_status"] == EntryStatus.STANDARDIZE_SUBMITTED
    assert item["published_year"] == 1776
