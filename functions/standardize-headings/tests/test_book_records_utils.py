"""Tests for the corpus sweep and the two S3 objects it hands to collect."""

import json

import pytest

from shared.tables.pipeline_entries import EntryStatus

from conftest import INDEX, INDEX_2, PROSE_ONLY_HTML


def _object(bucket, key):
    return bucket.Object(key).get()


# ── custom ids ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label, expected",
    [
        ("gutenberg-3300", "gutenberg-3300"),
        ("book/with spaces", "book_with_spaces"),
        ("a" * 100, "a" * 64),
    ],
    ids=["already-legal", "illegal-characters", "over-length"],
)
def test_sanitize_custom_id(label, expected):
    from book_records.utils import sanitize_custom_id

    assert sanitize_custom_id(label) == expected


# ── the handoff objects ───────────────────────────────────────────────


def test_a_book_record_is_written_as_json_under_its_index(bucket):
    from book_records.schemas import BookRecord
    from book_records.utils import save_book_record

    save_book_record(
        BookRecord(
            custom_id="gutenberg-1",
            index="gutenberg-1",
            tag_text_pairs=[("h1", "T")],
        )
    )

    written = _object(bucket, "standardize-batches/books/gutenberg-1.json")
    assert written["ContentType"] == "application/json; charset=utf-8"
    assert json.loads(written["Body"].read())["tag_text_pairs"] == [["h1", "T"]]


def test_the_batch_index_maps_custom_id_back_to_book_index(bucket):
    from book_records.schemas import BookRecord
    from book_records.utils import save_batch_index

    books = [
        BookRecord(custom_id="gutenberg-1", index="gutenberg-1", tag_text_pairs=[]),
        BookRecord(custom_id="gutenberg-2", index="gutenberg-2", tag_text_pairs=[]),
    ]

    save_batch_index("msgbatch_abc", books)

    written = json.loads(
        _object(bucket, "standardize-batches/index.json")["Body"].read()
    )
    assert written == {
        "batch_id": "msgbatch_abc",
        "custom_ids": {"gutenberg-1": "gutenberg-1", "gutenberg-2": "gutenberg-2"},
    }


# ── the sweep ─────────────────────────────────────────────────────────


def test_nothing_is_collected_when_no_book_has_been_scraped(entries):
    from book_records.utils import get_pending_book_records

    assert get_pending_book_records() == []


def test_an_in_flight_corpus_stops_the_sweep_before_it_starts(
    seed, scraped_book, entries, mocker
):
    """Finding any book at STANDARDIZE_SUBMITTED ends the run outright — the
    manifest sits at one fixed key, so a second batch would overwrite the index
    the first one still needs."""
    from book_records.utils import get_pending_book_records

    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX_2)
    scraped_book(INDEX)
    queried = mocker.spy(entries, "get_indexes")

    assert get_pending_book_records() == []
    assert [call.args[0] for call in queried.call_args_list] == [
        EntryStatus.STANDARDIZE_SUBMITTED
    ]


def test_one_record_is_built_and_saved_per_scraped_book(scraped_book, bucket):
    from book_records.utils import get_pending_book_records

    scraped_book(INDEX)
    scraped_book(INDEX_2)

    records = get_pending_book_records()

    assert {record.index for record in records} == {str(INDEX), str(INDEX_2)}
    assert {record.custom_id for record in records} == {str(INDEX), str(INDEX_2)}
    # Each record is on S3 by the time the sweep returns, not batched to the end.
    _object(bucket, f"standardize-batches/books/{INDEX}.json")
    _object(bucket, f"standardize-batches/books/{INDEX_2}.json")


def test_a_book_of_pure_prose_is_marked_skipped_and_left_out(scraped_book, statuses):
    from book_records.utils import get_pending_book_records

    scraped_book(INDEX, PROSE_ONLY_HTML)

    assert get_pending_book_records() == []
    assert statuses(INDEX) == EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS


def test_one_heading_anywhere_is_enough_to_survive(scraped_book):
    """The check is isdisjoint, not "mostly headings" — a single h-tag keeps the
    book in the batch."""
    from book_records.utils import get_pending_book_records

    scraped_book(INDEX, "<body><p>prose</p><p>more</p><h6>One heading</h6></body>")

    assert len(get_pending_book_records()) == 1


def test_a_book_whose_html_will_not_load_is_skipped_without_a_terminal_status(
    seed, scraped_book, statuses
):
    """Pins current behaviour, and it is a gap: the book keeps SCRAPED_HTML, so
    every future sweep re-fetches it and fails again. Nothing records the failure
    and nothing gives up on it."""
    from book_records.utils import get_pending_book_records

    seed(EntryStatus.SCRAPED_HTML, INDEX_2)  # no html object uploaded for this one
    scraped_book(INDEX)

    records = get_pending_book_records()

    assert [record.index for record in records] == [str(INDEX)]
    assert statuses(INDEX_2) == EntryStatus.SCRAPED_HTML


def test_a_bad_book_does_not_stop_the_ones_after_it(seed, scraped_book):
    from book_records.utils import get_pending_book_records

    # "gutenberg-11" sorts before "gutenberg-3300", so the broken book is first.
    seed(EntryStatus.SCRAPED_HTML, INDEX_2)
    scraped_book(INDEX)

    assert [record.index for record in get_pending_book_records()] == [str(INDEX)]
