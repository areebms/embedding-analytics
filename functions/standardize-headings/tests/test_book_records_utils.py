"""Tests for the submit stage's sweep: which books it picks up, which it passes over,
and what it leaves in the bucket for collect to render from.

The sweep is corpus-wide and costs money downstream, so the guards that stop it
matter as much as the happy path.
"""

import json

import pytest

from conftest import BOOK_HTML, BOOK_PAIRS, INDEX, INDEX_2, PROSE_ONLY_HTML, s3_body, s3_content_type
from shared.tables.pipeline_entries import EntryStatus, html_key

from book_records.constants import JSON_CONTENT_TYPE
from book_records.schemas import BookTagTextPairs
from book_records.utils import (
    get_pending_book_tag_text_pairs,
    sanitize_llm_index,
    save_book_tag_text_pairs,
)


def book_key(index):
    return f"standardize-headings/books/{index}.json"


# ── sanitize_llm_index ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label, expected",
    [
        ("gutenberg-3300", "gutenberg-3300"),
        ("under_score", "under_score"),
        ("has spaces", "has_spaces"),
        ("slash/colon:dot.", "slash_colon_dot_"),
        ("Ünïcødé", "_n_c_d_"),
    ],
)
def test_illegal_characters_become_underscores(label, expected):
    """Anthropic's custom_id accepts only ^[a-zA-Z0-9_-]{1,64}$."""
    assert sanitize_llm_index(label) == expected


def test_a_long_label_is_truncated_to_the_custom_id_limit():
    assert len(sanitize_llm_index("x" * 200)) == 64


def test_a_book_index_survives_sanitising_unchanged():
    """The whole corpus is gutenberg-N, so the common case must be a no-op."""
    assert sanitize_llm_index(INDEX) == "gutenberg-3300"


# ── save_book_tag_text_pairs ──────────────────────────────────────────


def test_the_manifest_is_written_as_json_under_the_books_prefix(bucket):
    book_tag_text_pairs = BookTagTextPairs(
        llm_index="gutenberg-3300", index=INDEX, tag_text_pairs=BOOK_PAIRS
    )

    save_book_tag_text_pairs(book_tag_text_pairs)

    saved = json.loads(s3_body(bucket, book_key(INDEX)))
    assert saved["index"] == "gutenberg-3300"
    assert saved["llm_index"] == "gutenberg-3300"
    assert [tuple(pair) for pair in saved["tag_text_pairs"]] == BOOK_PAIRS
    assert s3_content_type(bucket, book_key(INDEX)) == JSON_CONTENT_TYPE


# ── get_pending_book_tag_text_pairs ───────────────────────────────────


def test_a_book_at_scraped_html_is_swept_up_and_gets_a_manifest(scraped_book, bucket):
    scraped_book()

    pending = get_pending_book_tag_text_pairs()

    assert [book.index for book in pending] == [INDEX]
    assert pending[0].llm_index == "gutenberg-3300"
    assert pending[0].tag_text_pairs == BOOK_PAIRS
    # The manifest lands during the sweep, not after the batch is opened.
    assert json.loads(s3_body(bucket, book_key(INDEX)))["index"] == "gutenberg-3300"


def test_an_empty_corpus_sweeps_up_nothing(entries):
    assert get_pending_book_tag_text_pairs() == []


def test_a_batch_already_in_flight_stops_the_sweep_before_it_starts(
    scraped_book, seed, bucket
):
    """One fixed manifest key means a second batch would overwrite the index the
    first one still needs. Finding any book in flight has to stop the run outright."""
    scraped_book()
    seed(EntryStatus.STANDARDIZE_SUBMITTED, INDEX_2)

    assert get_pending_book_tag_text_pairs() == []
    # The eligible book was never even read, so no manifest was written for it.
    with pytest.raises(Exception):
        s3_body(bucket, book_key(INDEX))


def test_a_book_with_no_headings_is_marked_skipped_and_left_out(
    scraped_book, statuses
):
    """Nothing to classify, so it must not cost a request — but it must also stop
    being swept up on every later run."""
    scraped_book(INDEX, PROSE_ONLY_HTML)

    assert get_pending_book_tag_text_pairs() == []
    assert statuses(INDEX) == EntryStatus.SCRAPED_SKIPPED_NO_HEADINGS


def test_a_book_whose_html_is_missing_is_skipped_not_raised(seed, statuses):
    """One unreadable book must not abort a corpus-wide sweep."""
    seed(EntryStatus.SCRAPED_HTML, INDEX)

    assert get_pending_book_tag_text_pairs() == []
    # Skipped, but not reclassified: the row still says the html should be there.
    assert statuses(INDEX) == EntryStatus.SCRAPED_HTML


def test_the_good_books_still_come_back_when_a_neighbour_fails(
    scraped_book, seed, bucket
):
    scraped_book(INDEX_2, BOOK_HTML)
    seed(EntryStatus.SCRAPED_HTML, INDEX)  # no html object for this one

    pending = get_pending_book_tag_text_pairs()

    assert [book.index for book in pending] == [INDEX_2]


def test_every_swept_book_gets_its_own_manifest(scraped_book, bucket):
    scraped_book(INDEX)
    scraped_book(INDEX_2)

    pending = get_pending_book_tag_text_pairs()

    assert sorted(book.index for book in pending) == [INDEX_2, INDEX]
    assert json.loads(s3_body(bucket, book_key(INDEX)))["index"] == "gutenberg-3300"
    assert json.loads(s3_body(bucket, book_key(INDEX_2)))["index"] == "gutenberg-11"
