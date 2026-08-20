"""Tests for the one manifest that ties a batch to the books inside it.

It sits at a single fixed key for the whole corpus, and it is the only record of
which llm_index means which book. Collect cannot run without it.
"""

import json

import pytest

from conftest import BATCH_ID, INDEX, INDEX_2, s3_body, s3_content_type

from book_records.batch_index import BATCH_INDEX_KEY, load_batch_index, save_batch_index
from book_records.constants import JSON_CONTENT_TYPE
from book_records.schemas import BookTagTextPairs


def book(index, llm_index=None):
    return BookTagTextPairs(
        llm_index=llm_index or str(index),
        index=index,
        tag_text_pairs=[("h1", "Title")],
    )


def test_the_manifest_sits_at_one_fixed_key_with_no_batch_id_in_it():
    """The batch is named inside the object, not in its key — collect checks the two
    against each other rather than guessing a path."""
    assert BATCH_INDEX_KEY == "standardize-headings/batch-details/index.json"
    assert BATCH_ID not in BATCH_INDEX_KEY


def test_saving_then_loading_returns_the_batch_and_its_books(bucket):
    save_batch_index(BATCH_ID, [book(INDEX), book(INDEX_2)])

    batch_index = load_batch_index(BATCH_ID)

    assert batch_index.llm_batch_id == BATCH_ID
    assert batch_index.llm_index_mapping == {
        "gutenberg-3300": INDEX,
        "gutenberg-11": INDEX_2,
    }


def test_the_indexes_come_back_as_book_indexes_not_plain_strings(bucket):
    """standardize_from_batch passes them to key builders and to the pipeline table."""
    from shared.commons import BookIndex

    save_batch_index(BATCH_ID, [book(INDEX)])

    index = load_batch_index(BATCH_ID).llm_index_mapping["gutenberg-3300"]
    assert isinstance(index, BookIndex)
    assert index.source_id == 3300


def test_the_manifest_is_stored_as_json(bucket):
    save_batch_index(BATCH_ID, [book(INDEX)])

    assert s3_content_type(bucket, BATCH_INDEX_KEY) == JSON_CONTENT_TYPE
    assert json.loads(s3_body(bucket, BATCH_INDEX_KEY))["llm_batch_id"] == BATCH_ID


def test_an_empty_batch_still_round_trips(bucket):
    save_batch_index(BATCH_ID, [])

    assert load_batch_index(BATCH_ID).llm_index_mapping == {}


def test_loading_a_different_batch_than_the_manifest_holds_is_refused(bucket):
    """Collecting batch A against batch B's manifest would write A's classifications
    onto B's books."""
    save_batch_index("msgbatch_older", [book(INDEX)])

    with pytest.raises(ValueError, match="manifest is for batch msgbatch_older"):
        load_batch_index(BATCH_ID)


def test_loading_before_anything_was_saved_raises(bucket):
    with pytest.raises(Exception):
        load_batch_index(BATCH_ID)


def test_a_second_save_replaces_the_first(bucket):
    """One key for the whole corpus: this overwrite is exactly why submit refuses to
    open a second batch while one is in flight."""
    save_batch_index("msgbatch_first", [book(INDEX)])
    save_batch_index(BATCH_ID, [book(INDEX_2)])

    batch_index = load_batch_index(BATCH_ID)

    assert batch_index.llm_index_mapping == {"gutenberg-11": INDEX_2}
