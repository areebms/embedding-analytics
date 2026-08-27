"""Replaying a collected batch from S3, so an extractor fix costs no second batch.

The replay is keyed by heading position, so the tests that matter are the ones proving
it refuses a book whose headings moved rather than rendering it off by one.
"""

import pytest

from shared.tables.pipeline_entries import (
    EntryStatus,
    html_key,
    standardized_html_key,
    text_key,
)

from book_records.batch_index import save_batch_index
from book_records.schemas import BookTagTextPairs
from book_records.utils import sanitize_llm_index, save_book_tag_text_pairs
from llm_parse_response.fetch import save_batch_response
from llm_parse_response.rerender import rerender_from_saved_batch

from conftest import (
    BATCH_ID,
    BOOK_HTML,
    BOOK_PAIRS,
    INDEX,
    SUBJECT,
    s3_body,
    succeeded_response,
)


LLM_INDEX = sanitize_llm_index(str(INDEX))


@pytest.fixture
def collected_batch(seed, bucket):
    """A book left exactly as a finished RETRIEVE leaves one: standardized, with its
    manifest, its batch index, and the batch result still on S3."""

    def _collected(manifest_pairs=BOOK_PAIRS, html=BOOK_HTML):
        seed(EntryStatus.STANDARDIZED, INDEX, {SUBJECT})
        bucket.put_object(Key=html_key(INDEX), Body=html.encode("utf-8"))

        book = BookTagTextPairs(
            llm_index=LLM_INDEX, index=INDEX, tag_text_pairs=manifest_pairs
        )
        save_book_tag_text_pairs([book])
        save_batch_index(BATCH_ID, [book])
        save_batch_response(BATCH_ID, succeeded_response(LLM_INDEX))
        return INDEX

    return _collected


def test_a_collected_batch_replays_without_an_llm_call(collected_batch, bucket):
    collected_batch()

    assert rerender_from_saved_batch(BATCH_ID) == {
        "batch_id": BATCH_ID,
        "rerendered": 1,
        "skipped": [],
    }

    html = s3_body(bucket, standardized_html_key(INDEX))
    assert '<h1 data-block="title">The Wealth of Nations</h1>' in html
    assert '<h2 data-block="chapter">BOOK I.</h2>' in html
    assert '<h3 data-block="subsection">OF THE CAUSES OF IMPROVEMENT.</h3>' in html


def test_the_text_artifact_is_rewritten_too(collected_batch, bucket):
    collected_batch()
    rerender_from_saved_batch(BATCH_ID)

    assert s3_body(bucket, text_key(INDEX)).startswith("The Wealth of Nations\n\n")


def test_re_extraction_is_what_gets_rendered(collected_batch, bucket):
    """The point of the replay: the artifact comes from today's extractor, not from the
    text frozen in the manifest when the batch was sent."""
    stale = [(tag, "stale " + text) for tag, text in BOOK_PAIRS]
    collected_batch(manifest_pairs=stale)

    rerender_from_saved_batch(BATCH_ID)

    assert "stale" not in s3_body(bucket, standardized_html_key(INDEX))


def test_the_manifest_is_rewritten_so_a_later_retrieve_agrees(collected_batch):
    from llm_parse_response.fetch import load_book_tag_text_pairs

    collected_batch(manifest_pairs=[(tag, "stale") for tag, _ in BOOK_PAIRS])
    rerender_from_saved_batch(BATCH_ID)

    assert load_book_tag_text_pairs(INDEX).tag_text_pairs == BOOK_PAIRS


def test_a_book_whose_headings_moved_is_refused(collected_batch, bucket):
    """One heading fewer in the manifest means every saved position is off by one. The
    book needs a new batch; rendering it anyway would silently mislabel the book."""
    one_heading_fewer = [
        pair for pair in BOOK_PAIRS if pair != ("h2", "OF THE CAUSES OF IMPROVEMENT.")
    ]
    collected_batch(manifest_pairs=one_heading_fewer)

    assert rerender_from_saved_batch(BATCH_ID) == {
        "batch_id": BATCH_ID,
        "rerendered": 0,
        "skipped": [str(INDEX)],
    }

    with pytest.raises(bucket.meta.client.exceptions.ClientError):
        s3_body(bucket, standardized_html_key(INDEX))


def test_a_book_with_no_pipeline_row_is_skipped(bucket, entries):
    book = BookTagTextPairs(
        llm_index=LLM_INDEX, index=INDEX, tag_text_pairs=BOOK_PAIRS
    )
    save_book_tag_text_pairs([book])
    save_batch_index(BATCH_ID, [book])
    save_batch_response(BATCH_ID, succeeded_response(LLM_INDEX))

    assert rerender_from_saved_batch(BATCH_ID)["skipped"] == [str(INDEX)]
