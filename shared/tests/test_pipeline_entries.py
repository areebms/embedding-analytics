"""Tests for PipelineEntries' PipelineEntry boundary (parse, partial update, coexistence)."""

from decimal import Decimal

from shared.commons import BookIndex
from shared.tables.pipeline_entries import (
    EntryStatus,
    PipelineEntry,
    html_key,
    metadata_key,
)

INDEX = BookIndex(3300)


def test_put_then_get_returns_status_as_enum(pipeline_entries):
    assert pipeline_entries.put_entry(
        PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.CREATED)
    )

    entry = pipeline_entries.get_entry(INDEX)
    assert entry.platform_data == INDEX
    assert entry.pipeline_status is EntryStatus.CREATED


def test_put_entry_is_a_conditional_create(pipeline_entries):
    entry = PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.CREATED)

    assert pipeline_entries.put_entry(entry)
    assert not pipeline_entries.put_entry(entry)


def test_get_entry_restores_key_omitted_by_projection(pipeline_entries):
    """is_status projects only pipeline_status; the row still knows its own key."""
    pipeline_entries.put_entry(
        PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.SCRAPED_HTML)
    )

    entry = pipeline_entries.get_entry(INDEX, ["pipeline_status"])
    assert entry.platform_data == INDEX
    assert entry.pipeline_status is EntryStatus.SCRAPED_HTML
    # Derived from platform_data, so a projection that never fetched it still has it.
    assert entry.s3_html_key == "html/gutenberg-3300.html"


def test_read_returns_a_book_index_even_for_a_plain_string_key(pipeline_entries):
    """The key comes back typed however the caller spelled it going in."""
    pipeline_entries.put_entry(
        PipelineEntry(
            platform_data="gutenberg-3300", pipeline_status=EntryStatus.SCRAPED_HTML
        )
    )

    entry = pipeline_entries.get_entry("gutenberg-3300", ["pipeline_status"])
    assert isinstance(entry.platform_data, BookIndex)
    assert entry.platform_data.source_id == 3300
    assert pipeline_entries.get_indexes() == [BookIndex(3300)]


def test_get_entry_returns_none_for_missing_row(pipeline_entries):
    assert pipeline_entries.get_entry("gutenberg-404") is None


def test_s3_keys_are_derived_from_the_index():
    """The documented layout (docs/pipeline.md) -- pinned against literals."""
    assert metadata_key(INDEX) == "metadata/gutenberg-3300.json"
    assert html_key(INDEX) == "html/gutenberg-3300.html"

    entry = PipelineEntry(platform_data=INDEX)
    assert entry.s3_metadata_key == "metadata/gutenberg-3300.json"
    assert entry.s3_html_key == "html/gutenberg-3300.html"


def test_s3_keys_are_never_written_to_the_table(pipeline_entries):
    """Derived, not stored: the row carries only the key and the status."""
    pipeline_entries.put_entry(
        PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.SCRAPED_HTML)
    )

    item = pipeline_entries.table.get_item(Key={"platform_data": INDEX})["Item"]
    assert set(item) == {"platform_data", "pipeline_status"}


def test_update_leaves_unset_fields_intact(pipeline_entries):
    """exclude_unset: a scrape write must not clobber another stage's columns."""
    pipeline_entries.put_entry(PipelineEntry(platform_data=INDEX))
    pipeline_entries.table.update_item(
        Key={"platform_data": INDEX},
        UpdateExpression="SET author = :a",
        ExpressionAttributeValues={":a": "Marx, Karl"},
    )

    pipeline_entries.update_entries(
        PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.SCRAPED_HTML)
    )

    item = pipeline_entries.table.get_item(Key={"platform_data": INDEX})["Item"]
    assert item["author"] == "Marx, Karl"
    assert item["pipeline_status"] == EntryStatus.SCRAPED_HTML


def test_update_with_nothing_set_is_a_no_op(pipeline_entries):
    """exclude_unset can leave no columns at all; DynamoDB rejects `SET` with no
    assignments, so the empty update has to stop before it reaches the table."""
    pipeline_entries.put_entry(
        PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.CREATED)
    )

    pipeline_entries.update_entries(PipelineEntry(platform_data=INDEX))

    entry = pipeline_entries.get_entry(INDEX)
    assert entry.pipeline_status is EntryStatus.CREATED


def test_get_indexes_returns_only_the_requested_status(pipeline_entries):
    """The filtered path is a Query on the status GSI, not a client-side filter."""
    pipeline_entries.put_entry(
        PipelineEntry(platform_data=BookIndex(1), pipeline_status=EntryStatus.SCRAPED_HTML)
    )
    pipeline_entries.put_entry(
        PipelineEntry(
            platform_data=BookIndex(2), pipeline_status=EntryStatus.SCRAPED_METADATA
        )
    )

    assert pipeline_entries.get_indexes(EntryStatus.SCRAPED_HTML) == [BookIndex(1)]
    assert pipeline_entries.get_indexes(EntryStatus.SCRAPED_METADATA) == [BookIndex(2)]


def test_get_indexes_is_empty_for_a_status_no_row_is_at(pipeline_entries):
    pipeline_entries.put_entry(
        PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.CREATED)
    )

    assert pipeline_entries.get_indexes(EntryStatus.SCRAPED_HTML) == []


def test_get_indexes_keeps_status_less_rows_only_when_unfiltered(pipeline_entries):
    """exclude_unset writes no pipeline_status, so the GSI never indexes this row --
    it must still show up in the unfiltered Scan."""
    pipeline_entries.put_entry(PipelineEntry(platform_data=BookIndex(1)))
    pipeline_entries.put_entry(
        PipelineEntry(platform_data=BookIndex(2), pipeline_status=EntryStatus.CREATED)
    )

    assert pipeline_entries.get_indexes() == [BookIndex(1), BookIndex(2)]
    assert pipeline_entries.get_indexes(EntryStatus.CREATED) == [BookIndex(2)]


def test_get_indexes_sorts_keys_lexicographically(pipeline_entries):
    """Unchanged from the old sorted() -- gutenberg-10 sorts before gutenberg-9."""
    for source_id in (9, 10):
        pipeline_entries.put_entry(
            PipelineEntry(
                platform_data=BookIndex(source_id),
                pipeline_status=EntryStatus.SCRAPED_HTML,
            )
        )

    assert pipeline_entries.get_indexes(EntryStatus.SCRAPED_HTML) == [
        BookIndex(10),
        BookIndex(9),
    ]


def test_row_carrying_later_stage_fields_still_parses(pipeline_entries):
    """extra="ignore": publish and align write columns scrape's model never names."""
    pipeline_entries.put_entry(
        PipelineEntry(platform_data=INDEX, pipeline_status=EntryStatus.SCRAPED_HTML)
    )
    pipeline_entries.table.update_item(
        Key={"platform_data": INDEX},
        UpdateExpression="SET author = :a, mean_disparity = :d",
        ExpressionAttributeValues={":a": "Marx, Karl", ":d": Decimal("0.42")},
    )

    entry = pipeline_entries.get_entry(INDEX)
    assert entry.pipeline_status is EntryStatus.SCRAPED_HTML
    assert not hasattr(entry, "author")
