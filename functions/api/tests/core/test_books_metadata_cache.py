from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.core.dependencies import get_books_metadata_cache
from app.core.services import BooksMetadataCache
from shared.commons import BookIndex


def make_table(*entries):
    table = MagicMock()
    table.get_all_entries.return_value = list(entries)
    return table


@pytest.fixture
def table():
    # published_year arrives from DynamoDB as a Decimal, not an int.
    return make_table(
        {
            "platform_data": "gutenberg-1",
            "author": "Smith, Adam",
            "title": "Wealth of Nations",
            "published_year": Decimal("1776"),
            "s3_prefix_models": "models/1",
        },
        {
            "platform_data": "gutenberg-9",
            "author": "Say, J-B",
            "title": "Treatise",
            "published_year": Decimal("1803"),
        },
    )


def test_cache_keeps_only_aligned_books(table):
    # A book with no models built is one nothing downstream can measure.
    cache = BooksMetadataCache(table)

    assert cache.book_ids == [BookIndex(1)]


def test_cache_reads_the_year_as_an_int(table):
    assert BooksMetadataCache(table).books_metadata[BookIndex(1)].published_year == 1776


def test_cache_leaves_a_missing_year_none(table):
    cache = BooksMetadataCache(
        make_table(
            {
                "platform_data": "gutenberg-1",
                "author": "Smith, Adam",
                "title": "Wealth of Nations",
                "s3_prefix_models": "models/1",
            }
        )
    )

    assert cache.books_metadata[BookIndex(1)].published_year is None


def test_cache_scans_lazily_and_only_once(table):
    # Lazy so a request that never asks about books doesn't pay for the scan,
    # once so the warm container doesn't re-scan per request.
    cache = BooksMetadataCache(table)
    assert table.get_all_entries.call_count == 0

    cache.book_ids
    cache.books_metadata

    assert table.get_all_entries.call_count == 1


def test_cache_repr_carries_no_instance_address(table):
    # /books and /terms are wrapped in @cache, and fastapi-cache's default key
    # builder hashes the dependency's repr into the redis key -- an address in
    # here would make the cached response per-container.
    assert repr(BooksMetadataCache(table)) == "BooksMetadataCache()"


def test_cache_is_reused_across_requests_for_one_table():
    # Same contract as the term cache: the instance outlives the request that
    # built it, and is rebuilt only if the table itself is swapped.
    table = MagicMock()

    first = get_books_metadata_cache(table)

    assert get_books_metadata_cache(table) is first
    assert get_books_metadata_cache(MagicMock()) is not first
