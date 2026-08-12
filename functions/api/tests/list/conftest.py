"""Fixtures for list-route tests (/books, /terms).

Mocking philosophy:
- the pipeline table is reached through the BooksMetadataCache dependency, so the
  cache is overridden per test; get_term_table is patched at the import
  boundary in app.list.routers.
- FastAPICache backend is left uninitialized (REDIS_URL unset),
  so the dependencies.cache shim becomes a no-op.
- The list routes run for real against the app's full router.
"""

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_books_metadata_cache
from app.core.services import BooksMetadataCache

os.environ.pop("REDIS_URL", None)


# -- Helpers -----------------------------------------------------------------


def make_pipeline_entry(
    gutenberg_id: int,
    author: str,
    year: int,
    title: str,
    aligned: bool = True,
) -> dict:
    """Build a fake pipeline-table row."""
    entry = {
        "platform_data": f"gutenberg-{gutenberg_id}",
        "author": author,
        "published_year": year,
        "title": title,
    }
    if aligned:
        entry["s3_prefix_models"] = f"models/{gutenberg_id}"
    return entry


DEFAULT_PIPELINE = [
    make_pipeline_entry(1, "Smith, Adam", 1776, "Wealth of Nations"),
    make_pipeline_entry(2, "Ricardo, David", 1817, "Principles of Political Economy"),
]


# -- Storage fixtures --------------------------------------------------------


@pytest.fixture
def mock_pipeline_table():
    """Pipeline table with two aligned books by default."""
    table = MagicMock()
    table.get_all_entries.return_value = DEFAULT_PIPELINE
    return table


@pytest.fixture
def mock_term_table():
    """BookTermTable mock. Tests configure get_entries as needed."""
    table = MagicMock()
    table.get_entry.return_value = None
    table.get_entries.return_value = []
    return table


@pytest.fixture
def mock_books_metadata_cache(mock_pipeline_table):
    """One books cache per test, as production keeps one per warm container."""
    return BooksMetadataCache(mock_pipeline_table)


@pytest.fixture
def patch_tables(
    monkeypatch, mock_pipeline_table, mock_term_table, mock_books_metadata_cache
):
    """Patch list-route storage entry points."""
    from main import app

    app.dependency_overrides[get_books_metadata_cache] = (
        lambda: mock_books_metadata_cache
    )
    monkeypatch.setattr("app.list.routers.get_book_term_table", lambda: mock_term_table)
    yield mock_pipeline_table, mock_term_table
    app.dependency_overrides.pop(get_books_metadata_cache, None)


@pytest.fixture
def client(patch_tables):
    from main import app

    with TestClient(app) as _client:
        yield _client
