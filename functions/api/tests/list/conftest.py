"""Fixtures for list-route tests (/books, /terms).

Mocking philosophy:
- get_pipeline_table and get_term_table are patched at the import boundary
  in app.list.routers and app.list.services.
- FastAPICache backend is left uninitialized (REDIS_URL unset),
  so the dependencies.cache shim becomes a no-op.
- The list routes run for real against the app's full router.
"""

import os
import copy
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

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
    """Pipeline table with two aligned books by default.

    Uses side_effect (not return_value) so each call to get_all_entries
    returns a fresh deep copy. This prevents BookResponse.from_entry's
    pop('platform_data') from mutating shared state across calls within
    a single test.
    """
    table = MagicMock()
    table.get_all_entries.side_effect = lambda *a, **kw: copy.deepcopy(DEFAULT_PIPELINE)
    return table


@pytest.fixture
def mock_term_table():
    """BookTermTable mock. Tests configure get_entries as needed."""
    table = MagicMock()
    table.get_entry.return_value = None
    table.get_entries.return_value = []
    return table


@pytest.fixture
def patch_tables(monkeypatch, mock_pipeline_table, mock_term_table):
    """Patch list-route storage entry points."""
    monkeypatch.setattr("app.list.services.get_pipeline_table", lambda: mock_pipeline_table)
    monkeypatch.setattr("app.list.routers.get_book_term_table", lambda: mock_term_table)
    return mock_pipeline_table, mock_term_table


@pytest.fixture
def client(patch_tables):
    from main import app

    with TestClient(app) as _client:
        yield _client
