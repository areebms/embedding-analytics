"""Fixtures for search-route tests (/similarity, /parse-describe).

Mocking philosophy:
- get_term_table, get_pipeline_table, TermTable, and get_session are patched
  at the import boundary in app.search.routers and app.search.services.describe.
- FastAPICache backend is left uninitialized (REDIS_URL unset),
  so the dependencies.cache shim becomes a no-op.
- OpenAI client in services.describe is replaced with a tiny stub.
- evaluate_tree, parser, and CI math run for real against synthetic
  byte vectors built by make_term_entry.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


# -- Helpers -----------------------------------------------------------------


def make_term_entry(
    term: str,
    n_seeds: int = 5,
    dim: int = 4,
    count: int = 100,
    tags: set[str] | None = None,
    seed: int = 0,
) -> dict:
    """Build a TermTable-shaped dict with realistic byte vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n_seeds, dim))
    arr16 = vectors.astype(np.float16)
    return {
        "term": term,
        "count_": count,
        "tags": tags if tags is not None else {"N"},
        "vectors": [bytes(arr16[i].tobytes()) for i in range(arr16.shape[0])],
    }


# -- Storage fixtures --------------------------------------------------------


@pytest.fixture
def mock_pipeline_table():
    """Pipeline table stub with one aligned book.

    get_vocabulary iterates book IDs from this table before querying terms,
    so at least one entry with s3_prefix_models must be present.
    """
    table = MagicMock()
    table.get_all_entries.return_value = [
        {"platform_data": "gutenberg-1", "s3_prefix_models": "models/1"},
    ]
    return table


@pytest.fixture
def mock_term_table():
    """TermTable mock. Tests configure get_entry / get_entries as needed."""
    table = MagicMock()
    table.get_entry.return_value = None
    table.get_entries.return_value = []
    return table


@pytest.fixture
def patch_openai(monkeypatch):
    """Replace OpenAI with a stub."""
    import app.search.services.describe as describe_services

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='"market"'))]
    )
    monkeypatch.setattr(describe_services, "OpenAI", lambda **_: client)
    return client


@pytest.fixture
def patch_tables(monkeypatch, mock_pipeline_table, mock_term_table):
    """Patch search-route storage entry points."""
    monkeypatch.setattr("app.search.routers.get_term_table", lambda: mock_term_table)
    monkeypatch.setattr("app.search.services.describe.get_pipeline_table", lambda: mock_pipeline_table)
    monkeypatch.setattr("app.search.services.describe.TermTable", lambda session: mock_term_table)
    monkeypatch.setattr("app.search.services.describe.get_session", lambda: MagicMock())
    return mock_pipeline_table, mock_term_table


@pytest.fixture
def client(patch_tables, patch_openai):
    from main import app

    with TestClient(app) as _client:
        yield _client
