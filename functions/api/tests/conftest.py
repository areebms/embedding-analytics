"""Shared test fixtures for the api Lambda.

Mocking philosophy:
- get_pipeline_table and TermTable are mocked at the import boundary
  in routers and describe_services.
- FastAPICache backend is left uninitialized (REDIS_URL unset),
  so the dependencies.cache shim becomes a no-op.
- OpenAI client in describe_services is replaced with a tiny stub.
- evaluate_tree, parser, and CI math run for real against synthetic
  byte vectors built by make_term_entry.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# Set env vars BEFORE any src/ module imports.
os.environ.setdefault("PRODUCTION_DOMAIN", "https://example.test")
os.environ.pop("REDIS_URL", None)
os.environ.setdefault("REDIS_PREFIX", "test")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in (str(SRC), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# -- Helpers -----------------------------------------------------------------


def seed_bytes(arr: np.ndarray) -> list:
    """Encode a (n_seeds, dim) float array as the list-of-bytes format
    that TermTable stores in DynamoDB."""
    arr16 = arr.astype(np.float16)
    return [bytes(arr16[i].tobytes()) for i in range(arr16.shape[0])]


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
    return {
        "term": term,
        "count_": count,
        "tags": tags if tags is not None else {"N"},
        "vectors": seed_bytes(vectors),
    }


# Default pipeline entries used by most fixtures.
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
    import copy

    table = MagicMock()
    table.get_all_entries.side_effect = lambda *a, **kw: copy.deepcopy(DEFAULT_PIPELINE)
    return table


@pytest.fixture
def mock_term_table():
    """TermTable mock. Tests configure get_entry / get_entries as needed."""
    table = MagicMock()
    table.get_entry.return_value = None
    table.get_entries.return_value = []
    return table


@pytest.fixture
def patch_storage(monkeypatch, mock_pipeline_table, mock_term_table):
    """Patch every storage entry point used by routers and describe_services."""
    import routers
    import describe_services

    monkeypatch.setattr(routers, "get_pipeline_table", lambda: mock_pipeline_table)
    monkeypatch.setattr(routers, "TermTable", lambda session: mock_term_table)
    monkeypatch.setattr(routers, "get_session", lambda: MagicMock())

    monkeypatch.setattr(describe_services, "get_pipeline_table", lambda: mock_pipeline_table)
    monkeypatch.setattr(describe_services, "TermTable", lambda session: mock_term_table)
    monkeypatch.setattr(describe_services, "get_session", lambda: MagicMock())

    return mock_pipeline_table, mock_term_table


@pytest.fixture
def patch_openai(monkeypatch):
    """Replace describe_services._get_openai_client with a stub."""
    import describe_services

    stub = MagicMock()
    stub.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='"market"'))]
    )
    monkeypatch.setattr(describe_services, "_get_openai_client", lambda: stub)
    return stub


@pytest.fixture
def client(patch_storage, patch_openai):
    from fastapi.testclient import TestClient
    from app import app

    with TestClient(app) as c:
        yield c
