import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_books_metadata_cache
from app.core.services import BooksMetadataCache
from app.search.constants import (
    MAX_RANK_FOR_STABLE_TERM,
    MAX_RANK_FOR_UNSTABLE_TERM,
    NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY,
    NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING,
)
from app.search.dependencies import get_books_term_cache
from app.search.services.semantic_drift import BooksTermCache
from shared.tables.book_terms import get_book_term_table

os.environ.pop("REDIS_URL", None)

LOCAL_VOCAB_FLOOR = max(
    NUM_NEAREST_TERMS_FOR_LOCAL_COSINE_SIMILARITY,
    NUM_NEAREST_TERMS_FOR_SIMILARITY_CENTERING,
    MAX_RANK_FOR_STABLE_TERM,
    MAX_RANK_FOR_UNSTABLE_TERM,
)

NAMED_VOCAB = ["labour", "value", "wage", "rent", "stock", "price", "profit", "capital"]
FILLER_VOCAB = [f"filler{n:03d}" for n in range(LOCAL_VOCAB_FLOOR)]
VOCAB = NAMED_VOCAB + FILLER_VOCAB


# -- Helpers -----------------------------------------------------------------


def make_term_entry(
    term: str,
    n_seeds: int = 5,
    dim: int = 4,
    count: int = 100,
    tags: set[str] | None = None,
    seed: int = 0,
) -> dict:
    """Build a BookTermTable-shaped dict with realistic byte vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n_seeds, dim))
    arr16 = vectors.astype(np.float16)
    return {
        "term": term,
        "count_": count,
        "tags": tags if tags is not None else {"N"},
        "vectors": [bytes(arr16[i].tobytes()) for i in range(arr16.shape[0])],
    }


def book_rows(book_seed, vocab=VOCAB, n_seeds=5):
    return {
        term: make_term_entry(term, n_seeds=n_seeds, seed=book_seed * 100 + i)
        for i, term in enumerate(vocab)
    }


def set_multi_book_table(term_table, books):

    def get_entries(book_id, fields=None):
        return list(books.get(book_id.source_id, {}).values())

    def batch_get_entries(terms, book_id, fields=None):
        entries = books.get(book_id.source_id, {})
        return [entries[t] for t in terms if t in entries]

    term_table.get_entries.side_effect = get_entries
    term_table.batch_get_entries.side_effect = batch_get_entries


def set_term_table(term_table, entries_by_term, book_id="gutenberg-1"):
    """Configure get_entry and batch_get_entries for a single-book test."""

    def get_entry(term, platform_data, fields=None):
        if platform_data == book_id and term in entries_by_term:
            return entries_by_term[term]
        return None

    def batch_get_entries(terms, platform_data, fields=None):
        if platform_data != book_id:
            return []
        return [entries_by_term[t] for t in terms if t in entries_by_term]

    term_table.get_entry.side_effect = get_entry
    term_table.batch_get_entries.side_effect = batch_get_entries


# -- Storage fixtures --------------------------------------------------------


@pytest.fixture
def mock_pipeline_table():
    """Pipeline table stub with one aligned book.
    """
    table = MagicMock()
    table.get_all_entries.return_value = [
        {"platform_data": "gutenberg-1", "s3_prefix_models": "models/1"},
    ]
    return table


@pytest.fixture
def mock_term_table():
    """BookTermTable mock. Tests configure get_entry / get_entries as needed."""
    table = MagicMock()
    table.get_entry.return_value = None
    table.get_entries.return_value = []
    table.batch_get_entries.return_value = []
    return table


@pytest.fixture
def mock_books_metadata_cache(mock_pipeline_table):
    """The pipeline-metadata cache for one test."""
    return BooksMetadataCache(mock_pipeline_table)


@pytest.fixture
def mock_books_cache(mock_term_table):
    """The term-matrix cache for one test.

    In production this outlives the request, so overriding it per test is what
    keeps one test's books out of the next -- and tests that want a cold cache
    mid-test clear this instance rather than a module global.
    """
    return BooksTermCache(mock_term_table)


@pytest.fixture
def patch_openai(monkeypatch):
    """Replace OpenAI with a stub."""
    import app.search.services.describe as describe_services

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='"market"'))]
    )
    monkeypatch.setattr(describe_services, "get_openai_client", lambda: client)
    return client


@pytest.fixture
def patch_tables(
    monkeypatch,
    mock_pipeline_table,
    mock_term_table,
    mock_books_cache,
    mock_books_metadata_cache,
):
    """Patch search-route storage entry points."""
    from main import app

    app.dependency_overrides[get_book_term_table] = lambda: mock_term_table
    app.dependency_overrides[get_books_term_cache] = lambda: mock_books_cache
    app.dependency_overrides[get_books_metadata_cache] = (
        lambda: mock_books_metadata_cache
    )
    monkeypatch.setattr(
        "app.search.services.describe.get_book_term_table", lambda: mock_term_table
    )
    yield mock_pipeline_table, mock_term_table
    # Pop rather than clear(): the list-route fixtures register their own
    # overrides on this same app object.
    app.dependency_overrides.pop(get_book_term_table, None)
    app.dependency_overrides.pop(get_books_term_cache, None)
    app.dependency_overrides.pop(get_books_metadata_cache, None)


@pytest.fixture
def term_table(patch_tables):
    return patch_tables[1]


@pytest.fixture
def client(patch_tables, patch_openai):
    from main import app

    with TestClient(app) as _client:
        yield _client
