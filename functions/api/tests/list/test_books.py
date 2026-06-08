"""Tests for GET /books."""

import copy

from conftest import make_pipeline_entry
from app.list.schemas import BookResponse


def test_books_lists_aligned(client):
    response = client.get("/books")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {b["id"] for b in body} == {1, 2}


def test_books_excludes_unaligned(client, patch_tables):
    pipeline_table, _ = patch_tables
    custom = [
        make_pipeline_entry(1, "Smith, Adam", 1776, "Wealth", aligned=True),
        make_pipeline_entry(9, "Say, J-B", 1803, "Treatise", aligned=False),
    ]
    pipeline_table.get_all_entries.side_effect = lambda *a, **kw: copy.deepcopy(custom)

    body = client.get("/books").json()

    ids = {b["id"] for b in body}
    assert 1 in ids
    assert 9 not in ids


def test_books_response_validates_against_schema(client):
    body = client.get("/books").json()

    for book in body:
        BookResponse.model_validate(book)


def test_books_label_format(client):
    labels = {b["id"]: b["label"] for b in client.get("/books").json()}

    assert labels[1] == "Smith (1776)"
    assert labels[2] == "Ricardo (1817)"


def test_books_label_single_name_author(client, patch_tables):
    """Author without a comma uses the full name as label prefix."""
    pipeline_table, _ = patch_tables
    custom = [make_pipeline_entry(1, "Montesquieu", 1748, "Spirit of the Laws")]
    pipeline_table.get_all_entries.side_effect = lambda *a, **kw: copy.deepcopy(custom)

    book = client.get("/books").json()[0]

    assert book["label"] == "Montesquieu (1748)"


def test_books_empty(client, patch_tables):
    pipeline_table, _ = patch_tables
    pipeline_table.get_all_entries.side_effect = lambda *a, **kw: []

    response = client.get("/books")

    assert response.status_code == 200
    assert response.json() == []


def test_books_preserves_fields(client, patch_tables):
    pipeline_table, _ = patch_tables
    custom = [make_pipeline_entry(42, "Mill, John Stuart", 1848, "Principles of Political Economy")]
    pipeline_table.get_all_entries.side_effect = lambda *a, **kw: copy.deepcopy(custom)

    book = client.get("/books").json()[0]

    assert book["author"] == "Mill, John Stuart"
    assert book["title"] == "Principles of Political Economy"
    assert book["published_year"] == 1848
