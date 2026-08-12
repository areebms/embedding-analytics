"""Tests for GET /terms."""

from app.list.schemas import TermResponse


def _set_term_entries(term_table, entries_by_book):
    """Configure mock_term_table.get_entries per book_id.

    entries_by_book: {"gutenberg-1": [{"term": "labour", "tags": {"N"}}, ...]}
    """
    def get_entries(book_id, fields=None):
        return entries_by_book.get(book_id, [])

    term_table.get_entries.side_effect = get_entries


def test_terms_lists_multi_book_terms(client, patch_tables):
    """Only terms appearing in more than one book are returned."""
    _, term_table = patch_tables
    _set_term_entries(term_table, {
        "gutenberg-1": [
            {"term": "labour", "tags": {"N"}},
            {"term": "smith_only", "tags": {"N"}},
        ],
        "gutenberg-2": [
            {"term": "labour", "tags": {"N"}},
        ],
    })

    body = client.get("/terms").json()

    terms = {item["term"] for item in body}
    assert "labour" in terms
    assert "smith_only" not in terms


def test_terms_excludes_pos_tag_r(client, patch_tables):
    """Entries tagged {"R"} (adverb-only) are skipped."""
    _, term_table = patch_tables
    _set_term_entries(term_table, {
        "gutenberg-1": [
            {"term": "labour", "tags": {"N"}},
            {"term": "removed", "tags": {"R"}},
        ],
        "gutenberg-2": [
            {"term": "labour", "tags": {"N"}},
            {"term": "removed", "tags": {"R"}},
        ],
    })

    terms = {item["term"] for item in client.get("/terms").json()}

    assert "labour" in terms
    assert "removed" not in terms


def test_terms_response_validates_against_schema(client, patch_tables):
    _, term_table = patch_tables
    _set_term_entries(term_table, {
        "gutenberg-1": [{"term": "labour", "tags": {"N"}}],
        "gutenberg-2": [{"term": "labour", "tags": {"N"}}],
    })

    for entry in client.get("/terms").json():
        TermResponse.model_validate(entry)


def test_terms_books_field_contains_source_ids(client, patch_tables):
    _, term_table = patch_tables
    _set_term_entries(term_table, {
        "gutenberg-1": [{"term": "labour", "tags": {"N"}}],
        "gutenberg-2": [{"term": "labour", "tags": {"N"}}],
    })

    entry = client.get("/terms").json()[0]

    assert set(entry["books"]) == {1, 2}


def test_terms_empty_corpus(client, patch_tables):
    response = client.get("/terms")

    assert response.status_code == 200
    assert response.json() == []
