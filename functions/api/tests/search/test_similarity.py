"""Tests for POST /similarity/{book_id}.

Flow under test:
1. evaluate_tree resolves the query tree against BookTermTable per-seed bytes.
2. Every non-R term in the book is scored against the query via cosine CI.
3. Results returned as list[SimilarityResult].
"""

from conftest import make_term_entry
from app.search.schemas.search_expr import SimilarityResult


def _set_term_table(term_table, entries_by_term, book_id="gutenberg-1"):
    """Configure get_entry (for evaluate_tree) and get_entries (for scoring)."""
    def get_entry(term, platform_data, fields=None):
        if platform_data == book_id and term in entries_by_term:
            return entries_by_term[term]
        return None

    def get_entries(platform_data, fields=None):
        if platform_data == book_id:
            return list(entries_by_term.values())
        return []

    term_table.get_entry.side_effect = get_entry
    term_table.get_entries.side_effect = get_entries


def test_similarity_happy_path(client, patch_tables):
    _, term_table = patch_tables
    rows = {
        "labour": make_term_entry("labour", seed=1),
        "value": make_term_entry("value", seed=2),
    }
    _set_term_table(term_table, rows)

    response = client.post("/similarity/1", json={"tree": {"term": "labour"}})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    terms = {r["term"] for r in body}
    assert terms == {"labour", "value"}


def test_similarity_response_validates_against_schema(client, patch_tables):
    _, term_table = patch_tables
    rows = {"labour": make_term_entry("labour", seed=1)}
    _set_term_table(term_table, rows)

    body = client.post("/similarity/1", json={"tree": {"term": "labour"}}).json()

    for row in body:
        result = SimilarityResult.model_validate(row)
        assert len(result.similarity_ci) == 2
        assert result.similarity_ci[0] <= result.similarity_ci[1]


def test_similarity_skips_r_tagged_terms(client, patch_tables):
    """Terms with tags == {"R"} should not appear in results."""
    _, term_table = patch_tables
    rows = {
        "labour": make_term_entry("labour", seed=1),
        "removed": make_term_entry("removed", seed=2, tags={"R"}),
    }
    _set_term_table(term_table, rows)

    body = client.post("/similarity/1", json={"tree": {"term": "labour"}}).json()

    terms = {r["term"] for r in body}
    assert "labour" in terms
    assert "removed" not in terms


def test_similarity_compound_expression(client, patch_tables):
    _, term_table = patch_tables
    rows = {
        "labour": make_term_entry("labour", seed=1),
        "value": make_term_entry("value", seed=2),
    }
    _set_term_table(term_table, rows)

    response = client.post(
        "/similarity/1",
        json={"tree": {"op": "+", "args": [{"term": "labour"}, {"term": "value"}]}},
    )

    assert response.status_code == 200


def test_similarity_unknown_term_returns_404(client, patch_tables):
    _, term_table = patch_tables
    term_table.get_entry.return_value = None

    response = client.post("/similarity/1", json={"tree": {"term": "missing"}})

    assert response.status_code == 404


def test_similarity_invalid_payload_returns_422(client, patch_tables):
    response = client.post("/similarity/1", json={})
    assert response.status_code == 422


def test_similarity_invalid_op_returns_422(client, patch_tables):
    response = client.post(
        "/similarity/1",
        json={"tree": {"op": "*", "args": [{"term": "a"}, {"term": "b"}]}},
    )
    assert response.status_code == 422


def test_similarity_blank_term_returns_422(client, patch_tables):
    response = client.post("/similarity/1", json={"tree": {"term": "   "}})
    assert response.status_code == 422
