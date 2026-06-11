"""Tests for POST /similar-terms/detailed/{book_id}/ (per-term CI).

Accepts {terms, query_vectors}. query_vectors are obtained from
POST /similar-terms/quick/{book_id} and fed in directly.
"""

from conftest import make_term_entry, set_term_table
from app.search.schemas.search_expr import ConfidenceResult


def test_confidence_happy_path(client, patch_tables):
    _, term_table = patch_tables
    rows = {
        "labour": make_term_entry("labour", seed=1),
        "value": make_term_entry("value", seed=2),
    }
    set_term_table(term_table, rows)

    fast = client.post("/similar-terms/quick/1", json={"tree": {"term": "labour"}}).json()
    response = client.post(
        "/similar-terms/detailed/1/",
        json={"query_vectors": fast["query_vectors"], "terms": ["labour", "value"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    for row in body:
        result = ConfidenceResult.model_validate(row)
        assert len(result.similarity_ci) == 2
        assert result.similarity_ci[0] <= result.similarity_ci[1]


def test_confidence_omits_absent_terms(client, patch_tables):
    _, term_table = patch_tables
    rows = {"labour": make_term_entry("labour", seed=1)}
    set_term_table(term_table, rows)

    fast = client.post("/similar-terms/quick/1", json={"tree": {"term": "labour"}}).json()
    body = client.post(
        "/similar-terms/detailed/1/",
        json={"query_vectors": fast["query_vectors"], "terms": ["labour", "ghost"]},
    ).json()

    assert [r["term"] for r in body] == ["labour"]


def test_confidence_single_seed_has_zero_ci_width(client, patch_tables):
    _, term_table = patch_tables
    rows = {"labour": make_term_entry("labour", n_seeds=1, seed=1)}
    set_term_table(term_table, rows)

    fast = client.post("/similar-terms/quick/1", json={"tree": {"term": "labour"}}).json()
    body = client.post(
        "/similar-terms/detailed/1/",
        json={"query_vectors": fast["query_vectors"], "terms": ["labour"]},
    ).json()

    assert body[0]["similarity_ci"][0] == body[0]["similarity_ci"][1]


def test_confidence_requires_query_vectors(client, patch_tables):
    assert (
        client.post(
            "/similar-terms/detailed/1/", json={"terms": ["labour"]}
        ).status_code
        == 422
    )
