"""Tests for POST /similar-terms/quick/{book_id} (ranked list, Pinecone-only).

evaluate_tree resolves the query tree to per-seed vectors; their mean is sent to
Pinecone, which returns the nearest terms with pos + count from metadata. The
per-seed query vectors are returned alongside the results so the caller can feed
them into POST /similar-terms/detailed/{book_id}/.
"""

from conftest import make_term_entry, set_term_table, set_pinecone_matches
from app.search.schemas.search_expr import SimilarityResult


def test_similarity_happy_path(client, patch_tables, mock_pinecone_table):
    _, term_table = patch_tables
    rows = {
        "labour": make_term_entry("labour", seed=1),
        "value": make_term_entry("value", seed=2),
    }
    set_term_table(term_table, rows)
    set_pinecone_matches(mock_pinecone_table, rows)

    response = client.post("/similar-terms/quick/1", json={"tree": {"term": "labour"}})

    assert response.status_code == 200
    body = response.json()
    assert {r["term"] for r in body["results"]} == {"labour", "value"}


def test_similarity_returns_query_vectors(client, patch_tables, mock_pinecone_table):
    _, term_table = patch_tables
    rows = {"labour": make_term_entry("labour", n_seeds=5, dim=4, seed=1)}
    set_term_table(term_table, rows)
    set_pinecone_matches(mock_pinecone_table, rows)

    body = client.post("/similar-terms/quick/1", json={"tree": {"term": "labour"}}).json()

    vectors = body["query_vectors"]
    assert len(vectors) == 5  # n_seeds
    assert all(len(row) == 4 for row in vectors)  # dim


def test_similarity_result_shape(client, patch_tables, mock_pinecone_table):
    _, term_table = patch_tables
    rows = {"labour": make_term_entry("labour", seed=1, count=42)}
    set_term_table(term_table, rows)
    set_pinecone_matches(mock_pinecone_table, rows)

    body = client.post("/similar-terms/quick/1", json={"tree": {"term": "labour"}}).json()

    result = SimilarityResult.model_validate(body["results"][0])
    assert result.term == "labour"
    assert result.count == 42
    assert result.similarity == 1.0  # pinecone score from the stub
    assert "similarity_ci" not in body["results"][0]


def test_similarity_compound_expression(client, patch_tables, mock_pinecone_table):
    _, term_table = patch_tables
    rows = {
        "labour": make_term_entry("labour", seed=1),
        "value": make_term_entry("value", seed=2),
    }
    set_term_table(term_table, rows)
    set_pinecone_matches(mock_pinecone_table, rows)

    response = client.post(
        "/similar-terms/quick/1",
        json={"tree": {"op": "+", "args": [{"term": "labour"}, {"term": "value"}]}},
    )

    assert response.status_code == 200


def test_similarity_respects_top_k(client, patch_tables, mock_pinecone_table):
    _, term_table = patch_tables
    rows = {f"term{i}": make_term_entry(f"term{i}", seed=i) for i in range(5)}
    rows["labour"] = make_term_entry("labour", seed=9)
    set_term_table(term_table, rows)
    set_pinecone_matches(mock_pinecone_table, rows)

    body = client.post(
        "/similar-terms/quick/1", json={"tree": {"term": "labour"}, "top_k": 2}
    ).json()

    assert len(body["results"]) <= 2


def test_similarity_top_k_over_cap_returns_422(client, patch_tables):
    response = client.post(
        "/similar-terms/quick/1", json={"tree": {"term": "labour"}, "top_k": 9999}
    )
    assert response.status_code == 422


def test_similarity_unknown_term_returns_404(client, patch_tables):
    _, term_table = patch_tables
    term_table.get_entry.return_value = None

    response = client.post("/similar-terms/quick/1", json={"tree": {"term": "missing"}})

    assert response.status_code == 404


def test_similarity_invalid_payload_returns_422(client, patch_tables):
    assert client.post("/similar-terms/quick/1", json={}).status_code == 422


def test_similarity_invalid_op_returns_422(client, patch_tables):
    response = client.post(
        "/similar-terms/quick/1",
        json={"tree": {"op": "*", "args": [{"term": "a"}, {"term": "b"}]}},
    )
    assert response.status_code == 422


def test_similarity_blank_term_returns_422(client, patch_tables):
    response = client.post("/similar-terms/quick/1", json={"tree": {"term": "   "}})
    assert response.status_code == 422
