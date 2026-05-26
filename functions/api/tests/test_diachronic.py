"""Tests for POST /diachronic-similarity/{book_id}.

Flow:
1. Validate book_id and ref_book_ids against _aligned_book_ids (pipeline table).
2. get_reference_centroid calls evaluate_tree per ref book via TermTable.
3. compute_diachronic_similarity calls evaluate_tree for the target book.
4. Cosine CI between target seeds and reference centroid.
"""

from conftest import make_term_entry
from schemas import DiachronicResult


def _set_term_table_for_books(term_table, entries_by_book):
    """entries_by_book: {"gutenberg-1": {"labour": <entry>, ...}, ...}"""
    def get_entry(term, platform_data, fields=None):
        book = entries_by_book.get(platform_data, {})
        return book.get(term)

    term_table.get_entry.side_effect = get_entry


def test_diachronic_happy_path(client, patch_storage):
    _, term_table = patch_storage
    _set_term_table_for_books(term_table, {
        "gutenberg-1": {"labour": make_term_entry("labour", seed=1)},
        "gutenberg-2": {"labour": make_term_entry("labour", seed=2)},
    })

    response = client.post(
        "/diachronic-similarity/1",
        json={"tree": {"term": "labour"}, "ref_book_ids": [2]},
    )

    assert response.status_code == 200
    result = DiachronicResult.model_validate(response.json())
    assert result.id == 1
    assert result.n > 0
    assert result.similarity_ci[0] <= result.similarity_ci[1]


def test_diachronic_multi_ref(client, patch_storage):
    _, term_table = patch_storage
    _set_term_table_for_books(term_table, {
        "gutenberg-1": {"labour": make_term_entry("labour", seed=1)},
        "gutenberg-2": {"labour": make_term_entry("labour", seed=2)},
    })

    response = client.post(
        "/diachronic-similarity/1",
        json={"tree": {"term": "labour"}, "ref_book_ids": [1, 2]},
    )

    assert response.status_code == 200


def test_diachronic_unaligned_book_returns_404(client, patch_storage):
    response = client.post(
        "/diachronic-similarity/99",
        json={"tree": {"term": "labour"}, "ref_book_ids": [1]},
    )
    assert response.status_code == 404


def test_diachronic_unaligned_ref_returns_422(client, patch_storage):
    response = client.post(
        "/diachronic-similarity/1",
        json={"tree": {"term": "labour"}, "ref_book_ids": [99]},
    )
    assert response.status_code == 422


def test_diachronic_missing_term_in_all_refs_returns_422(client, patch_storage):
    """No ref book has the term, so get_reference_centroid raises."""
    _, term_table = patch_storage
    term_table.get_entry.return_value = None

    response = client.post(
        "/diachronic-similarity/1",
        json={"tree": {"term": "missing"}, "ref_book_ids": [2]},
    )

    assert response.status_code == 422


def test_diachronic_missing_term_in_target_returns_404(client, patch_storage):
    """Ref has the term but target book does not."""
    _, term_table = patch_storage
    _set_term_table_for_books(term_table, {
        "gutenberg-2": {"labour": make_term_entry("labour", seed=2)},
        # gutenberg-1 has no terms
    })

    response = client.post(
        "/diachronic-similarity/1",
        json={"tree": {"term": "labour"}, "ref_book_ids": [2]},
    )

    assert response.status_code == 404


def test_diachronic_empty_ref_list_returns_422(client, patch_storage):
    response = client.post(
        "/diachronic-similarity/1",
        json={"tree": {"term": "labour"}, "ref_book_ids": []},
    )
    assert response.status_code == 422


def test_diachronic_missing_ref_field_returns_422(client, patch_storage):
    response = client.post(
        "/diachronic-similarity/1",
        json={"tree": {"term": "labour"}},
    )
    assert response.status_code == 422
