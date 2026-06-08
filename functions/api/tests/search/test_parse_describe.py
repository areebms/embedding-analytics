"""Tests for POST /parse-describe."""

from pytest import fixture
from unittest.mock import MagicMock

from app.search.services.describe import get_vocabulary
from app.search.schemas.describe import ParseDescribeResponse


@fixture(autouse=True)
def clear_vocabulary_cache():
    get_vocabulary.cache_clear()
    yield


def mocked_openai(returns):
    stub = MagicMock()
    stub.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=returns))]
    )
    return lambda api_key : stub


def _set_vocabulary(term_table, terms):
    """Configure the term table so get_vocabulary picks up these terms.

    terms: list of term strings present in both books.
    """
    entries = [{"term": t, "tags": {"N"}} for t in terms]

    def get_entries(book_id, fields=None):
        return list(entries)

    term_table.get_entries.side_effect = get_entries


def test_parse_describe_happy_path(client, monkeypatch, patch_tables):
    _, term_table = patch_tables
    _set_vocabulary(term_table, ["labour", "value"])

    monkeypatch.setattr("app.search.services.describe.OpenAI", mocked_openai("labour"))

    response = client.post("/parse-describe", json={"message": "show me labour"})

    assert response.status_code == 200
    result = ParseDescribeResponse.model_validate(response.json())
    assert result.expression == "labour"
    assert result.terms == ["labour"]
    assert result.substitutions == []


def test_parse_describe_with_substitution(client, monkeypatch, patch_tables):
    """Fuzzy match produces a substitution."""
    _, term_table = patch_tables
    _set_vocabulary(term_table, ["labour"])

    monkeypatch.setattr("app.search.services.describe.OpenAI", mocked_openai("labor"))

    response = client.post("/parse-describe", json={"message": "show me labor"})

    assert response.status_code == 200
    result = ParseDescribeResponse.model_validate(response.json())
    assert result.expression == "labour"
    assert len(result.substitutions) == 1
    assert result.substitutions[0].original == "labor"
    assert result.substitutions[0].resolved == "labour"


def test_parse_describe_unparseable_returns_400(client, monkeypatch, patch_tables):
    _, term_table = patch_tables
    _set_vocabulary(term_table, ["labour"])

    monkeypatch.setattr("app.search.services.describe.OpenAI", mocked_openai(""))

    response = client.post("/parse-describe", json={"message": "nonsense"})

    assert response.status_code == 400


def test_parse_describe_unresolvable_returns_422(client, monkeypatch, patch_tables):
    _, term_table = patch_tables
    _set_vocabulary(term_table, ["labour"])

    monkeypatch.setattr("app.search.services.describe.OpenAI", mocked_openai("xyzzy"))
    response = client.post("/parse-describe", json={"message": "show me xyzzy"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["term"] == "xyzzy"
    assert "candidates" in detail


def test_parse_describe_missing_message_returns_422(client):
    response = client.post("/parse-describe", json={})
    assert response.status_code == 422


def test_parse_describe_llm_fallback_resolves(client, monkeypatch, patch_tables):
    """When fuzzy match is uncertain but candidates exist, LLM picks one."""
    _, term_table = patch_tables
    _set_vocabulary(term_table, ["labour", "value", "rent"])

    monkeypatch.setattr("app.search.services.describe.OpenAI", mocked_openai("labour"))

    response = client.post("/parse-describe", json={"message": "labourx"})

    assert response.status_code == 200
    result = ParseDescribeResponse.model_validate(response.json())
    assert result.expression == "labour"
