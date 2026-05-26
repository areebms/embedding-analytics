"""Tests for POST /parse-describe."""

from unittest.mock import MagicMock

from schemas import ParseChatResponse


def _stub_openai_to_return(text):
    stub = MagicMock()
    stub.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=text))]
    )
    return stub


def _set_vocabulary(term_table, terms):
    """Configure the term table so get_vocabulary picks up these terms.

    terms: list of term strings present in both books.
    """
    entries = [{"term": t, "tags": {"N"}} for t in terms]

    def get_entries(book_id, fields=None):
        return list(entries)

    term_table.get_entries.side_effect = get_entries


def test_parse_describe_happy_path(client, monkeypatch, patch_storage):
    _, term_table = patch_storage
    _set_vocabulary(term_table, ["labour", "value"])

    import describe_services
    describe_services.get_vocabulary.cache_clear()

    stub = _stub_openai_to_return("labour")
    monkeypatch.setattr(describe_services, "_get_openai_client", lambda: stub)

    response = client.post("/parse-describe", json={"message": "show me labour"})

    assert response.status_code == 200
    result = ParseChatResponse.model_validate(response.json())
    assert result.expression == "labour"
    assert result.terms == ["labour"]
    assert result.substitutions == []


def test_parse_describe_with_substitution(client, monkeypatch, patch_storage):
    """Fuzzy match produces a substitution."""
    _, term_table = patch_storage
    _set_vocabulary(term_table, ["labour"])

    import describe_services
    describe_services.get_vocabulary.cache_clear()

    stub = _stub_openai_to_return("labor")
    monkeypatch.setattr(describe_services, "_get_openai_client", lambda: stub)

    response = client.post("/parse-describe", json={"message": "show me labor"})

    assert response.status_code == 200
    result = ParseChatResponse.model_validate(response.json())
    assert result.expression == "labour"
    assert len(result.substitutions) == 1
    assert result.substitutions[0].original == "labor"
    assert result.substitutions[0].resolved == "labour"


def test_parse_describe_unparseable_returns_400(client, monkeypatch, patch_storage):
    _, term_table = patch_storage
    _set_vocabulary(term_table, ["labour"])

    import describe_services
    describe_services.get_vocabulary.cache_clear()

    stub = _stub_openai_to_return("")
    monkeypatch.setattr(describe_services, "_get_openai_client", lambda: stub)

    response = client.post("/parse-describe", json={"message": "nonsense"})

    assert response.status_code == 400


def test_parse_describe_unresolvable_returns_422(client, monkeypatch, patch_storage):
    _, term_table = patch_storage
    _set_vocabulary(term_table, ["labour"])

    import describe_services
    describe_services.get_vocabulary.cache_clear()

    stub = _stub_openai_to_return("xyzzy")
    monkeypatch.setattr(describe_services, "_get_openai_client", lambda: stub)

    response = client.post("/parse-describe", json={"message": "show me xyzzy"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["term"] == "xyzzy"
    assert "candidates" in detail


def test_parse_describe_missing_message_returns_422(client, patch_storage):
    response = client.post("/parse-describe", json={})
    assert response.status_code == 422


def test_parse_describe_llm_fallback_resolves(client, monkeypatch, patch_storage):
    """When fuzzy match is uncertain but candidates exist, LLM picks one."""
    _, term_table = patch_storage
    _set_vocabulary(term_table, ["labour", "value", "rent"])

    import describe_services
    describe_services.get_vocabulary.cache_clear()

    # First call (generates expression) returns "labourx".
    # Second call (resolves it via LLM fallback) returns "labour".
    stub = MagicMock()
    stub.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="labourx"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="labour"))]),
    ]
    monkeypatch.setattr(describe_services, "_get_openai_client", lambda: stub)

    response = client.post("/parse-describe", json={"message": "labourx"})

    assert response.status_code == 200
    result = ParseChatResponse.model_validate(response.json())
    assert result.expression == "labour"
