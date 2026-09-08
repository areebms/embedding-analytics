from unittest.mock import MagicMock

import pytest
from boto3.dynamodb.conditions import Key

from app.core.tables import BookTermTable
from shared.commons import BookIndex


@pytest.fixture
def table():
    session = MagicMock()
    book_term_table = BookTermTable(session)
    book_term_table.table.name = "BookTerms"
    return book_term_table


def test_get_entries_queries_the_deployed_index(table):
    table.table.query.return_value = {"Items": [{"term": "labour"}]}

    items = table.get_entries(BookIndex(3300), fields=["term", "tags"])

    params = table.table.query.call_args.kwargs
    assert params["IndexName"] == "platform_data-index"
    assert params["KeyConditionExpression"] == Key("platform_data").eq(
        "gutenberg-3300"
    )
    assert params["ExpressionAttributeNames"] == {"#term": "term", "#tags": "tags"}
    assert items == [{"term": "labour"}]


def test_get_entries_follows_the_pagination_cursor(table):
    table.table.query.side_effect = [
        {"Items": [{"term": "a"}], "LastEvaluatedKey": {"term": "a"}},
        {"Items": [{"term": "b"}]},
    ]

    items = table.get_entries(BookIndex(3300))

    assert items == [{"term": "a"}, {"term": "b"}]
    assert table.table.query.call_args.kwargs["ExclusiveStartKey"] == {"term": "a"}


def test_get_entry_keys_on_the_deployed_sort_key(table):
    table.table.get_item.return_value = {"Item": {"term": "labour"}}

    entry = table.get_entry("labour", BookIndex(3300))

    assert table.table.get_item.call_args.kwargs["Key"] == {
        "term": "labour",
        "platform_data": "gutenberg-3300",
    }
    assert entry == {"term": "labour"}


def test_batch_get_entries_keys_on_the_deployed_sort_key(table):
    table.dynamodb.batch_get_item.return_value = {
        "Responses": {"BookTerms": [{"term": "labour"}]}
    }

    entries = table.batch_get_entries(["labour", "value"], BookIndex(3300))

    request = table.dynamodb.batch_get_item.call_args.kwargs["RequestItems"]
    assert request["BookTerms"]["Keys"] == [
        {"term": "labour", "platform_data": "gutenberg-3300"},
        {"term": "value", "platform_data": "gutenberg-3300"},
    ]
    assert entries == [{"term": "labour"}]
