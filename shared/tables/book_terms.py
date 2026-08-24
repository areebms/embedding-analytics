import os

from boto3.dynamodb.conditions import Key

from shared.session import get_session
from shared.tables.base import BaseTable

BOOK_TERM_TABLE = os.getenv("BOOK_TERM_TABLE")

ADVERB_TAGS = {"R"}


_book_term_table = None


def get_book_term_table():
    global _book_term_table
    if _book_term_table is None:
        _book_term_table = BookTermTable(get_session())
    return _book_term_table


class BookTermTable(BaseTable):

    def __init__(self, session):
        super().__init__(session, BOOK_TERM_TABLE)

    def update_entry(self, term, book_id, field, value):
        super().update_entry(
            {"term": term, "book_id": book_id}, field, value
        )

    def update_entries(self, term, book_id, data):
        super().update_entries({"term": term, "book_id": book_id}, data)

    def get_entry(self, term, book_id, fields=None):
        return super().get_entry({"term": term, "book_id": book_id}, fields)

    def get_entries(self, book_id, fields=None):
        params = {
            "IndexName": "book_id-index",
            "KeyConditionExpression": Key("book_id").eq(book_id),
        }
        if fields:
            params["ProjectionExpression"] = ", ".join(f"#{f}" for f in fields)
            params["ExpressionAttributeNames"] = {f"#{f}": f for f in fields}
        items = []
        while True:
            response = self.table.query(**params)
            items.extend(response.get("Items", []))
            if "LastEvaluatedKey" not in response:
                break
            params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        return items

    def batch_get_entries(self, terms, book_id, fields=None):
        return super().batch_get_entries(
            [{"term": term, "book_id": book_id} for term in terms], fields
        )
