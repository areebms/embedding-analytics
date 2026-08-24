import os

from botocore.exceptions import ClientError

from shared.tables.base import BaseTable
from shared.session import get_session

PIPELINE_TABLE = os.getenv("PIPELINE_TABLE")

_pipeline_table = None


def get_pipeline_table():
    global _pipeline_table
    if _pipeline_table is None:
        _pipeline_table = PipelineTable(get_session())
    return _pipeline_table


class PipelineTable(BaseTable):

    def __init__(self, session):
        super().__init__(session, PIPELINE_TABLE)

    def update_entry(self, book_id, field, value):
        super().update_entry({"book_id": book_id}, field, value)

    def update_entries(self, book_id, data, condition=None, condition_values=None):
        return super().update_entries(
            {"book_id": book_id}, data, condition, condition_values
        )

    def add_to_set(self, book_id, field, values):
        super().add_to_set({"book_id": book_id}, field, values)

    def get_entry(self, book_id, fields=["book_id"]):
        return super().get_entry({"book_id": book_id}, fields)

    def put_entry(self, book_id, attributes=None):
        item = {"book_id": book_id, **(attributes or {})}
        try:
            self.table.put_item(
                Item=item, ConditionExpression="attribute_not_exists(book_id)"
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def get_all_entries(self, fields=None, **scan_kwargs):
        """Every row, optionally projected and optionally filtered.
        """
        scan_kwargs = dict(scan_kwargs)
        if fields:
            scan_kwargs["ProjectionExpression"] = ", ".join(
                [f"#{field}" for field in fields]
            )
            scan_kwargs["ExpressionAttributeNames"] = {
                **scan_kwargs.get("ExpressionAttributeNames", {}),
                **{f"#{field}": field for field in fields},
            }
        items = []
        response = self.table.scan(**scan_kwargs)
        while True:
            items.extend(response.get("Items", []))
            if "LastEvaluatedKey" not in response:
                break
            response = self.table.scan(
                ExclusiveStartKey=response["LastEvaluatedKey"],
                **scan_kwargs,
            )
        return items
