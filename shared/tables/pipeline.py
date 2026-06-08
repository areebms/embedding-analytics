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

    def update_entry(self, platform_data, field, value):
        super().update_entry({"platform_data": platform_data}, field, value)

    def update_entries(self, platform_data, data):
        super().update_entries({"platform_data": platform_data}, data)

    def get_entry(self, platform_data, fields=["platform_data"]):
        return super().get_entry({"platform_data": platform_data}, fields)

    def put_entry(self, platform_data):
        item = {"platform_data": platform_data}
        try:
            self.table.put_item(
                Item=item, ConditionExpression="attribute_not_exists(platform_data)"
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def get_all_entries(self, fields=None):
        scan_kwargs = {}
        if fields:
            scan_kwargs = {
                "ProjectionExpression": ", ".join([f"#{field}" for field in fields]),
                "ExpressionAttributeNames": {f"#{field}": field for field in fields},
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
