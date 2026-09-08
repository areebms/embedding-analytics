from botocore.exceptions import ClientError


class BaseTable:

    def __init__(self, session, table_name):
        self.dynamodb = session.resource("dynamodb")
        self.table = self.dynamodb.Table(table_name)

    def update_entry(self, key, field, value):
        self.table.update_item(
            Key=key,
            UpdateExpression=f"SET #{field} = :{field}",
            ExpressionAttributeNames={f"#{field}": field},
            ExpressionAttributeValues={f":{field}": value},
        )

    def update_entries(self, key, data, condition=None, condition_values=None):
        """Write `data` onto one row. Returns False when `condition` rejected the
        write, mirroring `PipelineTable.put_entry`, so a guard that turns a write
        down is an ordinary return value rather than an exception."""
        values = {f":{field}": value for field, value in data.items()}
        params = {
            "Key": key,
            "UpdateExpression": "SET "
            + ", ".join(f"#{field} = :{field}" for field in data),
            "ExpressionAttributeNames": {f"#{field}": field for field in data},
        }
        if condition is not None:
            params["ConditionExpression"] = condition
            values.update(condition_values or {})
        params["ExpressionAttributeValues"] = values

        try:
            self.table.update_item(**params)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def add_to_set(self, key, field, values):
        """`ADD` members onto a set attribute. Atomic and idempotent, so re-adding a
        member the row already carries is a no-op and needs no read first -- which is
        the whole reason `subject_ids` is a set and not a list."""
        self.table.update_item(
            Key=key,
            UpdateExpression=f"ADD #{field} :{field}",
            ExpressionAttributeNames={f"#{field}": field},
            ExpressionAttributeValues={f":{field}": values},
        )

    def batch_put_entries(self, items):
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)

    def batch_delete_entries(self, keys):
        with self.table.batch_writer() as batch:
            for key in keys:
                batch.delete_item(Key=key)

    def list_all(self, **params):
        items = []
        while True:
            response = self.table.query(**params)
            items.extend(response.get("Items", []))
            if "LastEvaluatedKey" not in response:
                break
            params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        return items

    def get_entry(self, key, fields=None):
        params = {"Key": key}
        if fields is not None:
            params["ProjectionExpression"] = ", ".join(f"#{field}" for field in fields)
            params["ExpressionAttributeNames"] = {f"#{field}": field for field in fields}
        return self.table.get_item(**params).get("Item")

    def batch_get_entries(self, keys, fields=None):
        if not keys:
            return []
        projection = {
            "ProjectionExpression": ", ".join(f"#{field}" for field in fields),
            "ExpressionAttributeNames": {f"#{field}": field for field in fields},
        } if fields else {}
        items = []
        for start in range(0, len(keys), 100):
            request = {self.table.name: {"Keys": keys[start : start + 100], **projection}}
            while request:
                response = self.dynamodb.batch_get_item(RequestItems=request)
                items.extend(response["Responses"].get(self.table.name, []))
                request = response.get("UnprocessedKeys") or None
        return items
