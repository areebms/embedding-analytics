class BaseTable:

    def __init__(self, session, table_name):
        self.dynamodb = session.resource("dynamodb")
        self.table = self.dynamodb.Table(table_name)

    def update_entry(self, key, field, value):
        self.table.update_item(
            Key=key,
            UpdateExpression=f"SET {field} = :{field}",
            ExpressionAttributeValues={f":{field}": value},
        )

    def update_entries(self, key, data):
        self.table.update_item(
            Key=key,
            UpdateExpression="SET "
            + ", ".join(f"{field} = :{field}" for field in data),
            ExpressionAttributeValues={
                f":{field}": value for field, value in data.items()
            },
        )

    def batch_put_entries(self, items):
        with self.table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)

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
            params["ProjectionExpression"] = ", ".join(fields)
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
