import csv
import io
import os
import json
import tempfile

from boto3 import Session
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError


AWS_REGION = os.getenv("AWS_REGION")
AWS_PROFILE = os.getenv("AWS_PROFILE")
S3_BUCKET = os.getenv("S3_BUCKET")
PIPELINE_TABLE = os.getenv("PIPELINE_TABLE")
TERM_TABLE = os.getenv("TERM_TABLE")


def get_session():
    return Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


class BaseTable:

    def __init__(self, session, table_name):
        self.table = session.resource("dynamodb").Table(table_name)

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

    def get_entry(self, key, fields=None):
        params = {"Key": key}
        if fields is not None:
            params["ProjectionExpression"] = ", ".join(fields)
        return self.table.get_item(**params).get("Item")


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


class TermTable(BaseTable):

    def __init__(self, session):
        super().__init__(session, TERM_TABLE)

    def update_entry(self, term, platform_data, field, value):
        super().update_entry(
            {"term": term, "platform_data": platform_data}, field, value
        )

    def update_entries(self, term, platform_data, data):
        super().update_entries({"term": term, "platform_data": platform_data}, data)

    def get_entry(self, term, platform_data, fields=None):
        return super().get_entry({"term": term, "platform_data": platform_data}, fields)

    def get_entries(self, platform_data, fields=None):
        params = {
            "IndexName": "platform_data-index",
            "KeyConditionExpression": Key("platform_data").eq(platform_data),
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


def upload_object(
    session, s3_key, file_bytes, content_type="text/plain; charset=utf-8"
):
    session.client("s3").upload_fileobj(
        io.BytesIO(file_bytes.encode("utf-8")),
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )


def upload_file(session, s3_key, path):
    session.client("s3").upload_file(
        path,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )


def yield_s3_files(session, s3_prefix, file_extension):
    s3_resource = session.resource("s3")
    bucket = s3_resource.Bucket(S3_BUCKET)
    for obj in bucket.objects.filter(Prefix=s3_prefix):
        if file_extension not in obj.key:
            continue

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            s3_resource.Object(S3_BUCKET, obj.key).download_fileobj(tmp_file)
            tmp_file.flush()
            try:
                yield obj.key, tmp_file.name
            finally:
                os.unlink(tmp_file.name)


def load_text_from_s3(session, s3_key):
    return (
        session.resource("s3")
        .Object(S3_BUCKET, s3_key)
        .get()["Body"]
        .read()
        .decode("utf-8")
    )


def yield_sentences_from_s3(session, s3_key):
    body = session.resource("s3").Object(S3_BUCKET, s3_key).get()["Body"]
    yield from csv.reader(io.TextIOWrapper(body, encoding="utf-8"))


def extract_index(event):
    if not isinstance(event, dict):
        return None

    if "index" in event:
        return event["index"]

    body = event.get("body")
    if not body:
        return None

    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(payload, dict):
        return payload.get("index")

    return None
